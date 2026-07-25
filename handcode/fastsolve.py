"""Batched solver for the value code -- same maths as `linsolve`, ~20x faster.

Three observations turn the reference implementation's per-token Python loop
into a couple of batched BLAS calls.

**1. The solve does not depend on T0.** Every design row has exactly `k` ones,
so with `baseline = T0 / k` the right-hand side collapses:

    wanted - baseline * (row sum)  =  (T0 + l + 1) - (T0/k) * k  =  l + 1

T0 therefore never enters the linear system -- it only shifts the solution by
the constant `T0 / k` afterwards. The whole T0 sweep costs one solve, not one
per value, and the sweep is really just asking "how large can T0 be before
float32 stops resolving +-1/2 against a sum of that size".

**2. Ridge has a primal and a dual form, and they are equal.**

    (A^T A + mu I)^-1 A^T  ==  A^T (A A^T + mu I)^-1

so the same solve serves the under- and over-determined regimes; we pick
whichever of the two matrices is smaller and batch it.

**3. The per-token problems can be padded into one tensor.** Facts are grouped
by second token once per fact set; dropping facts only flips entries of a
validity mask, so the grouping is never rebuilt.
"""

from dataclasses import dataclass

import torch

from .model import ModelShape


@dataclass(frozen=True)
class Grouping:
    """Facts bucketed by second token, padded to a rectangle. Built once."""

    index: torch.Tensor  # (n_vocab, width) long -- fact ids, padded with 0
    valid: torch.Tensor  # (n_vocab, width) bool -- which entries are real
    first: torch.Tensor  # (n_vocab, width) long -- first token of each entry
    wanted: torch.Tensor  # (n_vocab, width) float64 -- target l + 1

    @property
    def width(self) -> int:
        return self.index.shape[1]


def group_facts(facts: dict, n_vocab: int) -> Grouping:
    inputs, targets = facts["inputs"], facts["targets"]
    second = inputs[:, 1]
    counts = torch.bincount(second, minlength=n_vocab)
    width = int(counts.max().item()) if len(counts) else 0

    order = torch.argsort(second, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))
    slot = ranks - starts[second]  # position of each fact within its own bucket

    index = torch.zeros(n_vocab, width, dtype=torch.long)
    valid = torch.zeros(n_vocab, width, dtype=torch.bool)
    index[second, slot] = torch.arange(len(second))
    valid[second, slot] = True

    return Grouping(
        index=index,
        valid=valid,
        first=inputs[:, 0][index] * valid,
        wanted=((targets + 1).double()[index]) * valid,
    )


def solve_x(
    grouping: Grouping,
    mask: torch.Tensor,
    live: torch.Tensor,
    n_value: int,
    mu: float,
) -> torch.Tensor:
    """Batched ridge solve of `sum_{i in S_a} x_i[b] = l + 1`; returns (n_value, n_vocab).

    This is the T0-free core: add `T0 / k` to every entry afterwards to get the
    actual embedding values.
    """
    keep = grouping.valid & live[grouping.index]  # (n_vocab, width)
    design_all = mask[grouping.first] * keep.unsqueeze(-1)  # (n_vocab, width, n_value)
    rhs_all = grouping.wanted * keep  # (n_vocab, width)

    # The two ridge forms are equal in exact arithmetic but NOT in floating
    # point: with a tiny mu the primal squares the condition number when a
    # token's system is under-determined (A^T A singular), and the dual does the
    # same when it is over-determined (A A^T singular). So the choice has to be
    # made per token, not once from the global width -- getting this wrong is
    # silent and large (errors of ~1e2 against a decode tolerance of 1/2).
    under = keep.sum(1) <= n_value
    out = torch.zeros(n_value, grouping.index.shape[0], dtype=design_all.dtype)

    for tokens, use_dual in ((under.nonzero().flatten(), True),
                             ((~under).nonzero().flatten(), False)):
        if len(tokens) == 0:
            continue
        design = design_all[tokens]
        rhs = rhs_all[tokens]
        if use_dual:
            gram = design @ design.transpose(1, 2)
            gram.diagonal(dim1=1, dim2=2).add_(mu)
            dual = torch.linalg.solve(gram, rhs.unsqueeze(-1))
            out[:, tokens] = (design.transpose(1, 2) @ dual).squeeze(-1).T
        else:
            gram = design.transpose(1, 2) @ design
            gram.diagonal(dim1=1, dim2=2).add_(mu)
            solved = torch.linalg.solve(gram, design.transpose(1, 2) @ rhs.unsqueeze(-1))
            out[:, tokens] = solved.squeeze(-1).T
    return out


def solve_profile(
    grouping: Grouping,
    mask: torch.Tensor,
    n_value: int,
    mu: float,
    keep_frac: float,
    rounds: int,
    pertoken: bool,
    n_facts: int,
) -> torch.Tensor:
    """Run the greedy drop schedule and return the T0-free solution `x`.

    The drop decision uses the *unclamped* decode, which is what makes the whole
    schedule independent of T0: clamping is the only place T0 could matter, and
    a well-chosen T0 is precisely one large enough that no clamping happens.
    """
    live = torch.ones(n_facts, dtype=torch.bool)
    if pertoken:
        # each second token can satisfy at most n_value of its facts exactly
        excess = grouping.valid.cumsum(1) > n_value
        live[grouping.index[excess & grouping.valid]] = False

    x = solve_x(grouping, mask, live, n_value, mu)
    for _ in range(rounds - 1):
        decoded = (mask[grouping.first] * x.T.unsqueeze(1)).sum(-1)  # (n_vocab, width)
        flat = (decoded - grouping.wanted).abs()
        cutoff = torch.quantile(flat[grouping.valid], keep_frac)
        live = torch.zeros(n_facts, dtype=torch.bool)
        ok = grouping.valid & (flat <= cutoff)
        live[grouping.index[ok]] = True
        x = solve_x(grouping, mask, live, n_value, mu)
    return x


def assemble(
    shape: ModelShape, mask: torch.Tensor, x: torch.Tensor, k: int, t0: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a T0-free solution into the two weight matrices."""
    n_vocab, d_mlp = shape.input_vocab_size, shape.d_mlp
    n_value = d_mlp - 1
    values = (x + t0 / k).clamp(min=0.0)

    big = float(values.max()) + 1.0
    up = torch.empty(d_mlp, 2 * n_vocab, dtype=torch.float64)
    up[:n_value, :n_vocab] = torch.where(mask.T.bool(), 0.0, -big)
    up[:n_value, n_vocab:] = values
    up[n_value, :] = 1.0

    labels = torch.arange(shape.output_vocab_size, dtype=torch.float64) + 1
    down = torch.zeros(shape.output_vocab_size, d_mlp, dtype=torch.float64)
    down[:, :n_value] = labels.unsqueeze(1)
    down[:, n_value] = -labels * t0 / 2 - labels**2 / 4
    return up.float(), down.float()


# ── caches ────────────────────────────────────────────────────────────────────
# The grouping depends only on the fact set, and the selector mask only on
# (vocab, width, k, seed) -- both are reused across the whole hyperparameter
# sweep, so building them once per capacity probe removes them from the inner loop.
_GROUPING_CACHE: dict[tuple, Grouping] = {}
_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def cached_grouping(facts: dict, shape: ModelShape) -> Grouping:
    key = (len(facts["targets"]), shape.input_vocab_size, shape.output_vocab_size)
    if key not in _GROUPING_CACHE:
        if len(_GROUPING_CACHE) > 8:
            _GROUPING_CACHE.clear()
        _GROUPING_CACHE[key] = group_facts(facts, shape.input_vocab_size)
    return _GROUPING_CACHE[key]


def cached_mask(n_vocab: int, n_value: int, k: int, seed: int) -> torch.Tensor:
    from .linsolve import selector_masks

    key = (n_vocab, n_value, k, seed)
    if key not in _MASK_CACHE:
        if len(_MASK_CACHE) > 64:
            _MASK_CACHE.clear()
        _MASK_CACHE[key] = selector_masks(
            n_vocab, n_value, k, torch.Generator().manual_seed(seed)
        )
    return _MASK_CACHE[key]
