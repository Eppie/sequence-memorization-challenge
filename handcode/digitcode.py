"""A redundant value code: the label's digits carried by disjoint neuron groups.

`twosided` stores a fact's label as one number -- the sum of all active units
equals `t0 + (l+1)` -- which packs all `log2(d)` bits of the label into a
single coordinate read at a precision of one part in `d`. That is where its
fragility lives: FINDINGS.md 5 measures it dying at weight noise three orders
of magnitude below anything gradient descent produces, and the trained scaling
law `~d^2/log d` (against the value codes' `~d^2`) is what "you cannot put
log d bits in one number robustly" looks like as an exponent.

This module spreads the label over `m` coordinates constructively. Write the
label in base `p = ceil(d^(1/m))`, partition the `d - 1` value neurons into
`m` groups, and require each group's own active-sum to carry one digit:

    sum_{i in G_j} h_i  =  t0_j + (digit_j(l) + 1) * delta      (j = 1..m)

`m` equations per fact instead of one, so the capacity ceiling divides to
`~4d^2 / m` -- and each coordinate now holds only `p` levels instead of `d`,
so the code should tolerate `~d/p` times more noise per coordinate. At
`m = 1` this is exactly `twosided`; at `m = log2(d)` each group is binary.
Decoding is `m` independent quadratic decodes summed into one linear readout
(nearest digit per group, the bias neuron carrying the per-class constants),
so the model is still just two matrices.

Structurally each group is a self-contained `twosided` system: its targets sit
near its own stabilizer `t0_j`, its readout direction is its own ones vector,
and the one-sided ReLU error that motivates the capped step and the fitted
shift applies group by group. The solver below is therefore the same
freeze-the-pattern / per-token ridge / capped-step loop, with each token's
block stacking the `m` group-masked copies of its pattern rows. No gradient
of any loss is computed anywhere.

The question this family exists to answer: sweeping `m`, does the (capacity,
noise-tolerance) frontier pass through the trained model's point? If some `m`
matches trained capacity *and* trained sigma90 simultaneously, "what gradient
descent builds" has a constructive description -- a graded code whose ~log d
digits ride on neuron subpopulations. If no `m` reaches it, the shortfall
measures what equality-constrained codes cannot buy: the packing advantage of
satisfying margin *inequalities*, the game gradient descent's implicit bias
plays.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ModelShape
from .twosided import _capped_step, init_embeddings, pad_group

MU_VALUES = (1e-2, 1.0)
GROUP_SEEDS = (0, 1)


@dataclass(frozen=True)
class DigitCodeParams:
    m: int  # digits per label = neuron groups = equations per fact
    rho: float = 0.53
    mu: float = 1e-2
    rounds: int = 400
    sweeps: int = 4
    flip_cap: float = 0.02
    t0_scale: float = 16.0  # total stabilizer, in units of d, split across groups
    delta: float = 1.0  # spacing between adjacent digit levels
    group_seed: int = 0  # which random partition of the neurons into groups
    dtype: torch.dtype = torch.float32

    def __str__(self) -> str:
        return f"m={self.m}, rho={self.rho:g}, mu={self.mu:g}, part={self.group_seed}"


def digit_codebook(n_labels: int, m: int) -> tuple[torch.Tensor, int]:
    """(n_labels, m) base-p digits of each label, and p itself."""
    p = 2
    while p**m < n_labels:
        p += 1
    if m == 1:
        p = n_labels
    labels = torch.arange(n_labels)
    digits = torch.stack([(labels // p**j) % p for j in range(m)], dim=1)
    return digits.float(), p


def make_groups(n_value: int, m: int, generator: torch.Generator) -> torch.Tensor:
    """(m, n_value) 0/1 masks: a random near-equal partition of the neurons."""
    order = torch.randperm(n_value, generator=generator)
    masks = torch.zeros(m, n_value)
    for j in range(m):
        masks[j, order[j::m]] = 1.0
    return masks


class WeightedBlocks:
    """Per-token ridge systems for one frozen pattern, all equations stacked.

    Token `a`'s design has one row per (fact of a, group): the fact's frozen
    pattern row masked to that group's neurons. Solved in the primal with a
    float64 factorization, as in `twosided.TokenBlocks`.
    """

    def __init__(
        self,
        group,
        pattern: torch.Tensor,
        eq_weights: torch.Tensor,  # (m, n_value) group masks
        mu: float,
        n_keys: int,
        n_value: int,
        chunk: int = 64,
    ) -> None:
        self.group = group
        self.n_keys = n_keys
        self.n_value = n_value
        self.dtype = pattern.dtype
        self.n_eq = eq_weights.shape[0]
        self.chunks: list[tuple] = []

        for start in range(0, n_keys, chunk):
            stop = min(start + chunk, n_keys)
            index = group.index[start:stop]  # (c, W)
            keep = group.valid[start:stop]  # (c, W)
            base = pattern[index] * keep.unsqueeze(-1)  # (c, W, v)
            design = (base.unsqueeze(2) * eq_weights.view(1, 1, self.n_eq, -1)).reshape(
                stop - start, -1, n_value
            )  # (c, W*m, v)
            gram = (design.transpose(1, 2) @ design).double()
            gram.diagonal(dim1=1, dim2=2).add_(mu)
            self.chunks.append(
                (start, stop, index, keep, design, torch.linalg.lu_factor(gram))
            )

    def solve(self, residual: torch.Tensor) -> torch.Tensor:
        """residual: (n_facts, m) -> per-token correction (n_keys, n_value)."""
        out = torch.zeros(self.n_keys, self.n_value, dtype=self.dtype)
        for start, stop, index, keep, design, factor in self.chunks:
            rhs = (residual[index] * keep.unsqueeze(-1)).reshape(stop - start, -1, 1)
            normal = (design.transpose(1, 2) @ rhs).double()
            out[start:stop] = (
                torch.linalg.lu_solve(*factor, normal).squeeze(-1).to(self.dtype)
            )
        return out


@dataclass
class DigitCodeSolution:
    u: torch.Tensor
    v: torch.Tensor
    digits: torch.Tensor  # (n_labels, m)
    p: int
    masks: torch.Tensor  # (m, n_value)
    t0: torch.Tensor  # (m,) per-group stabilizer
    shift: torch.Tensor  # (m,) fitted decode offsets
    delta: float
    accuracy: float
    density: float
    round_index: int


def _digit_scores(
    z: torch.Tensor, digits: torch.Tensor, shift: torch.Tensor, delta: float
) -> torch.Tensor:
    """(n, L) nearest-codeword scores: sum over groups of the quadratic decode."""
    zz = (z - shift) / delta  # (n, m)
    lv = digits + 1.0  # (L, m)
    return zz @ lv.T - lv.pow(2).sum(1) / 2


def _fit_shifts(
    z: torch.Tensor,
    targets_z: torch.Tensor,
    digits: torch.Tensor,
    labels: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Per-group decode offset: quantiles of the observed error, scored
    jointly by the fraction of facts whose full label decodes correctly.

    Groups are searched greedily (coordinate descent, two passes): each
    group's candidate offsets are scored with the other groups' current
    offsets held fixed. Every step is an exact evaluation, not a fit.
    """
    m = z.shape[1]
    error = z - targets_z  # (n, m)
    quantiles = torch.linspace(0.0, 1.0, 9, dtype=z.dtype)
    shift = error.median(0).values.clone()
    for _ in range(2):
        for j in range(m):
            candidates = torch.cat(
                [shift[j].view(1), torch.quantile(error[:, j], quantiles)]
            )
            best_acc, best_c = -1.0, float(shift[j])
            for c in candidates:
                trial = shift.clone()
                trial[j] = c
                scores = _digit_scores(z, digits, trial, delta)
                acc = float((scores.argmax(1) == labels).float().mean())
                if acc > best_acc:
                    best_acc, best_c = acc, float(c)
            shift[j] = best_c
    return shift


def solve_digit_code(
    shape: ModelShape, facts: dict, params: DigitCodeParams, seed: int,
    verbose: bool = False,
) -> DigitCodeSolution:
    n_vocab, n_value = shape.input_vocab_size, shape.d_mlp - 1
    inputs, labels = facts["inputs"], facts["targets"]
    first, second = inputs[:, 0], inputs[:, 1]
    d = shape.d_mlp
    delta = params.delta

    digits, p = digit_codebook(shape.output_vocab_size, params.m)
    part_gen = torch.Generator().manual_seed(20_000 + params.group_seed)
    masks = make_groups(n_value, params.m, part_gen)
    # Split the stabilizer in proportion to group size, as twosided's t0 is to
    # the whole layer.
    group_frac = masks.sum(1) / n_value
    t0 = params.t0_scale * d * delta * group_frac  # (m,)

    wanted = t0 + (digits[labels] + 1.0) * delta  # (n, m)

    generator = torch.Generator().manual_seed(seed)
    u, v = init_embeddings(n_vocab, n_value, params.rho, generator, params.dtype)
    sums = torch.relu(u[first] + v[second]).sum(1)
    scale = float(t0.sum()) / float(sums.mean())
    u, v = u * scale, v * scale

    group_first = pad_group(first, n_vocab)
    group_second = pad_group(second, n_vocab)

    best: DigitCodeSolution | None = None
    pre = u[first] + v[second]
    for round_index in range(params.rounds):
        pattern = (pre > 0).to(params.dtype)
        held_u, held_v = u, v

        blocks_u = WeightedBlocks(
            group_first, pattern, masks, params.mu, n_vocab, n_value
        )
        blocks_v = WeightedBlocks(
            group_second, pattern, masks, params.mu, n_vocab, n_value
        )

        masked = pre * pattern
        z = masked @ masks.T  # (n, m) frozen-pattern group sums
        for _ in range(params.sweeps):
            step_u = blocks_u.solve(wanted - z)
            u = u + step_u
            z = z + (step_u[first] * pattern) @ masks.T
            step_v = blocks_v.solve(wanted - z)
            v = v + step_v
            z = z + (step_v[second] * pattern) @ masks.T

        step_pre = (u - held_u)[first] + (v - held_v)[second]
        step, crossings = _capped_step(pre, step_pre, params.flip_cap)
        u, v = held_u + step * (u - held_u), held_v + step * (v - held_v)
        pre = pre + step * step_pre

        hidden = torch.relu(pre)
        zc = hidden @ masks.T - t0  # (n, m) group sums, stabilizer removed
        shift = _fit_shifts(zc, (digits[labels] + 1.0) * delta, digits, labels, delta)
        scores = _digit_scores(zc, digits, shift, delta)
        accuracy = float((scores.argmax(1) == labels).float().mean())
        density = float((pre > 0).to(params.dtype).mean())

        if best is None or accuracy > best.accuracy:
            best = DigitCodeSolution(
                u=u.clone(), v=v.clone(), digits=digits, p=p, masks=masks,
                t0=t0, shift=shift.clone(), delta=delta, accuracy=accuracy,
                density=density, round_index=round_index,
            )
        if verbose:
            print(
                f"    r{round_index:3d} acc={accuracy:.4f} dens={density:.3f} "
                f"step={step:.3f}",
                flush=True,
            )
        if accuracy == 1.0:
            break
        if step == 1.0 and crossings == 0:
            break

    assert best is not None
    return best


def assemble(
    shape: ModelShape, solution: DigitCodeSolution, beta: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two weight matrices. Per label, the readout is the sum of the m
    per-group quadratic decodes; the bias neuron carries every constant."""
    n_vocab, d_mlp = shape.input_vocab_size, shape.d_mlp
    n_value = d_mlp - 1
    if beta is None:
        largest = float(max(solution.u.abs().max(), solution.v.abs().max()))
        beta = max(2.0 * largest, float(d_mlp))

    up = torch.empty(d_mlp, 2 * n_vocab, dtype=torch.float64)
    up[:n_value, :n_vocab] = solution.u.T.double()
    up[:n_value, n_vocab:] = solution.v.T.double()
    up[n_value, :] = beta / 2.0

    lv = solution.digits.double() + 1.0  # (L, m)
    masks = solution.masks.double()  # (m, v)
    delta = solution.delta
    # The decode reads (raw group sum - t0_j - shift_j) / delta, so the bias
    # column must carry both the stabilizer and the fitted offset.
    offset = (solution.t0 + solution.shift).double()  # (m,)
    down = torch.zeros(shape.output_vocab_size, d_mlp, dtype=torch.float64)
    down[:, :n_value] = (lv / delta) @ masks
    down[:, n_value] = -((lv * offset.view(1, -1)).sum(1) / delta
                         + lv.pow(2).sum(1) / 2) / beta
    return up.float(), down.float()
