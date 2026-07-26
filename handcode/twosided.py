"""A two-sided value code, with the ReLU itself as the gate.

`linsolve` stores a fact's label as a *number* -- the sum of the active
activations -- and reads it out with the quadratic decode of Appendix A. Its
ceiling is set by parameter counting: the first embedding is spent entirely on
the gate (`0` on selected neurons, `-BIG` elsewhere) and the unembedding is a
fixed function of the class index, so all fact information lives in the second
embedding's `(d - 1) * n_input_vocab ~ 2 d^2` values, against one equation per
fact. Measured capacities sit right at that ceiling at every `d`.

The obvious repair is to let *both* embeddings carry values. The obstacle is
that a mask depending on only one token makes the other embedding useless:

    sum_{i in S_a} (u_i[a] + v_i[b])  =  [sum_{i in S_a} u_i[a]]  +  [sum_{i in S_a} v_i[b]]
                                          (----- depends on a only -----)

so the `u` term collapses to one constant per first token and contributes
`n_input_vocab` unknowns rather than `(d-1) * n_input_vocab`. Every construction
with an explicitly built gate runs into a version of this, and the versions that
escape it need the gate to be dense enough that the gated values stay positive
(a ReLU cannot report a negative number), which caps them just as hard.

**This module drops the explicit gate entirely.** There are no mask matrices:
the active set is whatever the ReLU makes it,

    z_i = u_i[a] + v_i[b],      h_i = relu(z_i),      s = sum_i h_i,

and the requirement is one equation per fact, `s = (l + 1) * delta`. The active
set now depends on *both* tokens, through the signs of the very weights being
solved for, so neither embedding collapses: the unknowns are all `2 * (d-1) *
n_input_vocab ~ 4 d^2` of them, twice `linsolve`'s budget, and the ceiling
doubles with them.

The price is that the system is no longer linear -- the active set moves when
the weights move. It is, however, *piecewise* linear, and freezing the active
set makes it exactly linear:

    round 1..R:
        A = {i : z_i > 0}            <- read off the current weights
        solve  sum_{i in A} (u_i[a] + v_i[b]) = t   for (u, v), A held fixed

The solve is a block Gauss-Seidel sweep: with `v` fixed, each first token's
`u[a]` is an independent least-squares problem over that token's own facts, and
symmetrically for `v`. Each block is a ridge regression, which the challenge
rules name explicitly as allowed, and the outer loop is a fixed, small number of
rounds -- freezing a piecewise-linear system's active set and re-solving is the
same move `linsolve`'s greedy drop already makes, not a numerical optimizer.

Two things follow from the ReLU-as-gate choice, both of which matter for the
post's actual goal of resembling trained models:

* **A much smaller offset.** Both constructions add a constant `t0` to every
  target, but for different reasons and at different sizes. `linsolve` needs
  every *gated* value to stay positive against a label range of `d`, which
  forces `t0` to grow like `d^2` and drags its weights to `1e6`. Here a neuron
  that wants to be negative simply switches off -- that is the ReLU doing its
  job -- and `t0` only has to keep the sign pattern from churning between
  rounds, which `t0 = 16 d` achieves. Weights come out `O(d)` rather than
  `O(d^2)`, and the decode stays inside float32 at every size tested.
* **Density is free.** The fraction of active neurons is set by the
  initialization and then by the solve, not by a positivity constraint, so it
  can be put at the ~0.53 that trained models show.

The readout is the same quadratic decode as `linsolve`, assembled at the bottom
of this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ModelShape

# Candidate initial activation densities. Trained models sit at ~0.53 (see
# FINDINGS.md); the sweep brackets it because the density the *solve* settles
# at is not the density it starts from.
RHO_VALUES = (0.53, 0.35, 0.70, 0.20)
# Ridge strength for the per-token blocks. This is the sweep that matters most:
# a token's system is square at load `4 d^2 / 2`, and right at the transition
# the exact solve is badly conditioned, throws out an enormous update, and the
# flip cap responds by refusing to move. A larger ridge shrinks the update back
# to something the frozen pattern can absorb. Which value wins is a function of
# load, so all of them are tried.
MU_VALUES = (1e-2, 1e-8, 1.0)
# (keep, drop_from): what fraction of facts to keep once dropping starts, and
# the round it starts at. `keep < 1` sacrifices the worst-fitting facts so the
# rest can be solved exactly -- the right move under an `acc >= 0.9` budget,
# and the same greedy step `linsolve` uses. Ordered so the plain solve, which
# is the only one that can reach `acc = 1`, comes first.
SCHEDULES = ((1.00, 10**9), (0.92, 80), (0.85, 80))


@dataclass(frozen=True)
class TwoSidedParams:
    rho: float = 0.53  # initial fraction of active neurons
    mu: float = 1e-2  # ridge strength for the per-token blocks
    rounds: int = 150  # freeze-the-pattern-and-solve rounds
    sweeps: int = 4  # Gauss-Seidel sweeps per round
    flip_cap: float = 0.02  # largest fraction of the pattern a step may flip
    # Ablation: hold the first embedding at its random initialization, so only
    # `v` carries facts. That is `linsolve`'s parameter budget (`2 d^2`) under
    # this construction's gate, and it isolates how much of the capacity comes
    # from having freed the first embedding rather than from anything else here.
    freeze_first: bool = False
    keep: float = 1.0  # fraction of facts kept once dropping starts
    drop_from: int = 10**9  # first round that drops
    t0_scale: float = 16.0  # target offset, in units of d (see `t0`)
    delta: float = 1.0  # spacing between adjacent labels in `s`
    # Working precision of the solve. float32 is enough because `t0` scales
    # with `d`: the decode must place a sum of size `t0 + d ~ 17d` within half
    # a unit, a relative accuracy of `1/34d` (2e-4 at d=128), while a d-term
    # float32 sum costs about `sqrt(d) * 1e-7`. Two orders of margin -- which
    # the `t0 ~ d^2` parameterization did not have, and is the second reason to
    # have dropped it.
    dtype: torch.dtype = torch.float32

    def t0(self, shape: ModelShape) -> float:
        """Constant added to every target.

        Its job is to hold the sign pattern still. A neuron's pre-activation
        has to be large compared with the amount the solve moves it, or the
        active set churns; that amount is set by the label range, so the offset
        buys margin and `t0 = 0` fails outright (0.07 accuracy at d=64).

        It is bounded above by float32. The readout has to resolve a half-unit
        gap out of logits of size `d * (t0 + d)`, and it forms them by
        subtracting two nearly equal large numbers, so the offset cannot grow
        faster than the precision budget. `t0 ~ d^2` -- the natural-looking
        choice, and the one `linsolve` is forced into -- already costs 4% of
        the accuracy at d=128 while float64 still reads 1.000. Scaling with `d`
        instead keeps the ratio flat as models grow.
        """
        return self.t0_scale * shape.d_mlp * self.delta

    def __str__(self) -> str:
        return f"rho={self.rho:g}, mu={self.mu:g}, keep={self.keep:g}"


# ── padded grouping ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PadGroup:
    """Facts bucketed by one of their two tokens, padded to a rectangle."""

    index: torch.Tensor  # (n_keys, width) long -- fact ids, padded with 0
    valid: torch.Tensor  # (n_keys, width) bool -- which entries are real

    @property
    def width(self) -> int:
        return self.index.shape[1]


def pad_group(keys: torch.Tensor, n_keys: int) -> PadGroup:
    counts = torch.bincount(keys, minlength=n_keys)
    width = int(counts.max().item()) if len(counts) else 0

    order = torch.argsort(keys, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))
    slot = ranks - starts[keys]  # position of each fact within its own bucket

    index = torch.zeros(n_keys, width, dtype=torch.long)
    valid = torch.zeros(n_keys, width, dtype=torch.bool)
    index[keys, slot] = torch.arange(len(keys))
    valid[keys, slot] = True
    return PadGroup(index=index, valid=valid)


# ── the block solve ───────────────────────────────────────────────────────────


class TokenBlocks:
    """Every token's normal equations for one frozen pattern, factorized once.

    For each key (a first or second token) the rows of the design are the
    frozen activation patterns of that token's live facts, and asking for the
    correction that best fixes that token's share of the decode residual is

        min_x || P x - r ||^2 + mu ||x||^2

    with `P` that token's `(n_facts_of_token, n_value)` pattern matrix. Solving
    for the *correction* rather than for `u[a]` itself keeps whatever the
    current iterate has put in the null space, which is what makes the sweep a
    Gauss-Seidel step rather than a restart.

    The design depends only on the frozen pattern and the live set, both of
    which are constant across a round's sweeps, so the Gram matrices and their
    factorizations are built once here and reused. That is the whole cost of
    the algorithm: forming a Gram is `width * n_value^2` per token against
    `width * n_value` for applying it, so rebuilding it every sweep -- which is
    what a straightforward implementation does -- spends `n_value` times more
    arithmetic on setup than on the solves it exists to serve.

    Padded rows are exactly zero and carry a zero right-hand side, so they drop
    out of the primal Gram and decouple in the dual one -- they cost flops, not
    correctness. Under- and over-determined tokens must be split *per token*:
    the two ridge forms agree in exact arithmetic but not in floating point at
    these ridge strengths, and picking wrongly is silent and large.
    """

    def __init__(
        self,
        group: PadGroup,
        pattern: torch.Tensor,
        live: torch.Tensor,
        mu: float,
        n_keys: int,
        n_value: int,
        chunk: int = 128,
    ) -> None:
        self.group = group
        self.pattern = pattern
        self.n_keys = n_keys
        self.n_value = n_value
        self.dtype = pattern.dtype
        self.chunks: list[tuple] = []

        for start in range(0, n_keys, chunk):
            stop = min(start + chunk, n_keys)
            index = group.index[start:stop]
            keep = group.valid[start:stop] & live[index]  # (c, width)
            design = pattern[index] * keep.unsqueeze(-1)  # (c, width, n_value)
            under = keep.sum(1) <= n_value

            parts: list[tuple] = []
            for use_dual in (True, False):
                select = under if use_dual else ~under
                if not bool(select.any()):
                    continue
                sub = design[select]
                gram = sub @ sub.transpose(1, 2) if use_dual else sub.transpose(1, 2) @ sub
                # The Gram is formed in the working precision -- its entries are
                # counts of co-active neurons, so float32 holds them exactly --
                # but factorized in float64. A token with no live facts, or a
                # neuron it never activates, leaves an exactly zero row, and
                # `mu` is all that keeps the block invertible; at `mu = 1e-8`
                # against Gram entries of order `width`, adding it in float32 is
                # a no-op and the factorization fails outright.
                gram = gram.double()
                gram.diagonal(dim1=1, dim2=2).add_(mu)
                # The design is kept rather than re-gathered per sweep: it is
                # the same `n_facts * n_value` of memory the pattern already
                # occupies, and gathering it is a bigger cost than the solve.
                parts.append((use_dual, select, sub, torch.linalg.lu_factor(gram)))
            self.chunks.append((start, stop, index, keep, parts))

    def solve(self, residual: torch.Tensor) -> torch.Tensor:
        """Per-token least-squares correction; returns (n_keys, n_value)."""
        out = torch.zeros(self.n_keys, self.n_value, dtype=self.dtype)

        for start, stop, index, keep, parts in self.chunks:
            rhs = residual[index] * keep
            block = torch.zeros(stop - start, self.n_value, dtype=self.dtype)
            for use_dual, select, sub, factor in parts:
                sub_rhs = rhs[select].unsqueeze(-1)
                if use_dual:
                    dual = torch.linalg.lu_solve(*factor, sub_rhs.double())
                    block[select] = (sub.transpose(1, 2) @ dual.to(self.dtype)).squeeze(-1)
                else:
                    normal = (sub.transpose(1, 2) @ sub_rhs).double()
                    solved = torch.linalg.lu_solve(*factor, normal)
                    block[select] = solved.squeeze(-1).to(self.dtype)
            out[start:stop] = block

        return out


# ── initialization ────────────────────────────────────────────────────────────


def init_embeddings(
    n_vocab: int,
    n_value: int,
    rho: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random `(u, v)` whose pre-activations are positive a fraction `rho` of
    the time.

    With `u, v ~ N(m, 1)` independent, `z = u[a] + v[b] ~ N(2m, 2)`, so
    `P(z > 0) = Phi(m * sqrt(2))` and the offset that hits a density is
    `m = Phi^-1(rho) / sqrt(2)`. Note that `m` goes on *each* embedding, not
    half of it on each: the sum already carries the factor of two, and halving
    it here is a silent error that compresses the whole density sweep towards
    0.5 (rho=0.2 comes out at 0.33). `test_init_embeddings_hit_the_requested_density`
    pins this down.

    The overall scale is left to `_rescale`, which is a separate step because
    the pattern -- the only thing the initialization is really choosing -- is
    invariant to it.
    """
    quantile = torch.special.ndtri(torch.tensor(rho, dtype=torch.float64))
    offset = float(quantile) / 2.0**0.5
    u = torch.randn(n_vocab, n_value, generator=generator, dtype=dtype) + offset
    v = torch.randn(n_vocab, n_value, generator=generator, dtype=dtype) + offset
    return u, v


def _rescale(
    u: torch.Tensor,
    v: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    mean_target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale `(u, v)` so the decoded sums already average `mean_target`.

    Scaling both embeddings by the same factor leaves every activation pattern
    untouched and multiplies every decoded sum by it, so this costs nothing and
    fixes the one thing the initialization would otherwise get badly wrong: if
    the sums start orders of magnitude away from the targets, the first solve
    has to move every weight by more than its own size, and the sign pattern it
    was solved against is destroyed.
    """
    sums = torch.relu(u[first] + v[second]).sum(1)
    scale = mean_target / float(sums.mean())
    return u * scale, v * scale


# ── the construction ──────────────────────────────────────────────────────────


def _decode_accuracy(
    sums: torch.Tensor,
    targets: torch.Tensor,
    delta: float,
    shifts: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Fraction of facts the quadratic readout gets right, per candidate shift.

    The readout below is maximized at `c + 1 = round((s - shift) / delta)`
    clipped to the label range, so the argmax can be evaluated without ever
    forming the logits. `tests/test_twosided.py` checks the two agree.
    """
    predicted = torch.round((sums - shifts.unsqueeze(1)) / delta) - 1
    predicted = predicted.clamp(0, n_classes - 1)
    return (predicted == targets).to(sums.dtype).mean(1)


def _best_shift(
    sums: torch.Tensor,
    targets: torch.Tensor,
    delta: float,
    t0: float,
    n_classes: int,
) -> tuple[float, float]:
    """Pick the readout's offset from the empirical decode error.

    The offset is `t0` plus a correction. It needs one because freezing the
    active set and re-solving leaves a *one-sided* error: a neuron the solve
    wanted negative contributes zero instead of its negative value, so the true
    sum is never below the solved one, and every decoded label is biased upward
    by a common amount. One constant on the bias neuron removes it. Candidates
    are quantiles of the observed error, each scored exactly.
    """
    error = sums - (t0 + (targets + 1).to(sums.dtype) * delta)
    quantiles = torch.linspace(0.0, 1.0, 13, dtype=sums.dtype)
    candidates = t0 + torch.cat(
        [torch.zeros(1, dtype=sums.dtype), torch.quantile(error, quantiles)]
    )
    scores = _decode_accuracy(sums, targets, delta, candidates, n_classes)
    winner = int(scores.argmax())
    return float(candidates[winner]), float(scores[winner])


def _capped_step(
    pre: torch.Tensor,
    step_pre: torch.Tensor,
    flip_cap: float,
) -> tuple[float, int]:
    """The longest step that leaves the frozen active set nearly intact.

    The solve is only meaningful while the pattern it was solved against still
    holds, so the whole step is usually far too long: taking it can move a
    third of the neurons across zero, at which point the equations it satisfied
    describe a different model than the one that gets built. Worse, the mismatch
    is *one-sided* -- a neuron that ends up negative contributes zero instead of
    its negative value, so every decoded sum overshoots, the next solve responds
    by shrinking the weights, and more neurons fall below zero. Left alone that
    ratchet empties the network: measured densities fall to 0.05.

    Capping the flip fraction cuts the ratchet at the root, and it is
    self-scheduling -- the step starts tiny while the pattern is still being
    found and rises to 1 once it settles, which no fixed damping factor
    reproduces.

    The cap needs no search. Along the step, neuron `i` of fact `n` is an affine
    function `pre + a * step_pre` of the step length, so it crosses zero at
    exactly one `a`, and only when `step_pre` opposes it. Collect those crossing
    points and the number of flips at step `a` is just how many of them are
    below `a` -- so the longest admissible step is the `flip_cap` order
    statistic of the crossing points, one `kthvalue` over the whole array.

    Returns the step and how many units cross zero within the *full* step. Zero
    crossings at a full step means the solve reproduced the pattern it was
    handed: the iteration has reached its fixed point and further rounds
    re-solve an identical system, so the caller can stop.
    """
    crossing = ((pre > 0) & (step_pre < 0)) | ((pre <= 0) & (step_pre > 0))
    ratio = -pre / step_pre  # where `crossing`, the step that zeroes this unit
    k = max(1, int(flip_cap * pre.numel()))
    # Only crossings inside the step matter, and there are far fewer of them
    # than there are neurons; discarding the rest first turns the order
    # statistic from a scan of the whole array into a scan of a small one.
    at = ratio[crossing & (ratio <= 1.0)]
    if at.numel() < k:
        return 1.0, at.numel()  # fewer flips than the cap allows at a full step
    return float(at.kthvalue(k).values), at.numel()


@dataclass
class TwoSidedSolution:
    u: torch.Tensor  # (n_vocab, n_value)
    v: torch.Tensor  # (n_vocab, n_value)
    shift: float
    accuracy: float  # analytic decode accuracy, float64
    density: float  # fraction of active neurons at the solution
    round_index: int


def solve_two_sided(
    shape: ModelShape, facts: dict, params: TwoSidedParams, seed: int,
    verbose: bool = False,
) -> TwoSidedSolution:
    """Freeze the active set, solve, repeat -- keeping the best round."""
    n_vocab, n_value = shape.input_vocab_size, shape.d_mlp - 1
    inputs, targets = facts["inputs"], facts["targets"]
    first, second = inputs[:, 0], inputs[:, 1]
    n_facts = len(targets)
    t0 = params.t0(shape)
    wanted = t0 + (targets + 1).to(params.dtype) * params.delta

    generator = torch.Generator().manual_seed(seed)
    u, v = init_embeddings(n_vocab, n_value, params.rho, generator, params.dtype)
    u, v = _rescale(u, v, first, second, float(wanted.mean()))

    group_first = pad_group(first, n_vocab)
    group_second = pad_group(second, n_vocab)
    live = torch.ones(n_facts, dtype=torch.bool)

    best: TwoSidedSolution | None = None
    pre = u[first] + v[second]  # (n_facts, n_value) pre-activations
    for round_index in range(params.rounds):
        pattern = (pre > 0).to(params.dtype)
        held_u, held_v = u, v

        first_blocks = TokenBlocks(
            group_first, pattern, live, params.mu, n_vocab, n_value
        )
        second_blocks = TokenBlocks(
            group_second, pattern, live, params.mu, n_vocab, n_value
        )

        # `masked` is the decode the frozen pattern predicts. Each block solve
        # changes one embedding, so it can be carried forward with a single
        # gather instead of recomputing both -- the gathers, not the linear
        # algebra, are what this loop spends its time on.
        masked = (pre * pattern).sum(1)
        for _ in range(params.sweeps):
            if not params.freeze_first:
                step_u = first_blocks.solve(wanted - masked)
                u = u + step_u
                masked = masked + (step_u[first] * pattern).sum(1)
            step_v = second_blocks.solve(wanted - masked)
            v = v + step_v
            masked = masked + (step_v[second] * pattern).sum(1)

        step_pre = (u - held_u)[first] + (v - held_v)[second]
        step, crossings = _capped_step(pre, step_pre, params.flip_cap)
        u, v = held_u + step * (u - held_u), held_v + step * (v - held_v)
        pre = pre + step * step_pre

        # Score the *actual* model: relu, not the frozen pattern.
        sums = torch.relu(pre).sum(1)
        shift, accuracy = _best_shift(
            sums, targets, params.delta, t0, shape.output_vocab_size
        )
        density = float((pre > 0).to(params.dtype).mean())
        if best is None or accuracy > best.accuracy:
            best = TwoSidedSolution(
                u=u.clone(), v=v.clone(), shift=shift, accuracy=accuracy,
                density=density, round_index=round_index,
            )
        if verbose:
            print(
                f"    r{round_index:2d} acc={accuracy:.4f} dens={density:.3f} "
                f"step={step:.3f} max|u|={float(u.abs().max()):.3g}",
                flush=True,
            )
        if accuracy == 1.0:
            break
        if step == 1.0 and crossings == 0 and params.keep == 1.0:
            break  # fixed point: the next round would re-solve the same system

        if round_index + 1 >= params.drop_from and params.keep < 1.0:
            # Greedy drop: give up on the worst-fitting facts so the rest can
            # be solved exactly, rather than letting least squares spread a
            # little error over every fact.
            error = (sums - shift - wanted).abs()
            live = error <= torch.quantile(error, params.keep)

    assert best is not None
    return best


def assemble(
    shape: ModelShape,
    solution: TwoSidedSolution,
    delta: float,
    beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a solution into the two weight matrices.

    Neuron `d - 1` is the always-on bias neuron of Appendix A, held at `beta`.
    With `s` the sum over the other neurons,

        logit_c = (c + 1) (s - shift) / delta  -  (c + 1)^2 / 2
                = -((c + 1) - (s - shift)/delta)^2 / 2  +  const

    which is maximized at the label whose target `s` is nearest. Both terms are
    linear in the activations, so this is an ordinary unembedding.

    `beta` is a pure gauge. It appears as `beta / 2` in the embedding and
    divides the bias column of the unembedding, so it cancels exactly from
    every logit and cannot change accuracy -- it only decides which of the two
    matrices carries the offset's magnitude. Left at `d` it dumps all of it on
    the unembedding, whose largest entry is then `~16 d`. Setting it to twice
    the largest solved value instead puts the bias row just under the entries
    already present in the embedding, so the embedding's maximum is unchanged
    and the unembedding falls to `d` -- its floor, since the value columns are
    the labels themselves. Measured at d=64: `max|down|` 1056 -> 64, with
    `max|up|` and the accuracy both untouched.
    """
    n_vocab, d_mlp = shape.input_vocab_size, shape.d_mlp
    n_value = d_mlp - 1
    if beta is None:
        largest = float(max(solution.u.abs().max(), solution.v.abs().max()))
        beta = max(2.0 * largest, float(d_mlp))

    up = torch.empty(d_mlp, 2 * n_vocab, dtype=torch.float64)
    up[:n_value, :n_vocab] = solution.u.T.double()
    up[:n_value, n_vocab:] = solution.v.T.double()
    up[n_value, :] = beta / 2.0  # u = v = beta/2, so h = beta on every fact

    labels = torch.arange(shape.output_vocab_size, dtype=torch.float64) + 1
    down = torch.zeros(shape.output_vocab_size, d_mlp, dtype=torch.float64)
    down[:, :n_value] = (labels / delta).unsqueeze(1)
    down[:, n_value] = -(labels * solution.shift / delta + labels**2 / 2) / beta
    return up.float(), down.float()


def two_sided_weights(
    shape: ModelShape, facts: dict, params: TwoSidedParams, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    solution = solve_two_sided(shape, facts, params, seed)
    return assemble(shape, solution, params.delta)


def best_two_sided_accuracy(
    shape: ModelShape,
    facts: dict,
    rho_index: int,
    seed: int,
    threshold: float = 1.0,
    rounds: int = 150,
) -> float:
    """Best float32 accuracy over the (mu, schedule) sweep at one density.

    `rho_index` selects the initial density from `RHO_VALUES`; the capacity
    harness sweeps it as the one visible hyperparameter, exactly as it sweeps
    `k` for `linsolve`. Schedules that drop facts cannot reach `acc = 1`, so
    they are skipped when that is the threshold.
    """
    from .model import accuracy

    rho = RHO_VALUES[(rho_index - 1) % len(RHO_VALUES)]
    best = 0.0
    for keep, drop_from in SCHEDULES:
        if keep < 1.0 and threshold == 1.0:
            continue
        for mu in MU_VALUES:
            params = TwoSidedParams(
                rho=rho, mu=mu, rounds=rounds, keep=keep, drop_from=drop_from
            )
            solution = solve_two_sided(shape, facts, params, seed)
            if solution.accuracy < best:
                continue  # cannot beat the incumbent once float32 is applied
            up, down = assemble(shape, solution, params.delta)
            best = max(best, accuracy(up, down, facts))
            if best >= threshold:
                return best
    return best
