"""Storing the label as an activation *value*, by solving a linear system.

This is the construction that closes most of the gap to the trained model. Two
ideas are combined, both drawn from the post itself.

1. From the post's Appendix A: a ReLU neuron's output is a *number*, so a single
   neuron can carry a whole label rather than one bit of a pattern. Appendix A
   gives token `a` its own selector neuron, whose activation the second token
   sets to `L[a,b] + 1`; a quadratic readout then turns "activation is near
   l + 1" into "argmax is l". It stores every possible fact, but needs
   `d_MLP = n_input_vocab + 1` neurons -- one per first token -- which is twice
   the width the challenge allows.

2. From Dugan et al., which the post names as a valid entry: fix the *gating* at
   random, and what remains is a linear system in the value weights, which can
   be solved exactly.

Putting them together removes Appendix A's width requirement. Instead of one
selector per first token, give each first token a random set `S_a` of `k`
selector neurons and require only that the *sum* over them come out right:

    u_i[a] = 0 if i in S_a else -BIG          v_i[b] = w_i[b]
    =>  h_i = 1[i in S_a] * w_i[b]
    =>  sum_i h_i = sum_{i in S_a} w_i[b]      (the decoded value)

so the requirement "every fact decodes to its own label" is one linear equation
per fact,

    sum_{i in S_a} w_i[b] = T0 + l + 1,

in the `(d-1) * n_input_vocab` unknowns of the second embedding. The equations
for different second tokens `b` are disjoint, so this is `n_input_vocab`
independent small least-squares problems, each with about `N / n_input_vocab`
equations in `d - 1` unknowns -- exactly solvable while

    N  <~  n_input_vocab * (d - 1)  ~  2 d^2,

which is far above the capacity anything here actually reaches. What limits the
construction in practice is conditioning: as the per-token systems fill up, the
solved weights grow and the decoded sums stop landing within +-1/2 of their
targets.

Two details make it work:

* **The offset T0.** A ReLU cannot report a negative value, so every `w_i[b]`
  must stay positive. Targets `l + 1` span `[1, d]` with mean `(d+1)/2`, so the
  spread is as large as the mean and the solved weights sit right at zero.
  Adding a constant `T0` to every target lifts the whole solution safely
  positive, and costs nothing: `T0` is the same for every fact, so the readout
  subtracts it back off using the bias neuron.
* **The bias neuron.** One neuron is held permanently on (`u = v = 1`, so
  `h = 2`), exactly as in Appendix A, to supply the per-class thresholds the
  quadratic decode needs. With `s` the decoded sum,

      logit_c = (c+1) * (s - T0) - (c+1)^2 / 2 = -(c - l)^2 / 2 + const

  which is maximized at `c = l`. Both terms are linear in `h`, so this is an
  ordinary unembedding: `(c+1)` on every value neuron and
  `-(c+1) T0 / 2 - (c+1)^2 / 4` on the bias neuron.

Nothing here uses gradient descent: the weights come from `n_input_vocab`
independent ridge solves plus a closed-form readout.
"""

from dataclasses import dataclass

import torch

from .model import ModelShape

# Ridge strengths and target offsets tried for each k. Both are cheap to sweep:
# the solve is per second-token and tiny.
MU_VALUES = (1e-9, 1e-6, 1e-3)
# A bigger offset leaves more headroom before a solved value would have to go
# negative (and get clamped), which matters a lot. It is bounded above only by
# float32 round-off: the decode must still resolve +-1/2 against a sum of size
# T0, so the usable ceiling falls as d grows. Sweeping a few values covers it.
T0_SCALES = (20.0, 100.0, 300.0, 1000.0)
# (keep, rounds) pairs: (1.0, 1) is the plain exact solve, the others sacrifice
# the worst-fitting facts to concentrate error away from the rest.
# (keep, rounds, pertoken). `pertoken` first sacrifices, for each second token,
# every fact past the (d-1) that its own system can actually satisfy.
DROP_SCHEDULES = (
    (1.0, 1, False),
    (0.92, 6, False),
    (0.85, 6, False),
    (0.92, 6, True),
    (0.85, 3, True),
    (0.85, 6, True),
)


@dataclass(frozen=True)
class LinsolveParams:
    k: int  # selector neurons per first token
    mu: float  # ridge strength
    t0_scale: float  # target offset, in units of d
    keep: float = 1.0  # fraction of facts kept when re-solving (1.0 = no drop)
    rounds: int = 1  # solve passes; >1 enables the greedy drop
    pertoken: bool = False  # pre-drop each second token's excess facts

    def __str__(self) -> str:
        return f"k={self.k}, mu={self.mu:g}, keep={self.keep:g}, pertoken={self.pertoken}"


def selector_masks(
    n_vocab: int, n_value: int, k: int, generator: torch.Generator
) -> torch.Tensor:
    """(n_vocab, n_value) 0/1: which value neurons each first token selects.

    Random, and deliberately so -- this is the "gating chosen at random" step;
    all the fact-specific information ends up in the solved values instead.
    """
    mask = torch.zeros(n_vocab, n_value, dtype=torch.float64)
    for token in range(n_vocab):
        mask[token, torch.randperm(n_value, generator=generator)[:k]] = 1.0
    return mask


def _solve_once(
    facts: dict,
    mask: torch.Tensor,
    live: torch.Tensor,
    n_vocab: int,
    n_value: int,
    k: int,
    mu: float,
    t0: float,
) -> torch.Tensor:
    """One pass: solve `sum_{i in S_a} w_i[b] = T0 + l + 1` for each second token.

    Each column is an independent problem over the *live* facts sharing that
    second token. Under-determined columns take the minimum-norm solution (dual
    form), over-determined ones the ridge solution.
    """
    inputs, targets = facts["inputs"], facts["targets"]
    wanted = (targets + 1).double() + t0
    baseline = t0 / k  # every active neuron starts here, keeping w positive

    values = torch.full((n_value, n_vocab), baseline, dtype=torch.float64)
    for b in range(n_vocab):
        rows = ((inputs[:, 1] == b) & live).nonzero().flatten()
        if len(rows) == 0:
            continue
        design = mask[inputs[rows, 0]]  # (n_b, n_value)
        residual = wanted[rows] - baseline * k
        if design.shape[0] <= n_value:
            gram = design @ design.T + mu * torch.eye(design.shape[0], dtype=torch.float64)
            values[:, b] = baseline + design.T @ torch.linalg.solve(gram, residual)
        else:
            gram = design.T @ design + mu * torch.eye(n_value, dtype=torch.float64)
            values[:, b] = baseline + torch.linalg.solve(gram, design.T @ residual)

    return values.clamp(min=0.0)  # a ReLU would zero these anyway


def solve_values(
    facts: dict,
    mask: torch.Tensor,
    n_vocab: int,
    n_value: int,
    k: int,
    mu: float,
    t0: float,
    keep: float = 1.0,
    rounds: int = 1,
    pertoken: bool = False,
) -> torch.Tensor:
    """Solve for the values, optionally sacrificing the worst-fitting facts.

    Past the exact-solve threshold the per-token systems become
    over-determined, and least squares responds by spreading a little error
    over *every* fact -- which is the worst possible allocation when the metric
    is "how many facts decode correctly". Under an acc >= 0.9 budget the right
    move is the opposite: drop the hardest `1 - keep` of the facts and re-solve
    on the rest, concentrating all the damage on facts we have already written
    off. This is a greedy algorithm, which the challenge rules permit.
    """
    inputs, targets = facts["inputs"], facts["targets"]
    wanted = (targets + 1).double() + t0
    live = torch.ones(len(targets), dtype=torch.bool)

    if pertoken:
        # A second token's system has n_value unknowns, so it can satisfy at
        # most n_value of its facts exactly; write the excess off immediately
        # rather than letting least squares smear their error over the rest.
        for b in range(n_vocab):
            rows = (inputs[:, 1] == b).nonzero().flatten()
            if len(rows) > n_value:
                live[rows[n_value:]] = False

    values = _solve_once(facts, mask, live, n_vocab, n_value, k, mu, t0)
    for _ in range(rounds - 1):
        decoded = (mask[inputs[:, 0]] * values.T[inputs[:, 1]]).sum(1)
        error = (decoded - wanted).abs()
        live = error <= torch.quantile(error, keep)
        values = _solve_once(facts, mask, live, n_vocab, n_value, k, mu, t0)

    return values


def linsolve_weights(
    shape: ModelShape, facts: dict, params: LinsolveParams, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build both matrices. The last neuron is the always-on bias neuron."""
    n_vocab, d_mlp = shape.input_vocab_size, shape.d_mlp
    n_value = d_mlp - 1
    t0 = params.t0_scale * shape.output_vocab_size

    generator = torch.Generator().manual_seed(seed)
    mask = selector_masks(n_vocab, n_value, min(params.k, n_value), generator)
    values = solve_values(
        facts, mask, n_vocab, n_value, min(params.k, n_value), params.mu, t0,
        keep=params.keep, rounds=params.rounds, pertoken=params.pertoken,
    )

    # -BIG on unselected tokens keeps those neurons silent whatever the value is.
    big = float(values.max()) + 1.0
    up = torch.empty(d_mlp, 2 * n_vocab, dtype=torch.float64)
    up[:n_value, :n_vocab] = torch.where(mask.T.bool(), 0.0, -big)
    up[:n_value, n_vocab:] = values
    up[n_value, :] = 1.0  # bias neuron: u = v = 1 so h = 2 always

    labels = torch.arange(shape.output_vocab_size, dtype=torch.float64) + 1
    down = torch.zeros(shape.output_vocab_size, d_mlp, dtype=torch.float64)
    down[:, :n_value] = labels.unsqueeze(1)  # (c+1) * s
    down[:, n_value] = -labels * t0 / 2 - labels**2 / 4  # undo T0, add -(c+1)^2/2
    return up.float(), down.float()


def best_linsolve_accuracy(
    shape: ModelShape, facts: dict, k: int, seed: int
) -> float:
    """Best accuracy over the internal (mu, drop-schedule, T0) sweep for one k.

    Uses the batched solver in `fastsolve`. Because the linear system does not
    involve T0 at all (see that module), every T0 in the sweep is scored from a
    single solve -- the sweep is free.
    """
    from .fastsolve import assemble, cached_grouping, cached_mask, solve_profile
    from .model import accuracy

    n_value = shape.d_mlp - 1
    kk = min(k, n_value)
    grouping = cached_grouping(facts, shape)
    mask = cached_mask(shape.input_vocab_size, n_value, kk, seed)
    n_facts = len(facts["targets"])

    best = 0.0
    for mu in MU_VALUES:
        for keep, rounds, pertoken in DROP_SCHEDULES:
            x = solve_profile(grouping, mask, n_value, mu, keep, rounds, pertoken, n_facts)
            for t0_scale in T0_SCALES:
                up, down = assemble(
                    shape, mask, x, kk, t0_scale * shape.output_vocab_size
                )
                best = max(best, accuracy(up, down, facts))
                if best == 1.0:
                    return best
    return best
