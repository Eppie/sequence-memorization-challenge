"""A soft ridge seed: partial fit with a near-tie reservoir, no gradient descent.

FINDINGS.md 20 measured what the stride flow needs of its seed and found two
properties: enough fit (the bootstrap threshold sits between train accuracy
0.34 and 0.58) and a *soft* geometry -- gradient descent's fitting states keep
~0.5% of their pre-activations within ~1e-3 of zero (median ~1), a near-tie
reservoir the pedestal-stiffened constructions lack by ~12x. But the seed
families tested there confound the properties with their provenance: the GD
prefixes have both, the digit construction has neither (it is fully fit AND
stiff -- its pedestal exists precisely to push pre-activations away from
zero). The untested cell is a seed with both properties and no gradient
descent anywhere in its ancestry. This module builds that seed.

The recipe is the most boring rules-legal thing that could work:

    init (u, v) at the trained models' activation density and normalize the
    median |pre-activation| to 1 (a pure gauge: patterns and argmaxes are
    scale-invariant);
    each round: freeze the active set; ridge-fit the readout to one-hot
    targets; per-token ridge sweeps move the embeddings toward those targets;
    take the round's move only as far as the flip cap allows (twosided's
    order-statistic step rule);
    stop at the requested accuracy, or when the rounds run out.

Every solve is a ridge regression -- the challenge names them as allowed, and
they are the same moves `twosided`/`digitcode` already make. What differs is
the objective: no equality targets, no pedestal, no offset. `mu` regularizes
each round's per-token embedding *step*, so heavy `mu` keeps the geometry near
its (maximally soft) init while the fit climbs only as far as small motions
carry it -- `mu` is the knob that trades fit against softness. Whether any
(fit, softness) cell this reaches actually bootstraps the stride flow is the
experiment (`probe_flippolicy.py --phase softseed` builds the sweep;
`--phase stride --stride-seed soft:<tag>` runs a seed through the flow).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.special import ndtri

from .model import ModelShape


@dataclass(frozen=True)
class SoftSeedParams:
    rho: float = 0.53  # initial fraction of active neurons (trained models')
    mu: float = 1e-1  # ridge on each round's per-token embedding step
    lam: float = 1e-1  # ridge on the readout fit
    rounds: int = 40
    sweeps: int = 2  # Gauss-Seidel sweeps (u then v) per round
    flip_cap: float = 0.02  # largest pattern fraction one round may flip
    target: float = 1.0  # one-hot target magnitude for the readout fit
    acc_stop: float | None = None  # stop once train accuracy reaches this
    # The ridge rounds never build a near-tie reservoir on their own -- every
    # measured config sits at the Gaussian-init ratio (~6-9e-3), the digit
    # construction's stiffness class. soften_ratio, if set, CONSTRUCTS the
    # reservoir: compress the smallest-|pre| band toward zero (per-column
    # ridge, signs preserved) until q0.5%/median hits the target. GD's edge
    # states measure ~1.1e-3 (FINDINGS.md 20); choosing the property instead
    # of drifting into it is the point of the seed experiment.
    soften_ratio: float | None = None
    soften_q: float = 0.01  # |pre| quantile band the softening may touch
    # The ridge readout comes out ~19x smaller in rms than a GD fitting
    # state's, which breaks the stride flow two ways: box-scale vertex
    # nudges swamp it, and no fact's logit gap clears tau, so fit facts
    # never saturate and keep churning the oracle. w_rms rescales W after
    # the final fit (argmax-invariant: accuracy, pattern, and softness are
    # untouched; only the gap-vs-tau regime changes). GD's epoch-180
    # readout measures rms 1.28.
    w_rms: float | None = None

    def __str__(self) -> str:
        s = f"mu={self.mu:g}, lam={self.lam:g}, rounds={self.rounds}"
        if self.soften_ratio is not None:
            s += f", soften={self.soften_ratio:g}"
        return s


@dataclass
class SoftSeedResult:
    u: np.ndarray  # (n_vocab, d_mlp) float64
    v: np.ndarray  # (n_vocab, d_mlp) float64
    W: np.ndarray  # (n_labels, d_mlp) float64
    accuracy: float
    softness: dict
    history: list = field(default_factory=list)


def softness_report(pre: np.ndarray) -> dict:
    """The FINDINGS.md 20 near-tie profile of a pre-activation matrix.

    The separating statistic is the low quantile of |pre| against the median:
    the GD edge state sits at q0.5% = 0.0011 with median 0.98 (ratio ~1.1e-3),
    the digit construction ~12x farther out. The ratio is scale-invariant, so
    it compares states regardless of gauge.
    """
    a = np.abs(pre)
    q005, median = (float(q) for q in np.quantile(a, (0.005, 0.5)))
    return {
        "q005": q005,
        "median": median,
        "ratio": q005 / median if median > 0 else float("nan"),
        "density": float((pre > 0).mean()),
    }


def _capped_step(pre: np.ndarray, step_pre: np.ndarray, flip_cap: float) -> float:
    """Longest step fraction along step_pre flipping at most flip_cap of the
    signs -- the same order statistic as twosided's rule, numpy edition."""
    crossing = ((pre > 0) & (step_pre < 0)) | ((pre <= 0) & (step_pre > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(crossing, -pre / step_pre, np.inf)
    inside = ratio[ratio <= 1.0]
    k = max(1, int(flip_cap * pre.size))
    if inside.size < k:
        return 1.0
    return float(np.partition(inside, k - 1)[k - 1])


def _fit_readout(h: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Multi-output ridge of the one-hot targets on the activations."""
    d = h.shape[1]
    G = h.T @ h + lam * np.eye(d)
    return np.linalg.solve(G, h.T @ Y).T  # (n_labels, d)


def _soften(u, v, first, second, target_ratio, band_q, mu=1e-6):
    """Compress the near-zero |pre| band by a per-column ridge solve.

    Each column j is independent: a band cell (f, j) contributes one row
    delta_u[a_f, j] + delta_v[b_f, j] = (c - 1) * pre[f, j], where c < 1 is
    the compression that takes the current q0.5%/median ratio to the target.
    The compression is sign-preserving (cells scale toward zero, never
    across), and the moved cells are the ones already within ~1% of zero, so
    off-band cells sharing their tokens shift by less than the band width --
    the fit is untouched to first order. Minimum-norm via a tiny ridge."""
    pre = u[first] + v[second]
    a = np.abs(pre)
    q005, median = np.quantile(a, (0.005, 0.5))
    if median == 0 or q005 == 0:
        return u, v
    c = target_ratio * median / q005
    if c >= 1.0:
        return u, v  # already at least as soft as requested
    band = a <= np.quantile(a, band_q)
    n_vocab, d = u.shape
    u, v = u.copy(), v.copy()
    for j in range(d):
        f_idx = np.flatnonzero(band[:, j])
        if len(f_idx) == 0:
            continue
        M = np.zeros((len(f_idx), 2 * n_vocab))
        rows = np.arange(len(f_idx))
        M[rows, first[f_idx]] = 1.0
        M[rows, n_vocab + second[f_idx]] = 1.0
        A = M.T @ M + mu * np.eye(2 * n_vocab)
        delta = np.linalg.solve(A, M.T @ ((c - 1.0) * pre[f_idx, j]))
        u[:, j] += delta[:n_vocab]
        v[:, j] += delta[n_vocab:]
    return u, v


def solve_soft_seed(
    shape: ModelShape,
    facts: dict,
    params: SoftSeedParams,
    seed: int,
    verbose: bool = False,
) -> SoftSeedResult:
    """Freeze the active set, ridge toward one-hot targets, cap the step,
    repeat -- returning a seed normalized to median |pre| = 1."""
    n_vocab, d, n_labels = (shape.input_vocab_size, shape.d_mlp,
                            shape.output_vocab_size)
    inputs = facts["inputs"].numpy()
    targets = facts["targets"].numpy()
    first, second = inputs[:, 0], inputs[:, 1]
    n = len(targets)

    rng = np.random.default_rng(seed)
    offset = float(ndtri(params.rho)) / 2.0**0.5
    u = rng.standard_normal((n_vocab, d)) + offset
    v = rng.standard_normal((n_vocab, d)) + offset
    pre = u[first] + v[second]
    scale = 1.0 / float(np.median(np.abs(pre)))
    u, v, pre = u * scale, v * scale, pre * scale
    pattern0 = pre > 0

    Y = np.zeros((n, n_labels))
    Y[np.arange(n), targets] = params.target

    # Facts bucketed by each side's token, padded to a rectangle (twosided's
    # pad_group, numpy edition). Static across rounds; padded rows are all
    # zero, so they drop out of every Gram and right-hand side below.
    pads = []
    for tok in (first, second):
        groups = [np.flatnonzero(tok == t) for t in range(n_vocab)]
        width = max((len(g) for g in groups), default=1)
        index = np.zeros((n_vocab, width), dtype=np.int64)
        valid = np.zeros((n_vocab, width, 1))
        for t, g in enumerate(groups):
            index[t, : len(g)] = g
            valid[t, : len(g), 0] = 1.0
        pads.append((index, valid))

    W = _fit_readout(np.maximum(pre, 0.0), Y, params.lam)
    history: list = []
    for r in range(params.rounds):
        P = pre > 0
        Pf = P.astype(np.float64)
        W = _fit_readout(np.where(P, pre, 0.0), Y, params.lam)
        Gw = W.T @ W

        # Tokens on one side are independent blocks: a fact has exactly one
        # token per side, so with the other embedding and W fixed each side
        # is an exact simultaneous ridge solve -- one batched call. The
        # normal matrices A_t = Gw o (P_t' P_t) + mu I depend only on the
        # round-frozen pattern and readout, so each side's batch is
        # Cholesky-factored once (SPD: Schur product of PSD Grams, plus the
        # ridge) and reused across the sweeps.
        factors = []
        for index, valid in pads:
            padded = Pf[index] * valid  # (n_vocab, width, d)
            A = Gw * (padded.transpose(0, 2, 1) @ padded)
            A[:, np.arange(d), np.arange(d)] += params.mu
            factors.append(torch.linalg.cholesky(torch.from_numpy(A)))

        YW = Y @ W  # round-constant: (Y - masked W') W = YW - masked Gw
        held_u, held_v = u.copy(), v.copy()
        # Each block solve changes one embedding, so the masked decode is
        # carried forward with a single gather per solve instead of being
        # rebuilt from both embeddings (twosided's carry-forward trick).
        masked = np.where(P, pre, 0.0)
        for _ in range(params.sweeps):
            for side, emb, tok in ((0, u, first), (1, v, second)):
                index, valid = pads[side]
                RWP = (YW - masked @ Gw) * Pf  # (n, d) residuals through W
                B = (RWP[index] * valid).sum(1)  # (n_vocab, d)
                delta = torch.cholesky_solve(
                    torch.from_numpy(B[:, :, None]), factors[side]
                ).numpy()[:, :, 0]
                emb += delta
                masked += delta[tok] * Pf

        step_pre = (u[first] + v[second]) - pre
        t_step = _capped_step(pre, step_pre, params.flip_cap)
        u = held_u + t_step * (u - held_u)
        v = held_v + t_step * (v - held_v)
        pre = pre + t_step * step_pre

        h = np.maximum(pre, 0.0)
        acc = float(((h @ W.T).argmax(1) == targets).mean())
        soft = softness_report(pre)
        history.append({"round": r, "acc": acc, "step": t_step,
                        "net_drift": float(((pre > 0) != pattern0).mean()),
                        **soft})
        if verbose:
            print(f"    r{r:2d} acc={acc:.4f} step={t_step:.3f} "
                  f"q005/med={soft['ratio']:.2e} dens={soft['density']:.3f}",
                  flush=True)
        if params.acc_stop is not None and acc >= params.acc_stop:
            break

    if params.soften_ratio is not None:
        u, v = _soften(u, v, first, second, params.soften_ratio,
                       params.soften_q)
        pre = u[first] + v[second]

    # Re-fix the gauge (capped steps drift the scale) and refit the readout
    # at that scale, so the handed-off state is self-consistent.
    scale = 1.0 / float(np.median(np.abs(pre)))
    u, v, pre = u * scale, v * scale, pre * scale
    W = _fit_readout(np.maximum(pre, 0.0), Y, params.lam)
    if params.w_rms is not None:
        W *= params.w_rms / float(np.sqrt((W**2).mean()))
    acc = float(((np.maximum(pre, 0.0) @ W.T).argmax(1) == targets).mean())
    return SoftSeedResult(u=u, v=v, W=W, accuracy=acc,
                          softness=softness_report(pre), history=history)
