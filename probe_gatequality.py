"""What makes a gate good, and can that be measured from the pattern alone?

The construction problem is reduced to one object (FINDINGS.md 13): the LP
machinery reproduces trained robustness when handed the trained gate, and
stalls ~25x short on any solve-produced gate. So some property of the sign
pattern itself separates them. This probe defines candidate gate-quality
metrics computable from the pattern and the fact list alone -- no labels, no
values, no readout -- and tests them in three stages:

  metrics     compute the candidates on a zoo of gates at one load
              (d=32, n=1584): trained (full budget and first-acc=1),
              twosided, digit m=2 (pedestal-optimized), a density-matched
              random additive gate, and the training init's own gate.
              A useful metric must separate trained from the solves.

  predict     run the two-LP max-margin ascent (probe_geometry_ascent) on
              every gate in the zoo and measure sigma90 of the best point.
              A useful metric must *order* the gates the way sigma90 does.

  construct   build a gate that optimizes the surviving metric -- the
              metric never sees the labels, so this is gate construction
              from the fact inputs alone -- and see where its sigma90 lands.

Candidate metrics, both aimed at the same suspicion (the additive threshold
structure forces facts sharing a token into correlated active sets, and the
solves' pedestals make this much worse; interference then adds coherently
and the per-token value solves are ill-conditioned):

  same-token correlation   mean Pearson correlation between the pattern rows
                           of facts sharing a token, against a cross-token
                           baseline. Random additive gates sit near 1/3 by a
                           Gaussian calculation; a pedestal pushes it toward
                           1; decorrelation pushes it toward 0.

  per-token conditioning   each token's facts form an (n_t x d) 0/1 design
                           block; its singular-value spread (kappa, smin,
                           effective rank) governs how large the solved
                           values must be to hit any given targets.

    uv run python probe_gatequality.py --phase metrics
    uv run python probe_gatequality.py --phase predict
    uv run python probe_gatequality.py --phase drift --pressure spread
    uv run python probe_gatequality.py --phase curve --workers 11
"""

import argparse
import json
import os

import numpy as np
import torch

from handcode.data import generate_facts
from handcode.digitcode import DigitCodeParams
from handcode.digitcode import assemble as digit_assemble
from handcode.digitcode import solve_digit_code
from handcode.model import (
    ModelShape,
    accuracy,
    hidden_activations,
    random_init,
)
from handcode.twosided import MU_VALUES, RHO_VALUES, TwoSidedParams
from handcode.twosided import assemble as twosided_assemble
from handcode.twosided import solve_two_sided
from probe_digitcode import noise_curve, sigma90
from probe_geometry_ascent import embeddings_lp, margins_of, readout_lp
from probe_maxmargin import to_model
from probe_reachability import run_adam
from probe_remargin import best_ridge
from probe_robustness import margin_stats, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GATES_DIR = os.path.join(RESULTS_DIR, "gatequality_gates")
OUT_PATH = os.path.join(RESULTS_DIR, "gatequality.json")

D = 32
N_FACTS = 1584  # the digit code's capacity point; trained sigma90 here ~4.3e-2


# --------------------------------------------------------------------------
# gate zoo
# --------------------------------------------------------------------------

def build_trained(shape, facts, full_budget: bool):
    """The post's recipe. full_budget runs all 5000 epochs (margins keep
    growing past first acc=1); otherwise stop at the first 100% epoch."""
    up0, down0 = random_init(shape, 1000)
    if full_budget:
        _, _, up, down = run_adam(up0, down0, facts, n_epochs=5000, lr=1e-2)
    else:
        import torch.nn.functional as F

        up = up0.clone().requires_grad_(True)
        down = down0.clone().requires_grad_(True)
        opt = torch.optim.Adam([up, down], lr=1e-2)
        for _ in range(5000):
            opt.zero_grad()
            logits = hidden_activations(up, facts["inputs"]) @ down.T
            F.cross_entropy(logits, facts["targets"]).backward()
            opt.step()
            with torch.no_grad():
                if float((logits.argmax(-1) == facts["targets"]).float()
                         .mean()) == 1.0:
                    break
        up, down = up.detach(), down.detach()
    with torch.no_grad():
        pattern = (hidden_activations(up, facts["inputs"]) > 0).numpy()
    return pattern, up, down


def build_twosided(shape, facts):
    best = None
    for rho in RHO_VALUES:
        for mu in MU_VALUES:
            sol = solve_two_sided(shape, facts, TwoSidedParams(rho=rho, mu=mu),
                                  seed=1000)
            if best is None or sol.accuracy > best.accuracy:
                best = sol
            if sol.accuracy == 1.0:
                break
        if best.accuracy == 1.0:
            break
    up, down = twosided_assemble(shape, best, 1.0)
    pre = best.u[facts["inputs"][:, 0]] + best.v[facts["inputs"][:, 1]]
    n = len(facts["targets"])
    pattern = np.concatenate([(pre > 0).numpy(), np.ones((n, 1), bool)], axis=1)
    return pattern, up, down


def build_digit(shape, facts):
    sol = solve_digit_code(
        shape, facts, DigitCodeParams(m=2, rounds=1500, sweeps=8, t0_scale=1.0),
        seed=1000,
    )
    up, down = digit_assemble(shape, sol)
    pre = sol.u[facts["inputs"][:, 0]] + sol.v[facts["inputs"][:, 1]]
    n = len(facts["targets"])
    pattern = np.concatenate([(pre > 0).numpy(), np.ones((n, 1), bool)], axis=1)
    return pattern, up, down


def build_random_additive(shape, facts):
    """probe_maxmargin's scratch recipe: density-matched additive gate."""
    gen = torch.Generator().manual_seed(7)
    offset = float(torch.special.ndtri(torch.tensor(0.53))) / 2**0.5
    u0 = torch.randn(shape.input_vocab_size, D, generator=gen) + offset
    v0 = torch.randn(shape.input_vocab_size, D, generator=gen) + offset
    pre = u0[facts["inputs"][:, 0]] + v0[facts["inputs"][:, 1]]
    pattern = (pre > 0).numpy()
    up = torch.cat([u0.T, v0.T], dim=1)
    return pattern, up, None


def build_init(shape, facts):
    """The gate of training's own random init -- what the trained gate
    drifts 41% of its bits away from."""
    up, _ = random_init(shape, 1000)
    with torch.no_grad():
        pattern = (hidden_activations(up, facts["inputs"]) > 0).numpy()
    return pattern, up, None


GATE_BUILDERS = {
    "trained": lambda shape, facts: build_trained(shape, facts, True),
    "trained_early": lambda shape, facts: build_trained(shape, facts, False),
    "twosided": build_twosided,
    "digit_m2": build_digit,
    "random_additive": build_random_additive,
    "init": build_init,
}


def get_gate(name, shape, facts):
    """Build or load a cached (pattern, up, down) triple."""
    os.makedirs(GATES_DIR, exist_ok=True)
    path = os.path.join(GATES_DIR, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        up = torch.from_numpy(z["up"]).float() if z["up"].size else None
        down = torch.from_numpy(z["down"]).float() if z["down"].size else None
        return z["pattern"].astype(bool), up, down
    pattern, up, down = GATE_BUILDERS[name](shape, facts)
    np.savez(
        path,
        pattern=pattern,
        up=up.numpy() if up is not None else np.array([]),
        down=down.numpy() if down is not None else np.array([]),
    )
    return pattern, up, down


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _families(inputs: np.ndarray):
    """Fact-index lists for every (side, token) with at least 2 facts."""
    fams = []
    for side in (0, 1):
        toks = inputs[:, side]
        for t in np.unique(toks):
            idx = np.flatnonzero(toks == t)
            if len(idx) >= 2:
                fams.append(idx)
    return fams


def _row_corr_mean(X: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation between rows (nan-safe)."""
    if len(X) < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(X)
    iu = np.triu_indices(len(X), 1)
    return float(np.nanmean(C[iu]))


def _cross_corr_mean(P, inputs, n_pairs=20000, seed=0):
    """Baseline: mean row correlation over pairs sharing neither token."""
    rng = np.random.default_rng(seed)
    n = len(P)
    f = rng.integers(0, n, n_pairs)
    g = rng.integers(0, n, n_pairs)
    keep = (inputs[f, 0] != inputs[g, 0]) & (inputs[f, 1] != inputs[g, 1]) & (f != g)
    f, g = f[keep], g[keep]
    Xc = P - P.mean(1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1)
    ok = (norms[f] > 0) & (norms[g] > 0)
    f, g = f[ok], g[ok]
    return float(np.mean(np.einsum("ij,ij->i", Xc[f], Xc[g])
                         / (norms[f] * norms[g])))


def _entropy_rank(s: np.ndarray) -> float:
    p = s**2 / (s**2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def gate_metrics(pattern: np.ndarray, inputs: np.ndarray) -> dict:
    P_full = pattern.astype(np.float64)
    col_mean = P_full.mean(0)
    const_mask = (col_mean > 0.99) | (col_mean < 0.01)
    out = {
        "density": float(P_full.mean()),
        "n_const_neurons": int(const_mask.sum()),
    }
    for tag, P in (("all", P_full), ("nonconst", P_full[:, ~const_mask])):
        fams = _families(inputs)
        same = [_row_corr_mean(P[idx]) for idx in fams]
        cross = _cross_corr_mean(P, inputs)
        smin, kappa, erank_frac, overlaps = [], [], [], []
        for idx in fams:
            X = P[idx]
            s = np.linalg.svd(X, compute_uv=False)
            smin.append(float(s[-1]))
            kappa.append(float(s[0] / max(s[-1], 1e-12)))
            erank_frac.append(_entropy_rank(s) / min(X.shape))
            O = X @ X.T
            overlaps.extend(O[np.triu_indices(len(X), 1)].tolist())
        overlaps = np.array(overlaps)
        out[tag] = {
            "same_token_corr": float(np.nanmean(same)),
            "cross_token_corr": cross,
            "excess_corr": float(np.nanmean(same)) - cross,
            "smin_median": float(np.median(smin)),
            "smin_p10": float(np.percentile(smin, 10)),
            "kappa_median": float(np.median(kappa)),
            "erank_frac_median": float(np.median(erank_frac)),
            "overlap_cv": float(overlaps.std() / max(overlaps.mean(), 1e-12)),
        }
    return out


# --------------------------------------------------------------------------
# prediction: the two-LP ascent, snapshotting the best point
# --------------------------------------------------------------------------

def ascend_best(pattern, facts, shape, up0, down0, rounds=3, label=""):
    """probe_geometry_ascent.ascend, but keep the best-margin full model seen
    (the ascent can oscillate; the last state is not always the best)."""
    inputs, targets = facts["inputs"].numpy(), facts["targets"].numpy()

    u0 = up0[:, : shape.input_vocab_size].T.numpy().astype(np.float64)
    v0 = up0[:, shape.input_vocab_size:].T.numpy().astype(np.float64)
    W = down0.numpy().astype(np.float64)
    box_e = max(6.0, 1.05 * float(np.abs(np.concatenate([u0, v0])).max()))
    box_w = max(6.0, 1.05 * float(np.abs(W).max()))

    h = pattern * (u0[inputs[:, 0]] + v0[inputs[:, 1]])
    g0 = float(margins_of(h, W, targets).min())
    print(f"  [{label}] start: min margin {g0:.4f} (box_e={box_e:.0f}, "
          f"box_w={box_w:.0f})", flush=True)

    u, v = u0, v0
    best = {"gamma": g0, "u": u0, "v": v0, "W": W}
    history = [{"step": "start", "gamma": g0}]

    def consider(step, gamma, uu, vv, ww):
        history.append({"step": step, "gamma": gamma})
        if gamma > best["gamma"]:
            best.update({"gamma": gamma, "u": uu.copy(), "v": vv.copy(),
                         "W": ww.copy()})

    for r in range(rounds):
        u_new, v_new, g_e, info = embeddings_lp(
            pattern, inputs, targets, W, shape.input_vocab_size, box_e
        )
        if u_new is None:
            print(f"  [{label}] embeddings-LP failed: {info.get('message')}",
                  flush=True)
            break
        u, v = u_new, v_new
        h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
        consider(f"emb-{r}", g_e, u, v, W)
        print(f"  [{label}] emb-LP {r}: gamma={g_e:.4f}", flush=True)

        W_new, g_w, info_w = readout_lp(h, targets, shape.output_vocab_size,
                                        box_w)
        if W_new is None:
            print(f"  [{label}] readout-LP failed: {info_w.get('message')}",
                  flush=True)
            break
        W = W_new
        consider(f"ro-{r}", g_w, u, v, W)
        print(f"  [{label}] readout-LP {r}: gamma={g_w:.4f}", flush=True)

    return best, history


def evaluate_full(u, v, W, facts, label):
    up, down = to_model(u, v, W)
    entry = {
        "accuracy": accuracy(up, down, facts),
        "margins": margin_stats(up, down, facts),
        "sigma90_weight": sigma90(noise_curve(up, down, facts)),
        "rms_up": rms(up), "rms_down": rms(down),
    }
    s90 = entry["sigma90_weight"]
    print(f"{label:24s} acc={entry['accuracy']:.4f} "
          f"margin min={entry['margins']['min']:.3g} "
          f"sigma90={s90 if s90 is None else f'{s90:.2e}'}", flush=True)
    return entry


def predict_one(name, pattern, up0, down0, facts, shape):
    """Ascend from the gate's own start (or a ridge readout if it has none)
    and measure sigma90 of the best point."""
    if down0 is None:
        u0 = up0[:, : shape.input_vocab_size].T
        v0 = up0[:, shape.input_vocab_size:].T
        pre = u0[facts["inputs"][:, 0]] + v0[facts["inputs"][:, 1]]
        h0 = torch.from_numpy(pattern * pre.numpy()).float()
        down0, ridge_acc = best_ridge(h0, facts["targets"],
                                      shape.output_vocab_size)
        print(f"  [{name}] no readout; ridge on gate-seed h: "
              f"acc={ridge_acc:.4f}", flush=True)
    best, history = ascend_best(pattern, facts, shape, up0, down0, label=name)
    entry = {"history": history, "best_gamma": best["gamma"]}
    entry["metrics"] = evaluate_full(best["u"], best["v"], best["W"], facts,
                                     name)
    return entry


# --------------------------------------------------------------------------
# construct: pattern-drifting max-margin iteration
# --------------------------------------------------------------------------
#
# Every gate that supports storage was built by co-adapting the pattern with
# a value scheme: the ridge solves drift their patterns under *equality*
# pressure (fragile gates), gradient descent drifts under *soft-max-margin*
# pressure (the robust gate). The constructive analog: solve the max-min-
# margin LP with the sign-consistency rows dropped (the masked-linear
# margins, signs free), then step toward the LP point only as far as
# twosided's flip cap allows, re-read the pattern, refit the readout-LP,
# repeat. No gradient of any loss; the search direction is an exact solve.
#
# Two pressure structures, differing only in the emb-LP objective:
#
#   minmax   maximize the shared min margin gamma. All pressure concentrates
#            on the single worst fact; every round spends its full flip quota
#            chasing it (FINDINGS.md 14: churn, then collapse).
#
#   spread   per-fact margin variables m_f, maximize sum m_f with m_f <= tau
#            (free below). Facts at tau stop pulling, so pressure spreads
#            over every fact still below the cap -- the LP analog of the
#            saturating per-fact pull a softmax loss gives gradient descent.

def lp_free(pattern, inputs, targets, readout, n_vocab, box, logits_hint,
            k0=12, max_rounds=2):
    """Cutting-plane max-margin over embeddings, signs free."""
    from probe_maxmargin import solve_max_margin

    n = len(targets)
    hint = logits_hint.copy()
    hint[np.arange(n), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]
    u = v = None
    gamma = float("-inf")
    for _ in range(max_rounds):
        u, v, gamma, info = solve_max_margin(
            pattern, inputs, targets, readout, n_vocab, box,
            wrong_sets=wrong_sets, pattern_rows=False,
        )
        if u is None:
            return None, None, gamma, info
        h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
        logits = h @ readout.T
        correct = logits[np.arange(n), targets]
        logits[np.arange(n), targets] = -np.inf
        gaps = correct[:, None] - logits
        new = 0
        for f in np.flatnonzero((gaps < gamma - 1e-7).any(1)):
            for c in np.flatnonzero(gaps[f] < gamma - 1e-7):
                if c != targets[f] and c not in wrong_sets[f]:
                    wrong_sets[f].append(int(c))
                    new += 1
        if new == 0:
            break
    return u, v, gamma, info


def lp_spread(pattern, inputs, targets, readout, n_vocab, box, logits_hint,
              tau, k0=12, max_rounds=2):
    """Cutting-plane capped-sum margins over embeddings, signs free. Same
    hints and rounds as lp_free; only the objective differs. The cut rule is
    per fact: a left-out class must beat that fact's own solved m_f."""
    from probe_maxmargin import solve_max_margin

    n = len(targets)
    hint = logits_hint.copy()
    hint[np.arange(n), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]
    u = v = m = None
    obj = float("-inf")
    for _ in range(max_rounds):
        u, v, obj, info = solve_max_margin(
            pattern, inputs, targets, readout, n_vocab, box,
            wrong_sets=wrong_sets, pattern_rows=False, spread_tau=tau,
        )
        if u is None:
            return None, None, obj, None, info
        m = info["m"]
        h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
        logits = h @ readout.T
        correct = logits[np.arange(n), targets]
        logits[np.arange(n), targets] = -np.inf
        gaps = correct[:, None] - logits
        new = 0
        for f in np.flatnonzero((gaps < m[:, None] - 1e-7).any(1)):
            for c in np.flatnonzero(gaps[f] < m[f] - 1e-7):
                if c != targets[f] and c not in wrong_sets[f]:
                    wrong_sets[f].append(int(c))
                    new += 1
        if new == 0:
            break
    return u, v, obj, m, info


def capped_step(u, v, u_star, v_star, inputs, flip_cap):
    """Step from (u, v) toward the LP point, capped so at most flip_cap of
    the (fact, neuron) signs flip -- twosided's order-statistic step rule."""
    pre = u[inputs[:, 0]] + v[inputs[:, 1]]
    dpre = (u_star[inputs[:, 0]] + v_star[inputs[:, 1]]) - pre
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = -pre / dpre
    valid = (t_cross > 0) & (t_cross <= 1.0) & np.isfinite(t_cross)
    ts = np.sort(t_cross[valid])
    allowed = int(flip_cap * pre.size)
    t = 1.0 if len(ts) <= allowed else float(ts[allowed])
    return u + t * (u_star - u), v + t * (v_star - v), t, int(min(len(ts), allowed))


def drift(name, facts, shape, rounds, flip_cap=0.02, pressure="minmax",
          tau=0.5):
    """Margin-driven gate drift from a feasible construction's gate."""
    pattern0, up0, down0 = get_gate(name, shape, facts)
    inputs, targets = facts["inputs"].numpy(), facts["targets"].numpy()
    u = up0[:, : shape.input_vocab_size].T.numpy().astype(np.float64)
    v = up0[:, shape.input_vocab_size:].T.numpy().astype(np.float64)
    W = down0.numpy().astype(np.float64)
    box_e = max(6.0, 1.05 * float(np.abs(np.concatenate([u, v])).max()))
    box_w = max(6.0, 1.05 * float(np.abs(W).max()))

    pattern = pattern0.copy()
    history = []
    best = None

    def snapshot(tag):
        h = np.maximum(u[inputs[:, 0]] + v[inputs[:, 1]], 0.0)
        m = margins_of(h, W, targets)
        acc = float(((h @ W.T).argmax(1) == targets).mean())
        up_t, down_t = to_model(u, v, W)
        s90 = sigma90(noise_curve(up_t, down_t, facts))
        drift_frac = float((pattern != pattern0).mean())
        row = {"round": tag, "accuracy": acc, "min_margin": float(m.min()),
               "sigma90": s90, "drift_from_seed": drift_frac}
        history.append(row)
        print(f"  [{name}-drift] {tag}: acc={acc:.4f} "
              f"min_margin={m.min():.4f} sigma90={s90:.2e} "
              f"drift={drift_frac:.3f}", flush=True)
        return row, acc, s90

    row, acc, s90 = snapshot("start")
    best = {"sigma90": s90 if acc == 1.0 else 0.0, "u": u.copy(),
            "v": v.copy(), "W": W.copy(), "round": "start"}

    bad_rounds = 0
    for r in range(rounds):
        h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
        if pressure == "spread":
            u_star, v_star, obj, m, info = lp_spread(
                pattern, inputs, targets, W, shape.input_vocab_size, box_e,
                logits_hint=h @ W.T, tau=tau,
            )
            lp_line = (None if u_star is None else
                       f"lp_mean_m={obj / len(targets):.4f} "
                       f"min_m={m.min():.4f} "
                       f"at_cap={float((m > tau - 1e-6).mean()):.3f}")
        else:
            u_star, v_star, g_free, info = lp_free(
                pattern, inputs, targets, W, shape.input_vocab_size, box_e,
                logits_hint=h @ W.T,
            )
            lp_line = None if u_star is None else f"lp_gamma={g_free:.4f}"
        if u_star is None:
            print(f"  [{name}-drift] LP failed at round {r}: "
                  f"{info.get('message')}", flush=True)
            break
        u, v, t, flips = capped_step(u, v, u_star, v_star, inputs, flip_cap)
        pattern = (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0

        h = np.maximum(u[inputs[:, 0]] + v[inputs[:, 1]], 0.0)
        W_new, g_w, _ = readout_lp(h, targets, shape.output_vocab_size, box_w)
        if W_new is not None:
            W = W_new
        print(f"  [{name}-drift] round {r}: {lp_line} "
              f"step={t:.3f} flips={flips}", flush=True)
        row, acc, s90 = snapshot(r)
        if acc == 1.0 and s90 is not None and s90 > best["sigma90"]:
            best = {"sigma90": s90, "u": u.copy(), "v": v.copy(),
                    "W": W.copy(), "round": r}
        bad_rounds = bad_rounds + 1 if acc < 0.99 else 0
        if bad_rounds >= 3:
            print(f"  [{name}-drift] accuracy collapsed; stopping", flush=True)
            break

    entry = {"history": history, "best_round": best["round"],
             "best_sigma90": best["sigma90"], "pressure": pressure}
    if pressure == "spread":
        entry["tau"] = tau
    entry["best_metrics"] = evaluate_full(best["u"], best["v"], best["W"],
                                          facts, f"{name}-drift best")
    suffix = "_drift_best" if pressure == "minmax" else f"_drift_{pressure}_best"
    np.savez(os.path.join(GATES_DIR, f"{name}{suffix}.npz"),
             u=best["u"], v=best["v"], W=best["W"])
    return entry


# --------------------------------------------------------------------------
# curve: gate quality along the training trajectory
# --------------------------------------------------------------------------
#
# Section 15's relocation result (trained_early ~ trained, init infeasible)
# says the gate's quality is built during fitting. This phase measures when:
# checkpoint the post-recipe training run, freeze each checkpoint's pattern,
# give it the same two-LP ascent the zoo got, and read the sigma90 ceiling
# against epoch. The ceiling is a function of the pattern alone, so the curve
# moves only at flip events; per-epoch flip rate, accuracy, and drift from
# init/final are recorded alongside. Checkpoints are independent, so the
# ascents run in parallel worker processes.

CURVE_EPOCHS = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300,
                500, 750, 1000, 1500, 2000, 3000, 5000)
CURVE_CKPT_PATH = os.path.join(GATES_DIR, "curve_checkpoints.npz")


def train_curve_checkpoints(shape, facts, seed=1000, n_epochs=5000, lr=1e-2):
    """The post's recipe with the zoo's seed, snapshotting (up, down) at
    CURVE_EPOCHS plus the first acc=1 epoch. Accuracy is read from the
    pre-step logits, matching build_trained's early-stop check."""
    import torch.nn.functional as F

    up0, down0 = random_init(shape, seed)
    up = up0.clone().requires_grad_(True)
    down = down0.clone().requires_grad_(True)
    opt = torch.optim.Adam([up, down], lr=lr)
    with torch.no_grad():
        pat_prev = hidden_activations(up, facts["inputs"]) > 0
        logits0 = hidden_activations(up, facts["inputs"]) @ down.T
        acc0 = float((logits0.argmax(-1) == facts["targets"]).float().mean())

    snaps = {0: (up.detach().clone(), down.detach().clone())}
    accs = {0: acc0}
    flip_frac = np.zeros(n_epochs)
    first_acc1 = None
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        logits = hidden_activations(up, facts["inputs"]) @ down.T
        F.cross_entropy(logits, facts["targets"]).backward()
        opt.step()
        with torch.no_grad():
            acc = float((logits.argmax(-1) == facts["targets"]).float().mean())
            pat = hidden_activations(up, facts["inputs"]) > 0
            flip_frac[epoch - 1] = float((pat != pat_prev).float().mean())
            pat_prev = pat
        if first_acc1 is None and acc == 1.0:
            first_acc1 = epoch
            snaps[epoch] = (up.detach().clone(), down.detach().clone())
            accs[epoch] = acc
        if epoch in CURVE_EPOCHS:
            snaps[epoch] = (up.detach().clone(), down.detach().clone())
            accs[epoch] = acc
        if epoch % 500 == 0:
            print(f"  [curve-train] epoch {epoch}: acc={acc:.4f} "
                  f"flip={flip_frac[epoch - 1]:.4f}", flush=True)

    os.makedirs(GATES_DIR, exist_ok=True)
    arrays = {"epochs": np.array(sorted(snaps)),
              "acc": np.array([accs[e] for e in sorted(snaps)]),
              "flip_frac": flip_frac,
              "first_acc1": np.array(first_acc1 if first_acc1 else -1)}
    for e, (u_t, d_t) in snaps.items():
        arrays[f"up_{e}"] = u_t.numpy()
        arrays[f"down_{e}"] = d_t.numpy()
    np.savez(CURVE_CKPT_PATH, **arrays)
    print(f"  [curve-train] first acc=1 at epoch {first_acc1}; "
          f"{len(snaps)} checkpoints -> {CURVE_CKPT_PATH}", flush=True)


def curve_ascend_worker(epoch: int) -> dict:
    """Ascend one checkpoint's gate (runs in a worker process)."""
    shape = ModelShape.from_d(D)
    facts = generate_facts(N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, 42)
    z = np.load(CURVE_CKPT_PATH)
    up = torch.from_numpy(z[f"up_{epoch}"]).float()
    down = torch.from_numpy(z[f"down_{epoch}"]).float()
    with torch.no_grad():
        pattern = (hidden_activations(up, facts["inputs"]) > 0).numpy()
    best, history = ascend_best(pattern, facts, shape, up, down,
                                label=f"ep{epoch}")
    entry = {"epoch": epoch, "best_gamma": best["gamma"], "history": history}
    entry["metrics"] = evaluate_full(best["u"], best["v"], best["W"], facts,
                                     f"ep{epoch} ceiling")
    return entry


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def merge_out(update: dict) -> None:
    data = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            data = json.load(f)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",
                        choices=("metrics", "predict", "drift", "curve"),
                        default="metrics")
    parser.add_argument("--gates", nargs="*", default=None)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--pressure", choices=("minmax", "spread"),
                        default="minmax")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()

    shape = ModelShape.from_d(D)
    facts = generate_facts(N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, 42)
    inputs = facts["inputs"].numpy()
    names = args.gates or list(GATE_BUILDERS)

    if args.phase == "metrics":
        rows = {}
        for name in names:
            print(f"== building {name}", flush=True)
            pattern, up, down = get_gate(name, shape, facts)
            rows[name] = gate_metrics(pattern, inputs)
            m = rows[name]["nonconst"]
            print(f"{name:16s} density={rows[name]['density']:.3f} "
                  f"const={rows[name]['n_const_neurons']} | nonconst: "
                  f"same={m['same_token_corr']:.3f} "
                  f"cross={m['cross_token_corr']:.3f} "
                  f"smin_med={m['smin_median']:.2f} "
                  f"kappa_med={m['kappa_median']:.1f} "
                  f"overlap_cv={m['overlap_cv']:.3f}", flush=True)
        merge_out({"d": D, "n_facts": N_FACTS, "metrics": rows})
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "predict":
        rows = {}
        for name in names:
            print(f"== {name}", flush=True)
            pattern, up, down = get_gate(name, shape, facts)
            if name in ("trained", "trained_early"):
                rows[name + "_model_direct"] = evaluate_full(
                    up[:, : shape.input_vocab_size].T.numpy().astype(float),
                    up[:, shape.input_vocab_size:].T.numpy().astype(float),
                    down.numpy().astype(float), facts, name + " (direct)",
                )
            rows[name] = predict_one(name, pattern, up, down, facts, shape)
            merge_out({"predict": dict(rows)})
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "drift":
        key = "drift" if args.pressure == "minmax" else f"drift_{args.pressure}"
        for name in names:
            print(f"== drift from {name} ({args.pressure})", flush=True)
            entry = drift(name, facts, shape, rounds=args.rounds,
                          pressure=args.pressure, tau=args.tau)
            merge_out({key: {name: entry}})
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "curve":
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if not os.path.exists(CURVE_CKPT_PATH):
            train_curve_checkpoints(shape, facts)
        z = np.load(CURVE_CKPT_PATH)
        snap_epochs = [int(e) for e in z["epochs"]]
        accs = {int(e): float(a) for e, a in zip(z["epochs"], z["acc"])}

        def pattern_of(e):
            up = torch.from_numpy(z[f"up_{e}"]).float()
            with torch.no_grad():
                return (hidden_activations(up, facts["inputs"]) > 0).numpy()

        pat_init = pattern_of(0)
        pat_final = pattern_of(snap_epochs[-1])
        drift_stats = {
            e: {"from_init": float((pattern_of(e) != pat_init).mean()),
                "from_final": float((pattern_of(e) != pat_final).mean())}
            for e in snap_epochs
        }
        merge_out({"curve": {
            "first_acc1": int(z["first_acc1"]),
            "flip_frac_per_epoch": [round(float(f), 5) for f in z["flip_frac"]],
        }})

        rows = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(curve_ascend_worker, e): e
                       for e in snap_epochs}
            for fut in as_completed(futures):
                e = futures[fut]
                entry = fut.result()
                entry["train_acc"] = accs[e]
                entry.update(drift_stats[e])
                rows[str(e)] = entry
                merge_out({"curve": {"ascent": dict(rows)}})
                print(f"== ep{e}: ceiling "
                      f"sigma90={entry['metrics']['sigma90_weight']} "
                      f"acc={entry['metrics']['accuracy']:.3f} "
                      f"({len(rows)}/{len(snap_epochs)} done)", flush=True)
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
