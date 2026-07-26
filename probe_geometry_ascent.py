"""Coordinate max-margin ascent: two LPs, no gradients, monotone margins.

`probe_remargin.py` showed the naive pipeline dies at the first joint: a
ridge-fit readout supports no positive margin even on the *trained* gate, and
coordinate ascent cannot bootstrap from an infeasible point. The fix is to
never leave the feasible region. Both constructions come with a readout that
already decodes their gate at 100% accuracy (thin margins) -- so start there
and alternate two exact solves:

    embeddings-LP   max-min-margin over (u, v), pattern and readout frozen
                    (the probe_maxmargin machinery);
    readout-LP      max-min-margin over W, activations frozen -- also linear,
                    and much smaller (d*L + 1 variables).

Each half-step can only raise the min margin; both end states are consistent
full models. The open question this answers: with the *gate* fixed to what a
ridge-only construction discovered, how far up the robustness axis can exact
margin ascent climb? If it approaches the trained sigma90 at the same load,
geometry discovery reduces to the frozen-pattern solve plus margin ascent --
all constructive. If it stalls, the ceiling measures what is missing from
these gates that gradient descent's slow 41%-drift consolidation builds in.

    uv run python probe_geometry_ascent.py
"""

import json
import os

import numpy as np
import torch

from handcode.data import generate_facts
from handcode.digitcode import DigitCodeParams
from handcode.digitcode import assemble as digit_assemble
from handcode.digitcode import solve_digit_code
from handcode.model import ModelShape, accuracy
from handcode.twosided import MU_VALUES, RHO_VALUES, TwoSidedParams
from handcode.twosided import assemble as twosided_assemble
from handcode.twosided import solve_two_sided
from probe_digitcode import noise_curve, sigma90
from probe_maxmargin import solve_max_margin, to_model
from probe_robustness import margin_stats, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def margins_of(h: np.ndarray, readout: np.ndarray, targets: np.ndarray):
    logits = h @ readout.T
    n = len(targets)
    correct = logits[np.arange(n), targets]
    logits = logits.copy()
    logits[np.arange(n), targets] = -np.inf
    return correct - logits.max(1)


def embeddings_lp(pattern, inputs, targets, readout, n_vocab, box, k0=12,
                  max_rounds=4):
    """Max-min-margin over embeddings; hints from the current readout."""
    n = len(targets)
    # Feasible current point exists, so hint with its logits once available;
    # first call hints via readout row similarity through the pattern mass.
    hint = (pattern.astype(float) @ readout.T)
    hint[np.arange(n), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]
    u = v = None
    for _ in range(max_rounds):
        u, v, gamma, info = solve_max_margin(
            pattern, inputs, targets, readout, n_vocab, box, wrong_sets=wrong_sets
        )
        if u is None:
            return None, None, float("-inf"), info
        h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
        m = margins_of(h, readout, targets)
        true_gamma = float(m.min())
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
            return u, v, true_gamma, info
    return u, v, true_gamma, info


def readout_lp(h: np.ndarray, targets: np.ndarray, n_labels: int, box: float,
               k0=12, max_rounds=4):
    """Max-min-margin over the readout, activations frozen. Small LP:
    variables are W (L*d) and gamma; constraint rows (w_l - w_c) . h_f >= g."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, d = h.shape
    nvar = n_labels * d + 1
    gamma_col = nvar - 1

    # start from the most confusable classes by current-h similarity to the
    # class-mean activation, refined by cutting planes
    means = np.stack([h[targets == c].mean(0) if (targets == c).any()
                      else np.zeros(d) for c in range(n_labels)])
    hint = h @ means.T
    hint[np.arange(n), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]

    W = None
    for _ in range(max_rounds):
        rows, cols, data = [], [], []
        row_count = 0
        for f in range(n):
            lab = targets[f]
            wrong = wrong_sets[f]
            hf = h[f]
            nz = np.flatnonzero(hf)
            for c in wrong:
                r = row_count
                # -(w_lab - w_c) . h_f + gamma <= 0
                rows.extend([r] * (2 * len(nz) + 1))
                cols.extend((lab * d + nz).tolist())
                data.extend((-hf[nz]).tolist())
                cols.extend((c * d + nz).tolist())
                data.extend(hf[nz].tolist())
                cols.append(gamma_col)
                data.append(1.0)
                row_count += 1
        A = sparse.csr_matrix(
            (np.array(data), (np.array(rows), np.array(cols))),
            shape=(row_count, nvar),
        )
        c_obj = np.zeros(nvar)
        c_obj[gamma_col] = -1.0
        bounds = [(-box, box)] * (nvar - 1) + [(None, None)]
        res = linprog(c_obj, A_ub=A, b_ub=np.zeros(row_count), bounds=bounds,
                      method="highs-ipm", options={"time_limit": 300})
        if res.status != 0:
            return W, float("-inf"), {"status": res.status, "message": res.message}
        W = res.x[: n_labels * d].reshape(n_labels, d)
        gamma = float(-res.fun)
        m = margins_of(h, W, targets)
        true_gamma = float(m.min())
        logits = h @ W.T
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
            return W, true_gamma, {"status": 0}
    return W, true_gamma, {"status": 0}


def ascend(pattern, facts, shape, up0, down0, rounds=3, label=""):
    inputs, targets = facts["inputs"].numpy(), facts["targets"].numpy()
    n = len(targets)
    d = shape.d_mlp

    # Current feasible point: the construction itself.
    u0 = up0[:, : shape.input_vocab_size].T.numpy().astype(np.float64)
    v0 = up0[:, shape.input_vocab_size:].T.numpy().astype(np.float64)
    W = down0.numpy().astype(np.float64)
    box_e = 1.05 * float(np.abs(np.concatenate([u0, v0])).max())
    box_w = 1.05 * float(np.abs(W).max())

    h = pattern * (u0[inputs[:, 0]] + v0[inputs[:, 1]])
    g0 = float(margins_of(h, W, targets).min())
    print(f"  [{label}] start: min margin {g0:.4f} (box_e={box_e:.0f}, "
          f"box_w={box_w:.0f})", flush=True)

    u, v = u0, v0
    history = [{"step": "start", "gamma": g0}]
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
        history.append({"step": f"emb-{r}", "gamma": g_e})
        print(f"  [{label}] emb-LP {r}: gamma={g_e:.4f}", flush=True)

        W_new, g_w, info_w = readout_lp(h, targets, shape.output_vocab_size, box_w)
        if W_new is None:
            print(f"  [{label}] readout-LP failed: {info_w.get('message')}",
                  flush=True)
            break
        W = W_new
        history.append({"step": f"ro-{r}", "gamma": g_w})
        print(f"  [{label}] readout-LP {r}: gamma={g_w:.4f}", flush=True)

    return u, v, W, history


def evaluate(u, v, W, facts, label):
    up, down = to_model(u, v, W)
    entry = {
        "accuracy": accuracy(up, down, facts),
        "margins": margin_stats(up, down, facts),
        "sigma90_weight": sigma90(noise_curve(up, down, facts)),
        "rms_up": rms(up), "rms_down": rms(down),
    }
    s90 = entry["sigma90_weight"]
    print(f"{label:24s} acc={entry['accuracy']:.4f} "
          f"margin med={entry['margins']['median']:.3g} "
          f"min={entry['margins']['min']:.3g} "
          f"sigma90={s90 if s90 is None else f'{s90:.2e}'}", flush=True)
    return entry


def main() -> None:
    shape = ModelShape.from_d(32)
    out = {"d": 32}

    # -- gate A: the twosided construction at trained capacity ---------------
    n = 2080
    facts = generate_facts(n, shape.input_vocab_size, shape.output_vocab_size, 42)
    best = None
    for rho in RHO_VALUES:
        for mu in MU_VALUES:
            sol = solve_two_sided(shape, facts, TwoSidedParams(rho=rho, mu=mu), 1000)
            if best is None or sol.accuracy > best.accuracy:
                best = sol
            if sol.accuracy == 1.0:
                break
        if best.accuracy == 1.0:
            break
    up0, down0 = twosided_assemble(shape, best, 1.0)
    pre = best.u[facts["inputs"][:, 0]] + best.v[facts["inputs"][:, 1]]
    pattern = np.concatenate([(pre > 0).numpy(), np.ones((n, 1), bool)], axis=1)
    print("== twosided gate @2080 + ladder start", flush=True)
    u, v, W, hist = ascend(pattern, facts, shape, up0, down0, label="twosided@2080")
    out["twosided@2080"] = {"history": hist,
                            "metrics": evaluate(u, v, W, facts, "twosided@2080")}

    # -- gate B: the pedestal-optimized digit code at 1584 -------------------
    n2 = 1584
    facts2 = generate_facts(n2, shape.input_vocab_size, shape.output_vocab_size, 42)
    sol2 = solve_digit_code(
        shape, facts2, DigitCodeParams(m=2, rounds=1500, sweeps=8, t0_scale=1.0),
        seed=1000,
    )
    up2, down2 = digit_assemble(shape, sol2)
    pre2 = sol2.u[facts2["inputs"][:, 0]] + sol2.v[facts2["inputs"][:, 1]]
    pattern2 = np.concatenate([(pre2 > 0).numpy(), np.ones((n2, 1), bool)], axis=1)
    print(f"== digit gate @1584 (built acc={sol2.accuracy:.4f}) + digit start",
          flush=True)
    u2, v2, W2, hist2 = ascend(pattern2, facts2, shape, up2, down2,
                               label="digit@1584")
    out["digit@1584"] = {"history": hist2,
                         "metrics": evaluate(u2, v2, W2, facts2, "digit@1584")}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "geometry_ascent.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/geometry_ascent.json")


if __name__ == "__main__":
    main()
