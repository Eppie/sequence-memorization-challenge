"""Can the geometry be discovered without gradient descent?

The max-margin LPs settled that margins are free once the (pattern, readout)
geometry is adapted to the fact set, and that no generic geometry works. This
probe tests whether a *constructive* pipeline can discover an adapted
geometry:

    twosided solve  ->  gate      (the frozen-pattern ridge iteration already
                                   adapts the sign pattern to the facts)
    ridge fit       ->  readout   (high-rank, the shape trained readouts have)
    max-margin LP   ->  embeddings under that gate and readout
    refit ridge     ->  better readout on the re-solved activations
    ... alternate, ending on an LP step.

Every step is a ridge regression or a linear program; no gradient of any loss
is computed. If the result approaches the trained model's (capacity, sigma90)
point, geometry discovery -- the one thing left attributed to gradient
descent -- is constructive too. If it stalls, the gap between this gate and
the trained gate is the measurable thing gradient descent's early dynamics
buy.

Conditions at d=32, n=2080 (trained capacity):

  twosided+ridge    the full pipeline above
  twosided+random   ablation: same gate, random codebook (no readout fit)
  trained+ridge     calibration: the trained model's own gate, but with a
                    ridge-fit readout instead of its trained one -- how much
                    of the geometry's quality is the gate vs the readout fit
  gate drift        how far the trained gate actually moves from its random
                    init (Hamming), bounding how much discovery GD performs

    uv run python probe_remargin.py
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.model import ModelShape, accuracy, hidden_activations, random_init
from handcode.readouts import RIDGE_ALPHAS, ridge_down
from handcode.twosided import MU_VALUES, RHO_VALUES, TwoSidedParams, solve_two_sided
from probe_digitcode import noise_curve, sigma90
from probe_maxmargin import solve_max_margin, to_model
from probe_robustness import margin_stats, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BOX = 6.0  # trained-scale weight box for every LP


def lp_with_hints(pattern, inputs, targets, readout, n_vocab, logits_hint,
                  k0=12, max_rounds=4):
    """Cutting-plane max-margin with confusability hints from given logits."""
    n = len(targets)
    hint = logits_hint.copy()
    hint[np.arange(n), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]
    for _ in range(max_rounds):
        u, v, gamma, info = solve_max_margin(
            pattern, inputs, targets, readout, n_vocab, BOX, wrong_sets=wrong_sets
        )
        if u is None:
            return None, None, float("-inf"), info
        h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
        logits = h @ readout.T
        correct = logits[np.arange(n), targets]
        logits[np.arange(n), targets] = -np.inf
        true_gamma = float((correct - logits.max(1)).min())
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


def best_ridge(hidden: torch.Tensor, targets: torch.Tensor, n_classes: int):
    """Ridge readout over the alpha sweep, best by its own argmax accuracy."""
    best = None
    for alpha in RIDGE_ALPHAS:
        down = ridge_down(hidden, targets, n_classes, alpha)
        acc = float((hidden @ down.T).argmax(1).eq(targets).float().mean())
        if best is None or acc > best[1]:
            best = (down, acc)
    return best


def alternate(pattern, facts, shape, readout0, rounds=3, label=""):
    """LP embeddings <-> ridge readout, ending on an LP step."""
    inputs, targets = facts["inputs"].numpy(), facts["targets"].numpy()
    tt = facts["targets"]
    readout = readout0
    u = v = None
    history = []
    for r in range(rounds):
        if u is None:
            hint = np.zeros((len(targets), shape.output_vocab_size))
        else:
            h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
            hint = h @ readout.T
        u, v, gamma, info = lp_with_hints(
            pattern, inputs, targets, readout, shape.input_vocab_size, hint
        )
        if u is None:
            history.append({"round": r, "gamma": None, "note": "LP failed"})
            break
        h = torch.from_numpy(pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])).float()
        acc = float((h @ torch.from_numpy(readout).float().T)
                    .argmax(1).eq(tt).float().mean())
        history.append({"round": r, "gamma": gamma, "accuracy": acc,
                        "lp_seconds": info["seconds"]})
        print(f"  [{label}] round {r}: gamma={gamma:.4f} acc={acc:.4f} "
              f"({info['seconds']}s)", flush=True)
        if r < rounds - 1:
            new_down, ridge_acc = best_ridge(h, tt, shape.output_vocab_size)
            readout = new_down.numpy().astype(np.float64)
            print(f"  [{label}] refit ridge: own-acc={ridge_acc:.4f}", flush=True)
    return u, v, readout, history


def evaluate(u, v, readout, facts, label):
    up, down = to_model(u, v, readout)
    entry = {
        "accuracy": accuracy(up, down, facts),
        "margins": margin_stats(up, down, facts),
        "sigma90_weight": sigma90(noise_curve(up, down, facts)),
        "rms_up": rms(up),
    }
    s90 = entry["sigma90_weight"]
    print(f"{label:20s} acc={entry['accuracy']:.4f} "
          f"margin med={entry['margins']['median']:.3g} "
          f"min={entry['margins']['min']:.3g} "
          f"sigma90={s90 if s90 is None else f'{s90:.2e}'}", flush=True)
    return entry


def main() -> None:
    shape = ModelShape.from_d(32)
    n = 2080
    facts = generate_facts(n, shape.input_vocab_size, shape.output_vocab_size, 42)
    out = {"d": 32, "n_facts": n}

    # -- gate drift of an actual training run --------------------------------
    up0, down0 = random_init(shape, 1000)
    with torch.no_grad():
        init_pattern = hidden_activations(up0, facts["inputs"]) > 0
    up_t = up0.clone().requires_grad_(True)
    down_t = down0.clone().requires_grad_(True)
    opt = torch.optim.Adam([up_t, down_t], lr=1e-2)
    for _ in range(5000):
        opt.zero_grad()
        logits = hidden_activations(up_t, facts["inputs"]) @ down_t.T
        F.cross_entropy(logits, facts["targets"]).backward()
        opt.step()
        with torch.no_grad():
            if float((logits.argmax(-1) == facts["targets"]).float().mean()) == 1.0:
                break
    with torch.no_grad():
        trained_pattern = hidden_activations(up_t, facts["inputs"]) > 0
    drift = float((trained_pattern != init_pattern).float().mean())
    out["gate_drift_from_init"] = drift
    print(f"trained gate vs its init: {drift:.3f} of bits flipped\n", flush=True)

    # -- geometry candidates -------------------------------------------------
    print("building twosided gate at n=2080 ...", flush=True)
    best_sol = None
    for rho in RHO_VALUES:
        for mu in MU_VALUES:
            sol = solve_two_sided(
                shape, facts, TwoSidedParams(rho=rho, mu=mu), seed=1000
            )
            if best_sol is None or sol.accuracy > best_sol.accuracy:
                best_sol = sol
            if sol.accuracy == 1.0:
                break
        if best_sol.accuracy == 1.0:
            break
    # The twosided pattern spans the d-1 value neurons plus the bias neuron,
    # always on: reconstruct the full-width pattern.
    pre = best_sol.u[facts["inputs"][:, 0]] + best_sol.v[facts["inputs"][:, 1]]
    ts_pattern = np.concatenate(
        [(pre > 0).numpy(), np.ones((n, 1), dtype=bool)], axis=1
    )
    h_ts = torch.from_numpy(
        ts_pattern * np.concatenate(
            [pre.numpy(), np.full((n, 1), float(shape.d_mlp))], axis=1
        )
    ).float()
    ridge_ts, ridge_ts_acc = best_ridge(h_ts, facts["targets"], shape.output_vocab_size)
    print(f"twosided gate built (acc={best_sol.accuracy:.4f}); "
          f"ridge readout on its own h: acc={ridge_ts_acc:.4f}", flush=True)

    gen = torch.Generator().manual_seed(11)
    random_codebook = torch.randn(
        shape.output_vocab_size, shape.d_mlp, generator=gen
    ).numpy().astype(np.float64)

    h_tr = (hidden_activations(up_t.detach(), facts["inputs"])).detach()
    ridge_tr, ridge_tr_acc = best_ridge(h_tr, facts["targets"], shape.output_vocab_size)
    print(f"trained gate; ridge readout on its h: acc={ridge_tr_acc:.4f}\n", flush=True)

    conditions = {
        "twosided+ridge": (ts_pattern, ridge_ts.numpy().astype(np.float64)),
        "twosided+random": (ts_pattern, random_codebook),
        "trained+ridge": (trained_pattern.numpy(), ridge_tr.numpy().astype(np.float64)),
    }
    for name, (pat, w0) in conditions.items():
        print(f"== {name}", flush=True)
        u, v, w, history = alternate(pat, facts, shape, w0, rounds=3, label=name)
        entry = {"history": history}
        if u is not None and history and history[-1].get("gamma", -1) is not None \
                and history[-1]["gamma"] > 0:
            entry["metrics"] = evaluate(u, v, w, facts, name)
        out[name] = entry

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "remargin.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/remargin.json")


if __name__ == "__main__":
    main()
