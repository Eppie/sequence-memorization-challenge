"""Where the digit code's remaining fragility lives, and how much is removable.

Two measurements on top of `probe_digitcode.py`'s frontier:

  attribution   perturb only the embedding vs only the readout. The digit
                code's sigma90 is readout-dominated: its stabilizer inflates
                the activation norms, so readout noise is amplified by
                ||h|| ~ t0. The trained model is balanced between the two.
  pedestal      sweep the stabilizer down. sigma90 rises roughly as 1/t0
                until the frozen-pattern solve stops converging (t0_scale
                ~ 0.5 at 80% load), and the weights fall to trained scale.

Together with the frontier this decomposes the original ~1300x robustness gap
at matched load: ~2-10x recovered by digit redundancy, ~10-20x by shrinking
the pedestal, and a residual ~30x that no equality-constrained solve in this
family reaches -- the margin-inequality packing gradient descent's implicit
bias performs.

    uv run python probe_pedestal.py
"""

import json
import os

import torch

from handcode.data import generate_facts
from handcode.digitcode import DigitCodeParams, assemble, solve_digit_code
from handcode.model import ModelShape, accuracy
from probe_digitcode import SIGMAS, noise_curve, sigma90
from probe_robustness import build_trained, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sided_sigma90(up, down, facts, side, n_trials=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    s_up, s_down = rms(up), rms(down)
    curve = []
    for s in SIGMAS:
        accs = []
        for _ in range(n_trials):
            nu = up
            nd = down
            if side in ("up", "both"):
                nu = up + torch.randn(up.shape, generator=gen) * (s * s_up)
            if side in ("down", "both"):
                nd = down + torch.randn(down.shape, generator=gen) * (s * s_down)
            accs.append(accuracy(nu, nd, facts))
        curve.append(sum(accs) / len(accs))
    return sigma90(curve)


def best_digit(shape, facts, m, t0_scale, rounds=1500):
    best = None
    for mu in (1e-2, 1.0):
        sol = solve_digit_code(
            shape, facts,
            DigitCodeParams(m=m, rounds=rounds, sweeps=8, t0_scale=t0_scale, mu=mu),
            seed=1000,
        )
        if best is None or sol.accuracy > best.accuracy:
            best = sol
        if best.accuracy == 1.0:
            break
    return best


def main() -> None:
    d = 32
    shape = ModelShape.from_d(d)
    out = {"d": d, "attribution": {}, "pedestal": []}

    facts = generate_facts(1056, shape.input_vocab_size, shape.output_vocab_size, 42)
    sol = best_digit(shape, facts, m=3, t0_scale=16.0, rounds=1000)
    up, down = assemble(shape, sol)
    out["attribution"]["digit m=3 n=1056 t0=16d"] = {
        side: sided_sigma90(up, down, facts, side) for side in ("both", "up", "down")
    }
    weights, clean = build_trained(shape, facts)
    out["attribution"]["trained n=1056"] = {
        side: sided_sigma90(*weights, facts, side) for side in ("both", "up", "down")
    }
    for name, row in out["attribution"].items():
        print(f"{name}: " + "  ".join(f"{k}={v:.2e}" for k, v in row.items()), flush=True)

    for m, n in ((3, 1056), (2, 1584)):
        facts_n = generate_facts(n, shape.input_vocab_size, shape.output_vocab_size, 42)
        for t0_scale in (16.0, 4.0, 2.0, 1.0, 0.5, 0.25):
            sol = best_digit(shape, facts_n, m=m, t0_scale=t0_scale)
            row = {"m": m, "n_facts": n, "t0_scale": t0_scale,
                   "accuracy": sol.accuracy}
            if sol.accuracy == 1.0:
                u, dn = assemble(shape, sol)
                row["sigma90_weight"] = sigma90(noise_curve(u, dn, facts_n))
                row["rms_up"] = rms(u)
            out["pedestal"].append(row)
            s90 = row.get("sigma90_weight")
            print(f"m={m} n={n} t0_scale={t0_scale}: acc={sol.accuracy:.4f}"
                  + (f" s90={s90:.2e} rms_up={row['rms_up']:.1f}" if s90 else ""),
                  flush=True)

    path = os.path.join(RESULTS_DIR, "pedestal.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
