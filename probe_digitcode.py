"""The capacity-robustness frontier of the digit-code family, vs trained.

For each redundancy m (digits per label, equations per fact), find how many
facts the grouped digit code can store at acc=1 and how much weight noise the
result tolerates; train a model to acc=1 at the same fact count and measure
the same. One point per (m, n); m = 1 is `twosided` itself, so its measured
numbers (capacity 3168, sigma90 1.6e-5 at d=32) anchor that end.

If the family's frontier passes through the trained model's (capacity,
sigma90) point at some m, the trained code has a constructive description.
If the frontier stays below the trained point everywhere, the gap measures
what equality-constrained storage cannot buy at any redundancy.

    uv run python probe_digitcode.py --d 32
"""

import argparse
import json
import math
import os

import torch

from handcode.data import generate_facts
from handcode.digitcode import DigitCodeParams, assemble, solve_digit_code
from handcode.model import ModelShape, accuracy
from probe_robustness import build_trained, margin_stats, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SIGMAS = [10 ** (e / 2.0) for e in range(-10, 1)]  # 1e-5 .. 1, half decades
N_TRIALS = 10

MS = (2, 3, 5)
FRACTIONS = (0.5, 0.65, 0.8, 0.9)
CONFIGS = [
    dict(mu=mu, group_seed=g)
    for mu in (1e-2, 1.0)
    for g in (0, 1)
]
SEEDS = (1000, 1001)


def sigma90(curve: list[float]) -> float | None:
    for i, acc in enumerate(curve):
        if acc < 0.9:
            if i == 0:
                return SIGMAS[0]
            lo, hi = SIGMAS[i - 1], SIGMAS[i]
            frac = (curve[i - 1] - 0.9) / (curve[i - 1] - curve[i])
            return float(
                10 ** (math.log10(lo) + frac * (math.log10(hi) - math.log10(lo)))
            )
    return None


def noise_curve(up, down, facts, seed: int = 0) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    s_up, s_down = rms(up), rms(down)
    curve = []
    for sigma in SIGMAS:
        accs = []
        for _ in range(N_TRIALS):
            noisy_up = up + torch.randn(up.shape, generator=gen) * (sigma * s_up)
            noisy_down = down + torch.randn(down.shape, generator=gen) * (sigma * s_down)
            accs.append(accuracy(noisy_up, noisy_down, facts))
        curve.append(sum(accs) / len(accs))
    return curve


def build_digit(shape, facts, m: int, rounds: int):
    best = None
    for config in CONFIGS:
        for seed in SEEDS:
            params = DigitCodeParams(m=m, rounds=rounds, sweeps=8, **config)
            sol = solve_digit_code(shape, facts, params, seed)
            if best is None or sol.accuracy > best[0].accuracy:
                best = (sol, params)
            if sol.accuracy == 1.0:
                return sol, params
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "digitcode_frontier.json"))
    args = parser.parse_args()

    shape = ModelShape.from_d(args.d)
    n_unknowns = 2 * (shape.d_mlp - 1) * shape.input_vocab_size
    records = []

    for m in MS:
        ceiling = n_unknowns // m
        best_capacity = None
        for frac in FRACTIONS:
            n = int(ceiling * frac) // 8 * 8
            facts = generate_facts(
                n, shape.input_vocab_size, shape.output_vocab_size, 42
            )
            sol, params = build_digit(shape, facts, m, args.rounds)
            row = {
                "m": m, "p": sol.p, "n_facts": n, "fraction": frac,
                "ceiling": ceiling, "accuracy": sol.accuracy,
                "params": str(params), "round": sol.round_index,
            }
            if sol.accuracy == 1.0:
                up, down = assemble(shape, sol)
                curve = noise_curve(up, down, facts)
                row["sigma90_weight"] = sigma90(curve)
                row["weight_noise"] = curve
                row["margin"] = margin_stats(up, down, facts)
                row["rms_up"] = rms(up)
                best_capacity = row
            records.append(row)
            s90 = row.get("sigma90_weight")
            print(
                f"m={m} (p={sol.p}) n={n} ({frac:.0%} of {ceiling}): "
                f"acc={sol.accuracy:.4f}"
                + (f" sigma90={s90:.2e}" if s90 else ""),
                flush=True,
            )

        if best_capacity is not None:
            n = best_capacity["n_facts"]
            facts = generate_facts(
                n, shape.input_vocab_size, shape.output_vocab_size, 42
            )
            weights, clean = build_trained(shape, facts)
            curve = noise_curve(*weights, facts)
            trained_row = {
                "m": m, "n_facts": n, "condition": "trained-at-same-n",
                "accuracy": clean, "sigma90_weight": sigma90(curve),
                "weight_noise": curve, "margin": margin_stats(*weights, facts),
            }
            records.append(trained_row)
            print(
                f"  trained at n={n}: acc={clean:.4f} "
                f"sigma90={trained_row['sigma90_weight']:.2e}",
                flush=True,
            )

    payload = {
        "d": args.d, "sigmas": SIGMAS, "n_trials": N_TRIALS,
        "n_unknowns": n_unknowns, "records": records,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
