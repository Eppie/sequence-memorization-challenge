"""Does the benchmark reward fragility?

The challenge scores raw argmax accuracy, so nothing stops a construction from
being correct by arbitrarily thin margins. The two value codes here decide
every fact by less than half a logit unit against logits spanning hundreds,
while a trained model at its own capacity holds a median margin of ~8 on
logits of ~30. If part of the hand-coded/trained capacity gap is the trained
model buying robustness the metric never asks for, that is worth knowing --
it says "close the gap" is partly the wrong target, and suggests the metric
should charge for fragility.

The test: perturb the weights with Gaussian noise scaled to each matrix's own
RMS (so the comparison is invariant to the constructions' very different
weight scales), and watch accuracy fall. Every construction is built at its
own acc=1 capacity, and a trained model is trained to acc=1 at the *same*
fact count, so each comparison is load-matched. A second sweep perturbs the
hidden activations instead of the weights, which asks the same question about
interference arriving through the residual stream rather than through the
parameters.

Reported per condition: the noise level at which mean accuracy first drops
below 0.9 ("sigma90", interpolated in log-noise), plus the full curves.

Gradient descent appears here only to build the trained baselines -- this is
analysis, not a challenge entry.

    uv run python probe_robustness.py --d 32
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.handcoded import HandCodedParams, hand_coded_weights
from handcode.linsolve import (
    MU_VALUES as LIN_MU_VALUES,
    T0_SCALES,
    LinsolveParams,
    linsolve_weights,
)
from handcode.model import ModelShape, accuracy, hidden_activations, random_init
from handcode.twosided import (
    MU_VALUES,
    RHO_VALUES,
    TwoSidedParams,
    assemble,
    solve_two_sided,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Measured acc=1 capacities (results/scaling.json) and the sweep winners that
# reached them.
CAPACITIES = {
    32: {"hand-coded": 130, "linsolve": 1408, "trained": 2080, "twosided": 3168},
    64: {"hand-coded": 216, "linsolve": 6016, "trained": 7296, "twosided": 12800},
}
HAND_CODED_WINNERS = {
    32: HandCodedParams(S=10, top_fraction=0.15),
    64: HandCodedParams(S=15, top_fraction=0.2),
}
LINSOLVE_K = 4

SIGMAS = [10 ** (e / 4.0) for e in range(-20, 1)]  # 1e-5 .. 1, quarter decades
N_TRIALS = 20


def rms(w: torch.Tensor) -> float:
    return float(w.pow(2).mean().sqrt())


def margin_stats(up: torch.Tensor, down: torch.Tensor, facts: dict) -> dict:
    """Correct-minus-best-wrong logit, over facts; the raw fragility number."""
    with torch.no_grad():
        logits = hidden_activations(up, facts["inputs"]) @ down.T
        correct = logits.gather(1, facts["targets"].unsqueeze(1)).squeeze(1)
        masked = logits.scatter(
            1, facts["targets"].unsqueeze(1), -torch.inf
        )
        margin = correct - masked.max(1).values
        return {
            "median": float(margin.median()),
            "min": float(margin.min()),
            "logit_rms": float(logits.pow(2).mean().sqrt()),
        }


def weight_noise_curve(
    up: torch.Tensor, down: torch.Tensor, facts: dict, seed: int = 0
) -> list[float]:
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


def activation_noise_curve(
    up: torch.Tensor, down: torch.Tensor, facts: dict, seed: int = 0
) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        hidden = hidden_activations(up, facts["inputs"])
    s_h = rms(hidden)
    targets = facts["targets"]
    curve = []
    for sigma in SIGMAS:
        accs = []
        for _ in range(N_TRIALS):
            noisy = hidden + torch.randn(hidden.shape, generator=gen) * (sigma * s_h)
            logits = noisy @ down.T
            accs.append(float((logits.argmax(-1) == targets).float().mean()))
        curve.append(sum(accs) / len(accs))
    return curve


def sigma90(curve: list[float]) -> float | None:
    """Noise level where mean accuracy first crosses 0.9, log-interpolated."""
    for i, acc in enumerate(curve):
        if acc < 0.9:
            if i == 0:
                return SIGMAS[0]
            lo, hi = SIGMAS[i - 1], SIGMAS[i]
            a_lo, a_hi = curve[i - 1], curve[i]
            frac = (a_lo - 0.9) / (a_lo - a_hi)
            return float(10 ** (math.log10(lo) + frac * (math.log10(hi) - math.log10(lo))))
    return None  # never dropped below 0.9 in the swept range


# -- building each condition at 100% clean accuracy ---------------------------


def build_twosided(shape: ModelShape, facts: dict):
    best = None
    for seed in (1000, 1001, 1002):
        for rho in RHO_VALUES:
            for mu in MU_VALUES:
                params = TwoSidedParams(rho=rho, mu=mu)
                solution = solve_two_sided(shape, facts, params, seed)
                weights = assemble(shape, solution, params.delta)
                score = accuracy(*weights, facts)
                if best is None or score > best[0]:
                    best = (score, weights)
                if score == 1.0:
                    return weights, score
    return best[1], best[0]


def build_linsolve(shape: ModelShape, facts: dict):
    best = None
    for seed in (1000, 1001, 1002):
        for mu in LIN_MU_VALUES:
            for t0_scale in T0_SCALES:
                params = LinsolveParams(k=LINSOLVE_K, mu=mu, t0_scale=t0_scale)
                weights = linsolve_weights(shape, facts, params, seed)
                score = accuracy(*weights, facts)
                if best is None or score > best[0]:
                    best = (score, weights)
                if score == 1.0:
                    return weights, score
    return best[1], best[0]


def build_hand_coded(shape: ModelShape, facts: dict, params: HandCodedParams):
    best = None
    for seed in (1000, 1001, 1002):
        weights = hand_coded_weights(shape, facts, params, seed)
        score = accuracy(*weights, facts)
        if best is None or score > best[0]:
            best = (score, weights)
        if score == 1.0:
            return weights, score
    return best[1], best[0]


def build_trained(shape: ModelShape, facts: dict, n_epochs: int = 50000):
    """The post's recipe, run until it actually reaches acc=1 (or gives up).

    Weights are taken at the epoch that reaches 1.0, not after further
    training, so the baseline is the trained model exactly when the benchmark
    would have scored it.
    """
    inputs, targets = facts["inputs"], facts["targets"]
    best = None
    for seed in (1000, 1001, 1002):
        up, down = random_init(shape, seed)
        up = up.requires_grad_(True)
        down = down.requires_grad_(True)
        optimizer = torch.optim.Adam([up, down], lr=1e-2)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            logits = hidden_activations(up, inputs) @ down.T
            F.cross_entropy(logits, targets).backward()
            optimizer.step()
            with torch.no_grad():
                acc = float((logits.argmax(-1) == targets).float().mean())
            if acc == 1.0:
                return (up.detach(), down.detach()), 1.0
        if best is None or acc > best[0]:
            best = (acc, (up.detach(), down.detach()))
    return best[1], best[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.d not in CAPACITIES:
        raise SystemExit(f"capacities and sweep winners are recorded for {sorted(CAPACITIES)}")
    out = args.out or os.path.join(RESULTS_DIR, f"robustness_d{args.d}.json")

    shape = ModelShape.from_d(args.d)
    torch.manual_seed(0)
    capacity = CAPACITIES[args.d]

    # Each construction at its own acc=1 capacity, and a trained model at the
    # same fact count -- load-matched pairs at every n.
    plan = [
        ("hand-coded", capacity["hand-coded"]),
        ("trained", capacity["hand-coded"]),
        ("linsolve", capacity["linsolve"]),
        ("trained", capacity["linsolve"]),
        ("trained", capacity["trained"]),
        ("twosided", capacity["trained"]),
        ("twosided", capacity["twosided"]),
    ]

    records = []
    for condition, n_facts in plan:
        facts = generate_facts(
            n_facts, shape.input_vocab_size, shape.output_vocab_size, 42
        )
        if condition == "twosided":
            weights, clean = build_twosided(shape, facts)
        elif condition == "linsolve":
            weights, clean = build_linsolve(shape, facts)
        elif condition == "hand-coded":
            weights, clean = build_hand_coded(shape, facts, HAND_CODED_WINNERS[args.d])
        else:
            weights, clean = build_trained(shape, facts)
        up, down = weights

        margins = margin_stats(up, down, facts)
        w_curve = weight_noise_curve(up, down, facts)
        a_curve = activation_noise_curve(up, down, facts)
        record = {
            "condition": condition,
            "n_facts": n_facts,
            "clean_accuracy": clean,
            "margin": margins,
            "rms_up": rms(up),
            "rms_down": rms(down),
            "weight_noise": w_curve,
            "activation_noise": a_curve,
            "sigma90_weight": sigma90(w_curve),
            "sigma90_activation": sigma90(a_curve),
        }
        records.append(record)
        s90w = record["sigma90_weight"]
        s90a = record["sigma90_activation"]
        print(
            f"{condition:>10s} n={n_facts:<5d} clean={clean:.4f} "
            f"margin={margins['median']:.3g} (min {margins['min']:.3g}) "
            f"sigma90 weight={s90w if s90w is None else f'{s90w:.2e}'} "
            f"activation={s90a if s90a is None else f'{s90a:.2e}'}",
            flush=True,
        )

    payload = {"d": args.d, "sigmas": SIGMAS, "n_trials": N_TRIALS, "records": records}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
