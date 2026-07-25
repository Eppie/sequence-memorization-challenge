"""How each construction encodes a fact, measured the same way for all of them.

Reproduces the probes behind `FINDINGS.md` and extends them to the two-sided
construction. For each model we take the hidden activations on its own fact
set, coarsen the *magnitudes* while keeping the *pattern*, and retrain a linear
readout on each variant. A model whose accuracy survives binarisation is using
a pattern code; one that needs many magnitude levels is using a value code.

Retraining the readout matters: ridge under-reads badly here, because ridge
optimises L2 while the metric is argmax. Gradient descent is used for these
*probes* only -- that is analysis, not a challenge entry, so the no-gradient
rule does not apply to it.

**The retrained-readout probe is not a valid instrument for a value code.** Its
`full` column reads 0.07-0.10 for `linsolve` and `twosided`, whose own readouts
score 1.000 on the very same activations. Neither a small random init nor a
closed-form ridge start recovers the decode those constructions are *known* to
have: it is a narrow, large-weight solution and the probe's objective does not
lead to it. That is a failure to re-derive the readout, not an absence of
information, and any ratio built on it is meaningless. Those two rows are marked
`n/a` below; their coding scheme is known analytically instead -- the decoded sum
*is* the label, so they are magnitude codes by construction, and binarising
their activations destroys the label outright.

The columns that *are* valid for every construction are the ones that need no
retraining: accuracy, density, and weight scale.

    uv run python probe_coding.py --d 64
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.handcoded import HandCodedParams, hand_coded_weights
from handcode.model import ModelShape, accuracy, hidden_activations, random_init

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Fact counts at ~90% of each construction's own measured acc=1 capacity at
# d=64, from results/scaling.json.
LOADS = {
    "trained": 6566,
    "hand-coded": 194,
    "linsolve": 5414,
    "twosided": 11520,
}


def train_keeping_weights(
    shape: ModelShape, facts: dict, seed: int, n_epochs: int = 5000, lr: float = 1e-2
) -> tuple[torch.Tensor, torch.Tensor]:
    """The post's training recipe, but returning the weights.

    `model.train` clones its inputs and returns only the accuracy, so the
    trained weights are discarded -- analysing a trained model needs its own
    loop. Getting this wrong silently produces a table of random-init results.
    """
    up, down = random_init(shape, seed)
    up = up.detach().clone().requires_grad_(True)
    down = down.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([up, down], lr=lr)
    inputs, targets = facts["inputs"], facts["targets"]

    best, since = 0.0, 0
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = hidden_activations(up, inputs) @ down.T
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        best, since = (acc, 0) if acc > best else (best, since + 1)
        if best == 1.0 or since >= 100:
            break
    return up.detach(), down.detach()


def retrain_readout(
    hidden: torch.Tensor, targets: torch.Tensor, n_classes: int,
    n_epochs: int = 3000, lr: float = 1e-2,
) -> float:
    """Best accuracy a linear readout can reach on fixed features.

    Adam is started from the closed-form ridge fit rather than from a small
    random init, and the learning rate is scaled to the features. Without this
    the probe silently measures the *optimiser* instead of the features: a value
    code needs readout weights of order `1e4`, Adam will not travel that far
    from a `1/sqrt(d)` init, and the probe reports 0.07 for a construction whose
    own readout scores 1.000. That is an optimisation failure, not an
    information one, and it makes every ratio computed from it meaningless.
    """
    from handcode.readouts import ridge_down

    best_start, best_acc = None, -1.0
    for alpha in (1e-6, 1e-4, 1e-2, 1.0):
        candidate = ridge_down(hidden, targets, n_classes, alpha)
        acc = ((hidden @ candidate.T).argmax(-1) == targets).float().mean().item()
        if acc > best_acc:
            best_start, best_acc = candidate, acc

    down = best_start.clone().requires_grad_(True)
    # Adam's step size is absolute, so it has to be scaled to the weights it is
    # refining -- a 1e-2 step is a no-op on weights of size 1e4.
    scale = max(1.0, float(down.detach().abs().max()))
    optimizer = torch.optim.Adam([down], lr=lr * scale)

    best, since = best_acc, 0
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = hidden @ down.T
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        best, since = (acc, 0) if acc > best else (best, since + 1)
        if best == 1.0 or since >= 100:
            break
    return best


def quantise(hidden: torch.Tensor, levels: int | None) -> torch.Tensor:
    """Keep the pattern, coarsen the magnitudes to `levels` steps.

    `levels=1` is pure binarisation: every active neuron reports the same
    number, so all that survives is *which* neurons fired.
    """
    if levels is None:
        return hidden
    active = hidden > 0
    if levels == 1:
        return active.to(hidden.dtype)
    positive = hidden[active]
    if positive.numel() == 0:
        return hidden
    low, high = positive.min(), positive.max()
    step = (high - low) / levels
    coarse = torch.round((hidden - low) / step) * step + low
    return torch.where(active, coarse.clamp(min=low), torch.zeros_like(hidden))


def build(condition: str, shape: ModelShape, facts: dict, seed: int = 1000):
    if condition == "trained":
        return train_keeping_weights(shape, facts, seed)
    if condition == "hand-coded":
        best = None
        for S in (4, 8, 9, 12, 16):
            weights = hand_coded_weights(
                shape, facts, HandCodedParams(S=S, top_fraction=0.15), seed
            )
            score = accuracy(*weights, facts)
            if best is None or score > best[0]:
                best = (score, weights)
        return best[1]
    if condition == "linsolve":
        from handcode.fastsolve import assemble, cached_grouping, cached_mask, solve_profile
        from handcode.linsolve import DROP_SCHEDULES, MU_VALUES, T0_SCALES

        n_value = shape.d_mlp - 1
        grouping = cached_grouping(facts, shape)
        best = None
        for k in (4, 6, 8):
            mask = cached_mask(shape.input_vocab_size, n_value, k, seed)
            for mu in MU_VALUES:
                for keep, rounds, pertoken in DROP_SCHEDULES:
                    x = solve_profile(grouping, mask, n_value, mu, keep, rounds,
                                      pertoken, len(facts["targets"]))
                    for t0 in T0_SCALES:
                        weights = assemble(shape, mask, x, k,
                                           t0 * shape.output_vocab_size)
                        score = accuracy(*weights, facts)
                        if best is None or score > best[0]:
                            best = (score, weights)
        return best[1]
    if condition == "twosided":
        from handcode.twosided import (
            MU_VALUES, RHO_VALUES, TwoSidedParams, assemble, solve_two_sided,
        )

        best = None
        for rho in RHO_VALUES:
            for mu in MU_VALUES:
                params = TwoSidedParams(rho=rho, mu=mu)
                solution = solve_two_sided(shape, facts, params, seed)
                weights = assemble(shape, solution, params.delta)
                score = accuracy(*weights, facts)
                if best is None or score > best[0]:
                    best = (score, weights)
                if score == 1.0:
                    return weights
        return best[1]
    raise ValueError(f"unknown condition {condition!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--conditions", nargs="+", default=list(LOADS))
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 32])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "coding.json"))
    args = parser.parse_args()

    shape = ModelShape.from_d(args.d)
    records = []
    header = (f"{'construction':<13s}{'n':>7s}{'acc':>7s}{'density':>9s}"
              f"{'max|up|':>10s}{'max|down|':>11s}{'full':>7s}"
              + "".join(f"{'L=' + str(L):>7s}" for L in args.levels) + f"{'bin/full':>10s}")
    print(header)
    print("-" * len(header))

    for condition in args.conditions:
        n_facts = LOADS[condition]
        facts = generate_facts(
            n_facts, shape.input_vocab_size, shape.output_vocab_size, 42
        )
        up, down = build(condition, shape, facts)
        hidden = hidden_activations(up, facts["inputs"])

        row = {
            "condition": condition,
            "d": args.d,
            "n_facts": n_facts,
            "accuracy": accuracy(up, down, facts),
            "density": float((hidden > 0).float().mean()),
            "max_up": float(up.abs().max()),
            "max_down": float(down.abs().max()),
            "probe_full": retrain_readout(hidden, facts["targets"], shape.output_vocab_size),
        }
        for levels in args.levels:
            row[f"probe_{levels}"] = retrain_readout(
                quantise(hidden, levels), facts["targets"], shape.output_vocab_size
            )
        # A construction whose readout scores far above what the probe can
        # re-derive has defeated the probe, not revealed anything about its
        # features; see the module docstring.
        row["probe_valid"] = row["probe_full"] >= 0.9 * row["accuracy"]
        row["binary_over_full"] = (
            row["probe_1"] / row["probe_full"]
            if row["probe_valid"] and row["probe_full"]
            else float("nan")
        )
        records.append(row)
        cells = (
            "".join(f"{row['probe_' + str(L)]:>7.3f}" for L in args.levels)
            if row["probe_valid"]
            else "".join(f"{'n/a':>7s}" for _ in args.levels)
        )
        ratio = f"{row['binary_over_full']:>10.2f}" if row["probe_valid"] else f"{'n/a':>10s}"
        full = f"{row['probe_full']:>7.3f}" if row["probe_valid"] else f"{'n/a':>7s}"
        print(
            f"{condition:<13s}{n_facts:>7d}{row['accuracy']:>7.3f}{row['density']:>9.3f}"
            f"{row['max_up']:>10.4g}{row['max_down']:>11.4g}{full}{cells}{ratio}",
            flush=True,
        )

    if any(not r["probe_valid"] for r in records):
        print(
            "\nn/a: the retrained readout scored far below the construction's own,"
            "\n     so it failed to re-derive a decode that demonstrably exists."
            "\n     These are magnitude codes analytically -- the decoded sum is the"
            "\n     label -- so binarising destroys it. See the module docstring."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
