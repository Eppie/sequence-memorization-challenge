"""Are the trained model's ReLU patterns stabilized dynamically?

The account in docs/what-gd-builds.md claims trained models need no
stabilizer pedestal because gradient descent stabilizes the sign pattern
dynamically -- co-adapting the gate with everything else until it stops
moving. Until now that clause was inferred by elimination. Two direct
measurements:

  churn      during training (d=32, n=2080, the post's recipe), the fraction
             of (fact, neuron) pre-activation signs that flip per epoch.
             Dynamic stabilization predicts high churn early and ~zero at
             convergence, with the pattern settling *before* accuracy tops
             out -- the gate is found first, then margins are grown on it.

  co-sizing  under weight noise, the trained model's accuracy and its
             pattern should degrade *together* (the pattern is load-bearing
             and its stability margin is matched to the decision margin),
             while the exact-solve value codes should lose accuracy orders
             of magnitude before their pattern moves at all -- their decode
             precision, not their gate, is the fragile part. Measured: the
             noise level where 1% of pattern bits flip, against sigma90.

    uv run python probe_patterns.py
"""

import json
import os

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.digitcode import DigitCodeParams, assemble as digit_assemble, solve_digit_code
from handcode.model import ModelShape, accuracy, hidden_activations, random_init
from probe_digitcode import SIGMAS
from probe_reachability import build_construction
from probe_robustness import build_trained, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def churn_curve(shape, facts, n_epochs=5000, lr=1e-2, seed=1000):
    inputs, targets = facts["inputs"], facts["targets"]
    up, down = random_init(shape, seed)
    up = up.requires_grad_(True)
    down = down.requires_grad_(True)
    optimizer = torch.optim.Adam([up, down], lr=lr)
    prev = None
    curve = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pre = hidden_activations(up, inputs)
        logits = pre @ down.T
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
        with torch.no_grad():
            pattern = hidden_activations(up, inputs) > 0
            acc = float((logits.argmax(-1) == targets).float().mean())
        flip = float((pattern != prev).float().mean()) if prev is not None else 1.0
        prev = pattern
        if epoch in (0, 1, 3, 10, 30, 100, 300, 1000, 3000) or epoch == n_epochs - 1 \
                or acc == 1.0:
            curve.append({"epoch": epoch, "flip_frac": flip, "accuracy": acc})
        if acc == 1.0:
            break
    return curve


def pattern_vs_accuracy(up, down, facts, n_trials=10, seed=0):
    """Per noise level: mean accuracy and mean fraction of pattern bits flipped."""
    gen = torch.Generator().manual_seed(seed)
    s_up, s_down = rms(up), rms(down)
    with torch.no_grad():
        base = hidden_activations(up, facts["inputs"]) > 0
    rows = []
    for sigma in SIGMAS:
        accs, flips = [], []
        for _ in range(n_trials):
            nu = up + torch.randn(up.shape, generator=gen) * (sigma * s_up)
            nd = down + torch.randn(down.shape, generator=gen) * (sigma * s_down)
            accs.append(accuracy(nu, nd, facts))
            with torch.no_grad():
                flips.append(
                    float(((hidden_activations(nu, facts["inputs"]) > 0) != base)
                          .float().mean())
                )
        rows.append({"sigma": sigma, "accuracy": sum(accs) / len(accs),
                     "flip_frac": sum(flips) / len(flips)})
    return rows


def threshold(rows, key, level, direction):
    """First sigma where `key` crosses `level` (down for accuracy, up for flips)."""
    for r in rows:
        if (direction == "down" and r[key] < level) or \
                (direction == "up" and r[key] > level):
            return r["sigma"]
    return None


def main() -> None:
    shape = ModelShape.from_d(32)
    out = {}

    facts = generate_facts(2080, shape.input_vocab_size, shape.output_vocab_size, 42)
    out["churn"] = churn_curve(shape, facts)
    for row in out["churn"]:
        print(f"epoch {row['epoch']:>5d}: flip_frac={row['flip_frac']:.4f} "
              f"acc={row['accuracy']:.4f}", flush=True)

    models = {}
    models["trained@2080"] = (*build_trained(shape, facts)[0], facts)
    facts3 = generate_facts(3168, shape.input_vocab_size, shape.output_vocab_size, 42)
    models["twosided@3168"] = (*build_construction(shape, facts3), facts3)
    facts15 = generate_facts(1584, shape.input_vocab_size, shape.output_vocab_size, 42)
    sol = solve_digit_code(
        shape, facts15,
        DigitCodeParams(m=2, rounds=1500, sweeps=8, t0_scale=1.0), seed=1000,
    )
    models["digit m=2 t0=d @1584"] = (*digit_assemble(shape, sol), facts15)

    out["cosizing"] = {}
    for name, (up, down, ff) in models.items():
        rows = pattern_vs_accuracy(up, down, ff)
        s_acc = threshold(rows, "accuracy", 0.9, "down")
        s_flip = threshold(rows, "flip_frac", 0.01, "up")
        ratio = (s_flip / s_acc) if (s_acc and s_flip) else None
        out["cosizing"][name] = {"rows": rows, "sigma_acc90": s_acc,
                                 "sigma_flip1pct": s_flip, "ratio": ratio}
        print(f"{name:22s} sigma(acc<0.9)={s_acc:.2e} "
              f"sigma(1% flips)={s_flip:.2e} "
              f"pattern/decision = {ratio:.1f}x" if ratio else f"{name}: n/a",
              flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "patterns.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/patterns.json")


if __name__ == "__main__":
    main()
