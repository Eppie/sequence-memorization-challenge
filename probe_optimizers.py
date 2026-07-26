"""Would a different optimizer pack in more facts?

The capacity ladder at d=32 so far: the post's Adam recipe reaches 2080;
converged Adam ~2560; the frozen-pattern Newton solve (the twosided
construction) 3168. The claim in FINDINGS.md 7 is that the 2560 -> 3168 gap
is first-order optimization, not the objective. If that is right, a
curvature-aware full-batch method (L-BFGS) should land between converged
Adam and the Newton solve -- and whatever extra capacity it finds should be
paid for in robustness, because the extra facts only exist at thin margins
(FINDINGS.md 5: past ~2560 the solution must leave the noise-tolerance class
that any stochastic-step optimizer can hold).

Optimizers, all full-batch on cross-entropy from the standard init:

  adam-post   the post's recipe (lr 1e-2, 5k epochs, patience 100)
  adam-long   the same, 50k epochs, patience 3000 (002's converged baseline)
  sgd-mom     SGD + momentum 0.9, lr swept over (1.0, 0.1, 0.01)
  lbfgs       torch L-BFGS, strong-Wolfe line search, history 50

For any run that reaches 100% at n >= 2560, sigma90 and margins are recorded.

    uv run python probe_optimizers.py --d 32
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.model import ModelShape, hidden_activations, random_init
from probe_digitcode import noise_curve, sigma90
from probe_robustness import margin_stats, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_VALUES = (2560, 2880, 3168)
SEEDS = (1000, 1001)


def train_generic(shape, facts, seed, make_opt, n_steps, patience):
    inputs, targets = facts["inputs"], facts["targets"]
    up, down = random_init(shape, seed)
    up = up.requires_grad_(True)
    down = down.requires_grad_(True)
    optimizer = make_opt([up, down])
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    best_acc, best, since = 0.0, None, 0
    for _ in range(n_steps):
        def closure():
            optimizer.zero_grad()
            logits = hidden_activations(up, inputs) @ down.T
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            return loss

        if is_lbfgs:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

        with torch.no_grad():
            logits = hidden_activations(up, inputs) @ down.T
            acc = float((logits.argmax(-1) == targets).float().mean())
        if not torch.isfinite(logits).all():
            break
        if acc > best_acc:
            best_acc, since = acc, 0
            best = (up.detach().clone(), down.detach().clone())
        else:
            since += 1
        if best_acc == 1.0 or since >= patience:
            break
    return best_acc, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "optimizers.json"))
    args = parser.parse_args()

    shape = ModelShape.from_d(args.d)
    conditions = {
        "adam-post": dict(
            opts=[lambda p: torch.optim.Adam(p, lr=1e-2)],
            n_steps=5000, patience=100,
        ),
        "adam-long": dict(
            opts=[lambda p: torch.optim.Adam(p, lr=1e-2)],
            n_steps=50000, patience=3000,
        ),
        "sgd-mom": dict(
            opts=[
                (lambda lr: lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9))(lr)
                for lr in (1.0, 0.1, 0.01)
            ],
            n_steps=50000, patience=3000,
        ),
        "lbfgs": dict(
            opts=[lambda p: torch.optim.LBFGS(
                p, lr=1.0, history_size=50, max_iter=20,
                line_search_fn="strong_wolfe",
            )],
            n_steps=400, patience=60,  # each step is up to 20 inner iterations
        ),
    }

    records = []
    for n in N_VALUES:
        facts = generate_facts(n, shape.input_vocab_size, shape.output_vocab_size, 42)
        for name, cfg in conditions.items():
            best_acc, best, t = 0.0, None, time.time()
            for make_opt in cfg["opts"]:
                for seed in SEEDS:
                    acc, weights = train_generic(
                        shape, facts, seed, make_opt, cfg["n_steps"], cfg["patience"]
                    )
                    if acc > best_acc:
                        best_acc, best = acc, weights
                    if best_acc == 1.0:
                        break
                if best_acc == 1.0:
                    break
            row = {"n_facts": n, "optimizer": name, "best_acc": best_acc,
                   "seconds": round(time.time() - t, 1)}
            if best_acc == 1.0 and best is not None:
                up, down = best
                row["sigma90_weight"] = sigma90(noise_curve(up, down, facts))
                row["margins"] = margin_stats(up, down, facts)
                row["rms_up"] = rms(up)
            records.append(row)
            s90 = row.get("sigma90_weight")
            print(
                f"n={n} {name:10s} acc={best_acc:.4f} ({row['seconds']}s)"
                + (f" sigma90={s90:.2e} margin_med="
                   f"{row['margins']['median']:.3g}" if s90 else ""),
                flush=True,
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"d": args.d, "records": records}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
