"""When Adam walks off the construction, which facts does it spend?

`probe_reachability.py` established that Adam started at the two-sided
construction trades ~17% of the facts for ~18% less loss. The margin story
predicts *which* facts it gives up: cross-entropy pays for margin on the many
by surrendering the facts it would cost the most margin to keep -- so the
casualties should be the thin-margin facts, and the survivors' margins should
fatten toward what a trained-from-scratch model holds.

Measured here, at the construction's own acc=1 capacity:

  * each fact's decision margin at epoch 0 (the construction's), and whether
    that fact is still correct after Adam;
  * the survivors' margin distribution before and after.

    uv run python probe_walkaway.py --d 32
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.model import ModelShape, accuracy, hidden_activations
from probe_reachability import build_construction

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def margins(up: torch.Tensor, down: torch.Tensor, facts: dict) -> torch.Tensor:
    with torch.no_grad():
        logits = hidden_activations(up, facts["inputs"]) @ down.T
        correct = logits.gather(1, facts["targets"].unsqueeze(1)).squeeze(1)
        rest = logits.scatter(1, facts["targets"].unsqueeze(1), -torch.inf)
        return correct - rest.max(1).values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "walkaway.json"))
    args = parser.parse_args()

    capacity = {16: 696, 32: 3168, 64: 12800}
    shape = ModelShape.from_d(args.d)
    n_facts = capacity[args.d]
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)

    up, down = build_construction(shape, facts)
    start_margin = margins(up, down, facts)
    print(f"construction acc={accuracy(up, down, facts):.4f} "
          f"median margin={float(start_margin.median()):.3f}")

    up = up.clone().requires_grad_(True)
    down = down.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([up, down], lr=1e-2)
    inputs, targets = facts["inputs"], facts["targets"]
    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits = hidden_activations(up, inputs) @ down.T
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
    up, down = up.detach(), down.detach()

    end_margin = margins(up, down, facts)
    kept = end_margin > 0
    acc = float(kept.float().mean())
    print(f"after {args.epochs} Adam epochs: acc={acc:.4f}")

    # Does the starting margin predict survival?
    deciles = torch.quantile(start_margin, torch.linspace(0, 1, 11))
    survival = []
    for i in range(10):
        in_bin = (start_margin >= deciles[i]) & (start_margin <= deciles[i + 1])
        survival.append(
            {
                "margin_lo": float(deciles[i]),
                "margin_hi": float(deciles[i + 1]),
                "n": int(in_bin.sum()),
                "survival": float(kept[in_bin].float().mean()),
            }
        )
        print(f"  start margin [{deciles[i]:6.3f}, {deciles[i+1]:6.3f}]  "
              f"survived {survival[-1]['survival']:.3f}")

    # Rank correlation between starting margin and survival.
    order = start_margin.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n_facts, dtype=torch.float32)
    r = torch.corrcoef(torch.stack([ranks, kept.float()]))[0, 1]
    print(f"rank correlation(start margin, survived) = {float(r):.3f}")

    survivors = end_margin[kept]
    print(f"survivors' margin after: median={float(survivors.median()):.3f} "
          f"(construction was {float(start_margin.median()):.3f})")

    payload = {
        "d": args.d,
        "n_facts": n_facts,
        "epochs": args.epochs,
        "acc_after": acc,
        "rank_correlation": float(r),
        "start_margin_median": float(start_margin.median()),
        "survivor_margin_median": float(survivors.median()),
        "survival_by_decile": survival,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
