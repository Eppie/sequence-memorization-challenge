"""What stops gradient descent: the objective, or the optimizer?

The two-sided construction stores ~1.5x what Adam finds at d=32, and
`probe_reachability.py` shows Adam walks *away* from it -- the construction is
not a stationary point of cross-entropy. Two stories fit that fact:

  objective   cross-entropy prefers fat margins, the construction's margins
              are ~0.4 against a 0.5 ceiling, and the facts it gives up are
              the price of margin. The gap is what the loss *wants*.
  optimizer   the construction solves a 4d^2-unknown linear system exactly
              (Newton on the frozen-pattern equations); a first-order method
              cannot solve an ill-conditioned system to that precision in any
              reasonable budget. The gap is what Adam *can do*.

These could not be separated before because the loss is scale-invariant: any
margin target can be met by scaling the unembedding, so margin caps and hinge
losses do not bind (002's ruled-out list). Fixing the unembedding to the
construction's own quadratic decode changes that. The readout is homogeneous
degree-1 in the activations, so scaling still inflates *absolute* margins,
but the *relative* geometry is pinned: with the decode fixed, per-fact
relative margins are capped by the label spacing, and cross-entropy can no
longer trade facts for separation the readout is unable to express.

Under the fixed decode, Adam is training the embeddings to satisfy exactly
the construction's equations (sum of active units = label + 1). So:

  * if Adam-with-fixed-decode reaches the construction's capacity, the gap
    was the objective all along -- gradient descent solves value codes fine
    once the readout stops letting it buy margin;
  * if it stalls at the trained model's capacity, the gap is the optimizer --
    first-order descent cannot do what the Newton solve does, even on the
    same equations;
  * in between, the two effects decompose.

A frozen *random* readout is run as a control, separating "the quadratic
decode is special" from "training only the embedding is easier". Gradient
descent appears throughout -- this is analysis, not a challenge entry.

    uv run python probe_fixed_readout.py --d 32
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.model import ModelShape, hidden_activations, random_init

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# d=32 landmarks: trained acc=1 capacity 2080 (post recipe; ~2560 converged),
# twosided acc=1 capacity 3168, twosided acc>=0.9 capacity 3904.
N_VALUES = (1408, 2080, 2560, 3168, 3904)
LRS = (1e-2, 3e-3, 1e-3)
SEEDS = (1000, 1001)


def quad_down(shape: ModelShape, delta: float = 1.0) -> torch.Tensor:
    """The construction's readout with shift = 0: the embedding is free to
    learn any offset itself, so none is baked in."""
    beta = float(shape.d_mlp)
    n_value = shape.d_mlp - 1
    labels = torch.arange(shape.output_vocab_size, dtype=torch.float32) + 1
    down = torch.zeros(shape.output_vocab_size, shape.d_mlp)
    down[:, :n_value] = (labels / delta).unsqueeze(1)
    down[:, n_value] = -(labels**2 / 2) / beta
    return down


def train_up_only(
    down: torch.Tensor,
    shape: ModelShape,
    facts: dict,
    seed: int,
    lr: float,
    n_epochs: int,
    patience: int,
    bias_row: bool,
) -> tuple[float, torch.Tensor]:
    """Adam on the embedding only; the readout never moves.

    With `bias_row`, the last neuron's row starts at beta/2 so the decode's
    threshold neuron is born alive; everything after that is up to Adam.
    """
    inputs, targets = facts["inputs"], facts["targets"]
    up, _ = random_init(shape, seed)
    if bias_row:
        up[shape.d_mlp - 1, :] = float(shape.d_mlp) / 2.0
    up = up.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([up], lr=lr)

    best_acc, best_up, since = 0.0, up.detach().clone(), 0
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = hidden_activations(up, inputs) @ down.T
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
        with torch.no_grad():
            acc = float((logits.argmax(-1) == targets).float().mean())
        if acc > best_acc:
            best_acc, best_up, since = acc, up.detach().clone(), 0
        else:
            since += 1
        if best_acc == 1.0 or since >= patience:
            break
    return best_acc, best_up


def best_over_recipe(
    down: torch.Tensor,
    shape: ModelShape,
    facts: dict,
    n_epochs: int,
    patience: int,
    bias_row: bool,
) -> tuple[float, dict]:
    best, best_meta = 0.0, {}
    for lr in LRS:
        for seed in SEEDS:
            acc, _ = train_up_only(
                down, shape, facts, seed, lr, n_epochs, patience, bias_row
            )
            if acc > best:
                best, best_meta = acc, {"lr": lr, "seed": seed}
            if best == 1.0:
                return best, best_meta
    return best, best_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--ns", type=int, nargs="*", default=None,
                        help="fact counts to test (default: the d=32 landmarks)")
    parser.add_argument("--long-epochs", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=2000,
                        help="patience for the long-budget pass")
    parser.add_argument("--only-long", action="store_true",
                        help="skip the 5k-recipe and random-readout passes")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "fixed_readout.json"))
    args = parser.parse_args()

    shape = ModelShape.from_d(args.d)
    records = []

    for n_facts in args.ns or N_VALUES:
        facts = generate_facts(
            n_facts, shape.input_vocab_size, shape.output_vocab_size, 42
        )
        row = {"n_facts": n_facts}
        down = quad_down(shape)

        if not args.only_long:
            acc, meta = best_over_recipe(
                down, shape, facts, n_epochs=5000, patience=100, bias_row=True
            )
            row["quad_5k"] = acc
            row["quad_5k_meta"] = meta

        if args.only_long or row["quad_5k"] < 1.0:
            long_acc, long_meta = best_over_recipe(
                down, shape, facts,
                n_epochs=args.long_epochs, patience=args.patience, bias_row=True,
            )
            row["quad_long"] = long_acc
            row["quad_long_meta"] = long_meta

        if not args.only_long:
            _, rand_down = random_init(shape, 77)
            rand_acc, rand_meta = best_over_recipe(
                rand_down, shape, facts, n_epochs=5000, patience=100, bias_row=False
            )
            row["rand_5k"] = rand_acc
            row["rand_5k_meta"] = rand_meta

        records.append(row)
        print(
            f"n={n_facts:<5d}"
            + (f" quad_5k={row['quad_5k']:.4f}" if "quad_5k" in row else "")
            + (f" quad_long={row['quad_long']:.4f}" if "quad_long" in row else "")
            + (f" rand_5k={row['rand_5k']:.4f}" if "rand_5k" in row else ""),
            flush=True,
        )

    payload = {"d": args.d, "records": records}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
