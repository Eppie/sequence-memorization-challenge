"""Why doesn't gradient descent find the two-sided value code?

The construction stores 1.4-1.9x what a trained model of the same size does, so
the weights exist inside the architecture and gradient descent does not reach
them. This asks which of the two possible reasons it is:

  * the solution is not a place gradient descent would *stay* -- it is a saddle,
    or a sharp minimum that cross-entropy pushes off; or
  * the solution is perfectly stable, and gradient descent simply never gets
    near it from a small random initialisation.

Three measurements, at a fact count above the trained model's own capacity:

  hold      start Adam *at* the construction and keep training. If accuracy
            holds at 1.0, the solution is a fine place to be and the problem is
            purely one of reach.
  reach     train from the standard random init, with the post's recipe and
            with a much longer budget, and see how close it gets.
  distance  compare the weight scale the construction needs against the scale
            Adam actually travels to.

    uv run python probe_reachability.py --d 64
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from handcode.data import generate_facts
from handcode.model import ModelShape, accuracy, hidden_activations, random_init
from handcode.twosided import MU_VALUES, RHO_VALUES, TwoSidedParams, assemble, solve_two_sided

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_adam(
    up: torch.Tensor,
    down: torch.Tensor,
    facts: dict,
    n_epochs: int,
    lr: float,
    every: int = 0,
) -> tuple[float, list, torch.Tensor, torch.Tensor]:
    """The post's recipe -- Adam, full batch, cross-entropy -- keeping weights."""
    up = up.detach().clone().requires_grad_(True)
    down = down.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([up, down], lr=lr)
    inputs, targets = facts["inputs"], facts["targets"]

    best, trace = 0.0, []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = hidden_activations(up, inputs) @ down.T
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        best = max(best, acc)
        if every and (epoch % every == 0 or epoch == n_epochs - 1):
            trace.append((epoch, round(acc, 4), round(float(loss), 4)))
    return best, trace, up.detach(), down.detach()


def build_construction(shape: ModelShape, facts: dict, seed: int = 1000):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-facts", type=int, default=None,
                        help="default: the construction's measured acc=1 capacity")
    parser.add_argument("--long-epochs", type=int, default=50000)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "reachability.json"))
    args = parser.parse_args()

    shape = ModelShape.from_d(args.d)
    capacity = {16: 696, 32: 3168, 64: 12800}
    n_facts = args.n_facts or capacity[args.d]
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    print(f"d={args.d}, n_facts={n_facts}\n")

    up, down = build_construction(shape, facts)
    built = accuracy(up, down, facts)
    scale = (float(up.abs().max()), float(down.abs().max()))
    print(f"construction                     acc={built:.4f}  "
          f"max|up|={scale[0]:.1f} max|down|={scale[1]:.1f}")

    # 1. HOLD -- is the construction a place Adam would stay?
    held, hold_trace, held_up, held_down = run_adam(
        up, down, facts, n_epochs=2000, lr=1e-2, every=200
    )
    final = accuracy(held_up, held_down, facts)
    print(f"  + 2000 Adam epochs from there   acc={final:.4f} (best {held:.4f})")
    print(f"    trace {hold_trace}")

    # 2. REACH -- the post's recipe, and a much longer one.
    rand_up, rand_down = random_init(shape, 1000)
    short, _, s_up, s_down = run_adam(rand_up, rand_down, facts, n_epochs=5000, lr=1e-2)
    print(f"\ntrained, post's recipe (5k ep)   acc={short:.4f}  "
          f"max|up|={float(s_up.abs().max()):.1f} max|down|={float(s_down.abs().max()):.1f}")

    long, long_trace, l_up, l_down = run_adam(
        rand_up, rand_down, facts, n_epochs=args.long_epochs, lr=1e-2,
        every=max(1, args.long_epochs // 10),
    )
    print(f"trained, {args.long_epochs} epochs        acc={long:.4f}  "
          f"max|up|={float(l_up.abs().max()):.1f} max|down|={float(l_down.abs().max()):.1f}")
    print(f"    trace {long_trace}")

    payload = {
        "d": args.d, "n_facts": n_facts,
        "construction_acc": built, "construction_scale": scale,
        "hold_best": held, "hold_final": final, "hold_trace": hold_trace,
        "trained_5k": short, "trained_long": long,
        "trained_long_scale": [float(l_up.abs().max()), float(l_down.abs().max())],
        "long_trace": long_trace,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
