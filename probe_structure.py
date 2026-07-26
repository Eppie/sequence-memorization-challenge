"""Weight-level structure of the trained solution, vs the constructions.

Three measurements on models at 100% accuracy, d=32:

  radius     per-fact first-order distance to error in weight space:
             margin_f / ||grad_W margin_f||. sigma90 is the aggregate view;
             this is the distribution. A max-margin solution equalizes the
             binding radii; an exact solve leaves them uniformly thin.
  readout    singular spectrum of the unembedding: effective rank
             (participation ratio), and the fraction of each neuron's class
             profile down[:, i] explained by a *graded* (linear-in-class)
             ladder -- the structure every value code here reads out with.
  margins    the margin distribution's shape (CV), same models.

Models: trained at its capacity (snapshot at first acc=1), trained past that
(the post's full budget), the twosided construction at its capacity, and
Adam-under-the-fixed-quadratic-decode at trained capacity.

    uv run python probe_structure.py
"""

import json
import os

import torch

from handcode.data import generate_facts
from handcode.model import ModelShape, accuracy, hidden_activations
from probe_fixed_readout import quad_down, train_up_only
from probe_reachability import build_construction, run_adam
from probe_robustness import build_trained, margin_stats, rms

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def per_fact_radii(up, down, facts, n_sample: int = 512, seed: int = 0):
    """margin_f / ||grad margin_f|| over a random sample of facts."""
    n = len(facts["targets"])
    gen = torch.Generator().manual_seed(seed)
    sample = torch.randperm(n, generator=gen)[:n_sample]
    radii, margins = [], []
    for f in sample.tolist():
        u = up.detach().clone().requires_grad_(True)
        d = down.detach().clone().requires_grad_(True)
        inputs = facts["inputs"][f : f + 1]
        target = int(facts["targets"][f])
        logits = (hidden_activations(u, inputs) @ d.T).squeeze(0)
        wrong = logits.clone()
        wrong[target] = -torch.inf
        margin = logits[target] - wrong.max()
        margin.backward()
        grad_norm = float((u.grad.pow(2).sum() + d.grad.pow(2).sum()).sqrt())
        margins.append(float(margin))
        radii.append(float(margin) / max(grad_norm, 1e-12))
    r = torch.tensor(radii)
    m = torch.tensor(margins)
    return {
        "radius_median": float(r.median()),
        "radius_p10": float(r.quantile(0.10)),
        "radius_min": float(r.min()),
        "radius_cv": float(r.std() / r.mean()),
        "margin_cv": float(m.std() / m.mean()),
        "n_sample": len(radii),
    }


def readout_structure(down: torch.Tensor):
    """Spectrum shape and graded-ladder content of the unembedding."""
    s = torch.linalg.svdvals(down.double())
    participation = float(s.sum() ** 2 / (s * s).sum())
    energy = (s * s) / (s * s).sum()
    rank90 = int((energy.cumsum(0) < 0.90).sum()) + 1

    # Fit down[c, i] ~ a_i + b_i * c per neuron; fraction of class-profile
    # variance explained by the graded ladder.
    L, D = down.shape
    c = torch.arange(L, dtype=torch.float64)
    x = torch.stack([torch.ones(L, dtype=torch.float64), c], dim=1)  # (L, 2)
    y = down.double()  # (L, D)
    beta = torch.linalg.lstsq(x, y).solution  # (2, D)
    resid = y - x @ beta
    var = y.var(0, unbiased=False)
    frac = 1.0 - resid.pow(2).mean(0) / var.clamp(min=1e-12)
    return {
        "singular_values": [round(float(v), 4) for v in s],
        "participation_ratio": participation,
        "rank90": rank90,
        "graded_fraction_mean": float(frac.mean()),
        "graded_fraction_median": float(frac.median()),
    }


def main() -> None:
    d = 32
    shape = ModelShape.from_d(d)
    results = {}

    facts_2080 = generate_facts(2080, shape.input_vocab_size, shape.output_vocab_size, 42)
    facts_3168 = generate_facts(3168, shape.input_vocab_size, shape.output_vocab_size, 42)

    models = {}

    weights, clean = build_trained(shape, facts_2080)
    models["trained@2080 (first acc=1)"] = (weights, facts_2080, clean)

    # The post's full budget: keep training after reaching 1.0 for the rest of
    # the 5000 epochs, as probe_reachability's trainer does.
    _, _, up_c, down_c = run_adam(*weights, facts_2080, n_epochs=5000, lr=1e-2)
    models["trained@2080 (post budget)"] = (
        (up_c, down_c), facts_2080, accuracy(up_c, down_c, facts_2080)
    )

    up, down = build_construction(shape, facts_3168)
    models["twosided@3168"] = ((up, down), facts_3168, accuracy(up, down, facts_3168))

    qdown = quad_down(shape)
    acc, qup = train_up_only(
        qdown, shape, facts_2080, seed=1000, lr=1e-2, n_epochs=50000,
        patience=2000, bias_row=True,
    )
    models["fixed-quad GD@2080"] = ((qup, qdown), facts_2080, acc)

    for name, ((w_up, w_down), facts, clean) in models.items():
        entry = {
            "clean_accuracy": clean,
            "rms_up": rms(w_up),
            "rms_down": rms(w_down),
            "margins": margin_stats(w_up, w_down, facts),
            "radii": per_fact_radii(w_up, w_down, facts),
            "readout": readout_structure(w_down),
        }
        results[name] = entry
        r, ro = entry["radii"], entry["readout"]
        print(
            f"{name:28s} acc={clean:.4f}\n"
            f"    radius median={r['radius_median']:.3e} p10={r['radius_p10']:.3e} "
            f"min={r['radius_min']:.3e} cv={r['radius_cv']:.2f} "
            f"(margin cv={r['margin_cv']:.2f})\n"
            f"    readout: participation={ro['participation_ratio']:.1f} "
            f"rank90={ro['rank90']} graded-ladder fraction "
            f"mean={ro['graded_fraction_mean']:.2f} "
            f"median={ro['graded_fraction_median']:.2f}",
            flush=True,
        )

    out = os.path.join(RESULTS_DIR, "structure.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
