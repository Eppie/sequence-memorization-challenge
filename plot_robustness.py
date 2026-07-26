"""Draw the robustness curves: accuracy under relative weight noise.

One panel per load-matched pair -- each construction at its own acc=1 capacity
against a model trained to acc=1 at the same fact count -- rows are model sizes.

    uv run python plot_robustness.py
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLORS = {
    "trained": "#2a78d6",
    "twosided": "#eb6834",
    "hand-coded": "#1baf7a",
    "linsolve": "#eda100",
}
LABELS = {
    "trained": "trained",
    "twosided": "twosided",
    "hand-coded": "their hand-coded",
    "linsolve": "linsolve",
}
# Each panel: (construction, which n to use). The trained partner at the same
# n is drawn in every panel.
PAIRS = ("hand-coded", "linsolve", "twosided")


def by_condition(records: list[dict]) -> dict[tuple, dict]:
    return {(r["condition"], r["n_facts"]): r for r in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "robustness.png"))
    args = parser.parse_args()

    payloads = []
    for d in (32, 64):
        with open(os.path.join(RESULTS_DIR, f"robustness_d{d}.json")) as f:
            payloads.append(json.load(f))

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6.4), sharey=True, sharex=True, constrained_layout=True
    )
    for row, payload in enumerate(payloads):
        d = payload["d"]
        sigmas = payload["sigmas"]
        recs = by_condition(payload["records"])
        # The twosided panel compares at the *trained* model's capacity; the
        # other two at the construction's own capacity.
        trained_ns = sorted({n for (c, n) in recs if c == "trained"})
        for col, cond in enumerate(PAIRS):
            ax = axes[row][col]
            if cond == "twosided":
                n = max(trained_ns)
            else:
                n = next(nn for (c, nn) in recs if c == cond)
            for name, key in (("trained", ("trained", n)), (cond, (cond, n))):
                r = recs[key]
                ax.plot(
                    sigmas,
                    r["weight_noise"],
                    color=COLORS[name],
                    lw=2,
                    label=LABELS[name],
                )
                s90 = r["sigma90_weight"]
                if s90 is not None:
                    ax.plot([s90], [0.9], "o", ms=5, color=COLORS[name])
            ax.set_xscale("log")
            ax.set_ylim(0, 1.04)
            ax.set_title(f"d={d},  n={n} ({LABELS[cond]}'s pair)", fontsize=10)
            ax.grid(True, which="major", alpha=0.25, lw=0.5)
            ax.axhline(0.9, color="#999999", lw=0.8, ls=":")
            if row == 1:
                ax.set_xlabel("weight noise  $\\sigma$  (relative to each matrix's RMS)")
            if col == 0:
                ax.set_ylabel("accuracy")
            ax.legend(loc="lower left", fontsize=9, frameon=False)

    fig.suptitle(
        "Accuracy under weight noise: each construction vs a trained model at the same load\n"
        "(dots mark $\\sigma_{90}$, where mean accuracy over 20 draws falls below 0.9)",
        fontsize=11,
    )
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
