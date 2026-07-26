"""The gate-quality curve: when training builds a usable gate (FINDINGS.md 16).

Top panel: each checkpoint's frozen-pattern sigma90 ceiling (two-LP ascent,
embeddings and readout exactly optimal) against training epoch. The gate is
infeasible -- no positive margin exists -- through train accuracy 0.77, then
crosses to 4.5x the best constructed gate inside a ~50-epoch window and is at
92% of its final ceiling when training first reaches accuracy 1. Bottom
panel: the training context on one axis (both are fractions) -- accuracy and
net pattern drift from init.

    uv run python plot_gatecurve.py
"""

import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLORS = {
    "ceiling": "#2a78d6",
    "constructed": "#eb6834",
    "accuracy": "#1baf7a",
    "drift": "#eda100",
}
BEST_CONSTRUCTED = 3.18e-3  # spread-pressure drift, one step (FINDINGS.md 15)
INFEASIBLE_Y = 4e-6  # display shelf for gamma = 0 checkpoints


def main() -> None:
    with open(os.path.join(RESULTS_DIR, "gatequality.json")) as f:
        curve = json.load(f)["curve"]
    first_acc1 = curve["first_acc1"]
    rows = sorted(
        ((int(e), r) for e, r in curve["ascent"].items()), key=lambda t: t[0]
    )
    rows = [(max(e, 1), r) for e, r in rows]  # epoch 0 -> 1 for the log axis

    feas = [(e, r["metrics"]["sigma90_weight"]) for e, r in rows
            if r["metrics"]["accuracy"] == 1.0]
    infeas = [e for e, r in rows if r["metrics"]["accuracy"] < 1.0]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.8), sharex=True, constrained_layout=True,
        height_ratios=[2.2, 1],
    )

    for a in (ax, ax2):
        a.axvspan(150, first_acc1, color="#f2e8d5", alpha=0.55, zorder=0)
    ax.text(232, 1.1e-4, "the window where\nthe gate becomes good",
            color="#8a7340", fontsize=9, ha="center")

    xs, ys = zip(*feas)
    ax.plot(xs, ys, "o-", color=COLORS["ceiling"], lw=2, ms=5,
            label="gate ceiling (two-LP ascent on the frozen pattern)")
    ax.plot(infeas, [INFEASIBLE_Y] * len(infeas), "x", color=COLORS["ceiling"],
            ms=6, label="gate infeasible (no positive margin exists)")
    ax.annotate("infeasible through\ntrain acc 0.77", (12, INFEASIBLE_Y),
                xytext=(8, 3.2e-5), color=COLORS["ceiling"], fontsize=9)
    ax.annotate(f"4.06e-2 at first acc=1\n(4.40e-2 by epoch 5000)",
                (first_acc1, 4.06e-2), xytext=(600, 8e-3),
                color=COLORS["ceiling"], fontsize=9,
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))

    ax.axhline(BEST_CONSTRUCTED, color=COLORS["constructed"], lw=1.2, ls="--")
    ax.text(1.1, BEST_CONSTRUCTED * 1.35, "best constructed gate (3.18e-3)",
            color=COLORS["constructed"], fontsize=9)

    ax.axvline(first_acc1, color="#888888", lw=1, ls=":")
    ax.text(first_acc1 * 1.07, 6.5e-6, f"first acc=1\n(epoch {first_acc1})",
            color="#555555", fontsize=9)

    ax.set_yscale("log")
    ax.set_ylabel("gate ceiling  $\\sigma_{90}$  (relative)")
    ax.set_title("When gradient descent builds the gate, d=32, n=1584\n"
                 "(each point: the LP-optimal robustness of that epoch's "
                 "frozen sign pattern)")
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    ax.legend(loc="center right", fontsize=9, frameon=False)

    ep = [max(int(e), 1) for e, _ in sorted(
        ((int(e), r) for e, r in curve["ascent"].items()), key=lambda t: t[0])]
    acc = [r["train_acc"] for _, r in rows]
    drift = [r["from_init"] for _, r in rows]
    ax2.plot(ep, acc, "o-", color=COLORS["accuracy"], lw=2, ms=4)
    ax2.text(1.6, 0.16, "train accuracy", color=COLORS["accuracy"], fontsize=9)
    ax2.plot(ep, drift, "s-", color=COLORS["drift"], lw=2, ms=4)
    ax2.text(30, 0.44, "net pattern drift from init", color=COLORS["drift"],
             fontsize=9)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("training epoch")
    ax2.set_ylabel("fraction")
    ax2.grid(True, which="major", alpha=0.25, lw=0.5)

    out = os.path.join(RESULTS_DIR, "gatecurve.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
