"""Redraw Figure 5: max facts vs model dimension, ours against the published fits.

    uv run python plot.py
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt

from handcode.capacity import CONDITIONS, fit_scaling
from run_scaling import PUBLISHED, RESULTS_DIR

COLORS = {
    "trained": "#1f77b4",
    "hybrid": "#2ca02c",
    "hand-coded": "#d62728",
    "rand-emb": "#7f7f7f",
    "hc-tiebreak": "#ff7f0e",
    "hc-ridge": "#ffbb78",
    "coin-tiebreak": "#9467bd",
    "coin-ridge": "#c5b0d5",
    "linsolve": "#8c564b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "scaling.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "scaling.png"))
    args = parser.parse_args()

    with open(args.results) as f:
        payload = json.load(f)
    records = payload["records"]

    by_key: dict[tuple, dict[int, int]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["threshold"]), {})[r["d"]] = r["max_facts"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, threshold in zip(axes, (1.0, 0.9)):
        for condition in CONDITIONS:
            got = by_key.get((condition, threshold))
            if not got:
                continue
            ds = sorted(got)
            ys = [got[d] for d in ds]
            color = COLORS[condition]
            ax.plot(ds, ys, "o", color=color, label=f"{condition} (ours)", markersize=7)

            # our fit
            if len(ds) >= 2:
                a, b = fit_scaling(ds, ys)
                grid = [ds[0] * (ds[-1] / ds[0]) ** (i / 50) for i in range(51)]
                ax.plot(grid, [a * g**b / math.log(g) for g in grid], "-", color=color, lw=1.5)

            # published fit, where the authors reported one
            published = PUBLISHED.get((condition, threshold))
            if published:
                pa, pb = published
                grid = [ds[0] * (ds[-1] / ds[0]) ** (i / 50) for i in range(51)]
                ax.plot(
                    grid,
                    [pa * g**pb / math.log(g) for g in grid],
                    "--",
                    color=color,
                    lw=1.2,
                    alpha=0.65,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("model dimension $d$")
        ax.set_title("acc = 1" if threshold == 1.0 else r"acc $\geq$ 0.9")
        ax.grid(True, which="both", alpha=0.2)

    axes[0].set_ylabel("max facts memorised")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="k", ls="--", alpha=0.65))
    labels.append("published fit (dashed)")
    axes[0].legend(handles, labels, fontsize=9, loc="upper left")
    fig.suptitle(
        "Sequence memorisation capacity: reproduction (markers + solid) vs "
        "Linsefors & Bushnaq (dashed)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
