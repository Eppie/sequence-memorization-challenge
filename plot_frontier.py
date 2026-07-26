"""One picture of the whole story: the capacity-robustness plane at d=32.

Every solution measured this session, placed by (facts stored at acc=1,
relative weight-noise tolerance sigma90). The trained models trace a frontier;
the authors' construction sits near it; every equality-constrained value code
sits orders of magnitude below; the max-margin LP on the trained geometry
lands on the trained point.

    uv run python plot_frontier.py
"""

import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COLORS = {
    "trained": "#2a78d6",
    "authors": "#1baf7a",
    "equality": "#eb6834",
    "hybrid": "#eda100",
}

TRAINED = [(130, 4.6e-1), (392, 2.08e-1), (1056, 1.02e-1), (1408, 6.3e-2),
           (1584, 4.3e-2), (2080, 2.0e-2)]
AUTHORS = [(130, 2.4e-1)]
EQUALITY = {
    "linsolve": [(1408, 1.0e-5)],
    "twosided": [(2080, 1.5e-5), (3168, 1.6e-5)],
    "digit m=2/3/5": [(1584, 1.1e-4), (1056, 1.4e-4), (392, 3.5e-4)],
    "digit, pedestal-optimized": [(1584, 1.3e-3), (1056, 1.6e-3)],
}
HYBRID = {
    "Adam under fixed value decode": [(2080, 3.3e-4)],
    "max-margin LP of trained geometry": [(2080, 1.57e-2)],
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 6.2), constrained_layout=True)

    xs, ys = zip(*TRAINED)
    ax.plot(xs, ys, "o-", color=COLORS["trained"], lw=2, ms=6,
            label="trained (gradient descent)")
    ax.annotate("trained frontier", (1408, 6.3e-2), xytext=(500, 2.5e-1),
                color=COLORS["trained"], fontsize=10)

    ax.plot(*zip(*AUTHORS), "s", color=COLORS["authors"], ms=9,
            label="authors' hand-coded")
    ax.annotate("their construction", AUTHORS[0], xytext=(160, 1.7e-1),
                color=COLORS["authors"], fontsize=10)

    first = True
    for name, pts in EQUALITY.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "^", color=COLORS["equality"], ms=8,
                label="equality-solve value codes" if first else None)
        first = False
    ax.annotate("exact solves\n(twosided, linsolve)", (2300, 4.5e-5),
                color=COLORS["equality"], fontsize=10, ha="left")
    ax.annotate("digit codes", (400, 4.5e-4), color=COLORS["equality"], fontsize=10)
    ax.annotate("pedestal-\noptimized", (1620, 1.7e-3), color=COLORS["equality"],
                fontsize=10)

    markers = {"Adam under fixed value decode": "D",
               "max-margin LP of trained geometry": "*"}
    for name, pts in HYBRID.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, markers[name], color=COLORS["hybrid"],
                ms=12 if markers[name] == "*" else 8, label=name)

    for n, lo, hi, text in ((2080, 1.5e-5, 2.0e-2, "~1300×"),
                            (1584, 1.3e-3, 4.3e-2, "~30×")):
        ax.annotate("", (n * 1.06, hi), (n * 1.06, lo),
                    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1))
        ax.text(n * 1.10, (lo * hi) ** 0.5, text, color="#555555", fontsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("facts stored at accuracy = 1")
    ax.set_ylabel("weight-noise tolerance  $\\sigma_{90}$  (relative)")
    ax.set_title("The capacity-robustness plane, d=32\n"
                 "(the benchmark scores only the x-axis)")
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    out = os.path.join(RESULTS_DIR, "frontier.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
