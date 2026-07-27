"""Order-structure statistics of gates: the theory.md problem 6'' probe.

Theorem 2 (docs/theory.md) parameterizes every additively realizable gate
column by a token ordering plus per-row thresholds: with x_a = u[a,j] and
y_b = -v[b,j], cell (a,b) is active iff x_a > y_b, so the column IS the
interleaving of the x's and y's on the line. FINDINGS.md 14 refuted
magnitude/correlation statistics as gate-quality predictors, but no
*order-structure* statistic -- a function of the orderings and thresholds
alone, invariant to any monotone transform of the column values -- has ever
been tested. A statistic that separates good gates from bad ones across
provenances would be the first declarative handle on what gradient descent
builds, and the recipe for a construction that chooses its orderings
(problem 6''); a refutation extends 14's negative result to the full
parameterization and strengthens the incompressibility position (6''').

The zoo now contains the sharpest contrast the program has produced: the
ridge-built soft seed (storage-infeasible, ceiling 0) and the fw-flow product
grown from it (ceiling 2.3-2.4e-2), 7.6% of bits apart, both of fully
understood ancestry -- plus the GD, construction, and null gates of 14.

Statistics, all computed per column in rank space and pooled:

  rank margin rho    for fact (a,b): (# tokens b' with y_b' < x_a) minus
                     (rank of y_b) -- signed token-step distance of the
                     observed cell to the staircase boundary. rho >= 1 iff
                     active. |2 rho - 1| / 2 is the distance to the
                     boundary in steps.

  rank softness      fraction of observed cells within one token step of
                     the boundary; quantiles of the step distance. The
                     order-space analog of the near-tie reservoir.

  fact attention     step distance of observed (fact) cells vs all V^2
                     cells: does the boundary run close to the facts?

  ordering diversity mean |Kendall tau| over column pairs, for the y
                     orderings and for the row-threshold vectors r_x.

  interleaving       per-column runs statistic of the merged x/y sequence
                     (z-scored against the random interleaving null).

  threshold profile  dispersion of row thresholds r_x(a)/V (Ferrers shape).

  per-fact degree    active-column count per fact: min, p10, CV.

  label structure    same-label vs cross-label pattern correlation, and the
                     label share of rank-margin variance.

    uv run python probe_ordering.py                  # full zoo
    uv run python probe_ordering.py --gates trained fw_product seed
"""

import argparse
import json
import os

import numpy as np
import torch

from handcode.data import generate_facts
from handcode.model import ModelShape, hidden_activations
from probe_gatequality import GATES_DIR, RESULTS_DIR, get_gate

D = 32
N_FACTS = 1584
OUT_PATH = os.path.join(RESULTS_DIR, "ordering.json")

SOFTSEED_TAG = "mu0.01-lam0.1-r600-c0.1-w1.28"
FW_STATE = os.path.join(
    GATES_DIR, f"r200_stride_soft-{SOFTSEED_TAG}_fw_state.npz")


# --------------------------------------------------------------------------
# gate loading: everything reduces to (u, v) in (V, d) coordinates
# --------------------------------------------------------------------------

def split_up(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = up.shape[0]
    n_vocab = up.shape[1] // 2
    return (up[:, :n_vocab].T.astype(np.float64),
            up[:, n_vocab:].T.astype(np.float64))


def load_uv(name: str, shape, facts):
    """(u, v) for a zoo member. Gates with a bias neuron keep it; constant
    columns are dropped by the caller before any ordering statistic."""
    if name == "seed":
        z = np.load(os.path.join(GATES_DIR, f"softseed_{SOFTSEED_TAG}.npz"))
        return z["u"].astype(np.float64), z["v"].astype(np.float64)
    if name == "fw_product":
        z = np.load(FW_STATE)
        return split_up(z["up"])
    if name == "ep180":
        z = np.load(os.path.join(GATES_DIR, "edge_state.npz"))
        return split_up(z["up_180"])
    pattern, up, down = get_gate(name, shape, facts)
    return split_up(up.numpy() if isinstance(up, torch.Tensor) else up)


ZOO = ("trained", "trained_early", "fw_product", "digit_m2", "twosided",
       "seed", "ep180", "random_additive", "init")

# Known pattern ceilings at n=1584 (results/gatequality.json,
# results/flippolicy.json); "inf" marks storage-infeasible gates.
CEILINGS = {
    "trained": 4.40e-2, "trained_early": 4.07e-2, "fw_product": 2.31e-2,
    "digit_m2": 2.85e-3, "twosided": 1.41e-5, "seed": 0.0, "ep180": 0.0,
    "random_additive": 0.0, "init": 0.0,
}


# --------------------------------------------------------------------------
# rank-space machinery
# --------------------------------------------------------------------------

def column_ranks(u_col, v_col, first, second):
    """Per-column rank structure.

    Returns (rho, r_x, order_y): rho[f] is the signed rank margin of fact
    f's cell, r_x[a] the row threshold (# of y's below x_a, i.e. row a's
    total-grid active count), order_y the ascending argsort of y."""
    x = u_col
    y = -v_col
    order_y = np.argsort(y, kind="stable")
    y_sorted = y[order_y]
    rank_y = np.empty_like(order_y)
    rank_y[order_y] = np.arange(len(y))
    r_x = np.searchsorted(y_sorted, x, side="left")
    rho = r_x[first] - rank_y[second]
    return rho, r_x, rank_y


def runs_z(x, y):
    """z-score of the number of runs in the merged x/y ordering against the
    two-type Wald-Wolfowitz null (random interleaving)."""
    V = len(x)
    labels = np.concatenate([np.zeros(V, bool), np.ones(V, bool)])
    merged = labels[np.argsort(np.concatenate([x, y]), kind="stable")]
    runs = 1 + int((merged[1:] != merged[:-1]).sum())
    n = 2 * V
    mean = 2 * V * V / n + 1
    var = 2 * V * V * (2 * V * V - n) / (n * n * (n - 1))
    return (runs - mean) / np.sqrt(var)


def kendall_mean(orderings):
    """Mean |Kendall tau| over all column pairs of rank vectors."""
    from scipy.stats import kendalltau

    d = len(orderings)
    taus = []
    for i in range(d):
        for j in range(i + 1, d):
            t, _ = kendalltau(orderings[i], orderings[j])
            taus.append(abs(t))
    return float(np.mean(taus))


def gate_ordering_stats(u, v, inputs, targets, n_labels):
    """All order-structure statistics for one gate."""
    first, second = inputs[:, 0], inputs[:, 1]
    n, V = len(first), u.shape[0]
    pre = u[first] + v[second]
    P = pre > 0
    col_mean = P.mean(0)
    keep = (col_mean > 0.01) & (col_mean < 0.99)  # drop constant columns
    cols = np.flatnonzero(keep)

    rhos, r_xs, rank_ys, runs = [], [], [], []
    for j in cols:
        rho, r_x, rank_y = column_ranks(u[:, j], v[:, j], first, second)
        rhos.append(rho)
        r_xs.append(r_x)
        rank_ys.append(rank_y)
        runs.append(runs_z(u[:, j], -v[:, j]))
    rho = np.stack(rhos, axis=1)          # (n, d')
    r_x = np.stack(r_xs, axis=0)          # (d', V)

    # step distance to the boundary: active cells have rho >= 1, inactive
    # rho <= 0; distance in steps is rho - 1 (active) or -rho (inactive),
    # i.e. |rho - 0.5| - 0.5.
    dist = np.abs(rho - 0.5) - 0.5
    # the same quantity for every cell of the total grid, for the
    # fact-attention contrast (vectorized: rho_all[a, b] = r_x[a] - rank[b])
    dist_all_q = []
    soft_all = []
    for k, j in enumerate(cols):
        rho_all = r_xs[k][:, None] - rank_ys[k][None, :]
        d_all = np.abs(rho_all - 0.5) - 0.5
        dist_all_q.append(np.median(d_all))
        soft_all.append((d_all <= 1).mean())
    per_fact_degree = P[:, keep].sum(1)

    # label structure: same-label vs cross-label pattern correlation
    Pc = P[:, keep].astype(np.float64)
    Pc -= Pc.mean(1, keepdims=True)
    norms = np.linalg.norm(Pc, axis=1)
    ok = norms > 0
    rng = np.random.default_rng(0)
    f = rng.integers(0, n, 40000)
    g = rng.integers(0, n, 40000)
    m = (f != g) & ok[f] & ok[g]
    corr = np.einsum("ij,ij->i", Pc[f[m]], Pc[g[m]]) / (norms[f[m]] * norms[g[m]])
    same = targets[f[m]] == targets[g[m]]
    # label share of rank-margin variance (one-way ANOVA R^2, pooled cols)
    z = (rho - rho.mean(0)) / (rho.std(0) + 1e-12)
    label_means = np.stack([z[targets == c].mean(0) if (targets == c).any()
                            else np.zeros(z.shape[1])
                            for c in range(n_labels)])
    r2 = float((label_means[targets] ** 2).sum() / (z ** 2).sum())

    return {
        "n_const_cols": int((~keep).sum()),
        "density": float(P.mean()),
        "rank_soft_frac_1step": float((dist <= 1).mean()),
        "rank_dist_q05": float(np.quantile(dist, 0.05)),
        "rank_dist_median": float(np.median(dist)),
        "fact_attention": float(np.mean(dist_all_q) - np.median(dist)),
        "soft_all_cells": float(np.mean(soft_all)),
        "order_diversity_y": kendall_mean(rank_ys),
        "order_diversity_thresh": kendall_mean([r for r in r_x]),
        "runs_z_mean": float(np.mean(runs)),
        "thresh_disp": float(np.std(r_x / V)),
        "fact_degree_min": int(per_fact_degree.min()),
        "fact_degree_p10": float(np.percentile(per_fact_degree, 10)),
        "fact_degree_cv": float(per_fact_degree.std()
                                / max(per_fact_degree.mean(), 1e-9)),
        "same_label_corr": float(corr[same].mean()),
        "cross_label_corr": float(corr[~same].mean()),
        "label_corr_excess": float(corr[same].mean() - corr[~same].mean()),
        "label_rankvar_r2": r2,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", nargs="*", default=None)
    args = parser.parse_args()

    shape = ModelShape.from_d(D)
    facts = generate_facts(N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, 42)
    inputs = facts["inputs"].numpy()
    targets = facts["targets"].numpy()

    rows = {}
    for name in (args.gates or ZOO):
        u, v = load_uv(name, shape, facts)
        rows[name] = gate_ordering_stats(u, v, inputs, targets,
                                         shape.output_vocab_size)
        rows[name]["ceiling"] = CEILINGS.get(name)
        r = rows[name]
        print(f"{name:16s} ceil={r['ceiling']}  "
              f"soft1={r['rank_soft_frac_1step']:.3f} "
              f"attn={r['fact_attention']:+.2f} "
              f"divY={r['order_diversity_y']:.3f} "
              f"runsZ={r['runs_z_mean']:+.1f} "
              f"thrD={r['thresh_disp']:.3f} "
              f"degMin={r['fact_degree_min']} "
              f"lblXs={r['label_corr_excess']:+.4f} "
              f"lblR2={r['label_rankvar_r2']:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            data = json.load(f)
    data.setdefault("zoo", {}).update(rows)
    data["d"] = D
    data["n_facts"] = N_FACTS
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
