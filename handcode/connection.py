"""Assigning neurons to labels (post: "Assigning neurons to labels").

A connection matrix is a (d_mlp, n_labels) binary matrix; conn[i, l] = 1 means
neuron i is assigned to label l. We want

  * exactly S ones per column   -- every label gets S neurons,
  * balanced row sums (~S*T/D)  -- every neuron serves about equally many labels,
  * pairwise column overlap <=1 -- no two labels share more than one neuron.

The last is a combinatorial-design constraint, only achievable when
T <= D(D-1) / (S(S-1)); beyond that we just minimize the number of violations.

This is a port of `make_connection_matrix` from the authors' hc2.py: greedy
column-by-column init, simulated-annealing refinement, then a greedy balance
pass. Results are cached on disk since the matrix depends only on (D, T, S, seed).
"""

from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "conn_cache")
_MEM_CACHE: dict[tuple, np.ndarray] = {}


def make_connection_matrix(
    D: int,
    T: int,
    S: int,
    seed: int | None = None,
    n_restarts: int = 5,
    sa_steps: int = 30_000,
) -> np.ndarray:
    if not (0 < S <= D):
        raise ValueError(f"Need 0 < S <= D, got S={S}, D={D}")

    rng = np.random.default_rng(seed)
    best_matrix, best_score = None, float("inf")

    for _ in range(n_restarts):
        m = _greedy_init(D, T, S, rng)
        m = _sa_improve(m, T, rng, steps=sa_steps)
        m = _balance_fix(m, T)
        score = _total_score(m)
        if score < best_score:
            best_score, best_matrix = score, m.copy()

    return best_matrix


def get_connection_matrix(D: int, T: int, S: int, seed: int, **kwargs) -> np.ndarray:
    """Cached `make_connection_matrix` (memory + disk)."""
    key = (D, T, S, seed)
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"conn_D{D}_T{T}_S{S}_seed{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            m = np.array(json.load(f), dtype=np.int8)
    else:
        m = make_connection_matrix(D, T, S, seed=seed, **kwargs)
        with open(path, "w") as f:
            json.dump(m.tolist(), f)

    _MEM_CACHE[key] = m
    return m


def evaluate_connection_matrix(M: np.ndarray) -> dict:
    D, T = M.shape
    row_sums, col_sums = M.sum(axis=1), M.sum(axis=0)
    ov = M.T.astype(np.int32) @ M.astype(np.int32)
    np.fill_diagonal(ov, 0)
    upper = ov[np.triu_indices(T, k=1)] if T > 1 else np.array([0])
    return {
        "shape": (D, T),
        "col_sum_min": int(col_sums.min()),
        "col_sum_max": int(col_sums.max()),
        "row_sum_min": int(row_sums.min()),
        "row_sum_max": int(row_sums.max()),
        "row_sum_std": float(row_sums.std()),
        "max_pairwise_overlap": int(upper.max()),
        "overlap_violation_pairs": int((upper > 1).sum()),
    }


# ── internals ────────────────────────────────────────────────────────────────


def _greedy_init(D: int, T: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """Pick S rows per column, minimizing row sum (balance) and co-occurrence
    with rows already chosen for this column (overlap)."""
    matrix = np.zeros((D, T), dtype=np.int8)
    row_sums = np.zeros(D, dtype=np.int32)
    pair_count = np.zeros((D, D), dtype=np.int32)  # columns containing both rows

    for col in range(T):
        available = np.ones(D, dtype=bool)
        selected: list[int] = []

        for _ in range(S):
            cands = np.where(available)[0]
            scores = row_sums[cands].astype(np.float64)
            for sel in selected:
                scores += pair_count[sel, cands] * 1_000.0
            scores += rng.uniform(0.0, 0.01, len(cands))  # random tie-break
            pick = int(cands[np.argmin(scores)])
            selected.append(pick)
            available[pick] = False

        for i in selected:
            matrix[i, col] = 1
            row_sums[i] += 1
        for i, k in combinations(selected, 2):
            pair_count[i, k] += 1
            pair_count[k, i] += 1

    return matrix


def _sa_improve(matrix: np.ndarray, T: int, rng: np.random.Generator, steps: int) -> np.ndarray:
    """Swap a 1 with a 0 inside a random column (column sums stay fixed).
    Objective = overlap_violations * 1000 + var(row_sums)."""
    m = matrix.copy()
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = m.T.astype(np.int32) @ m.astype(np.int32)
    np.fill_diagonal(ov, 0)

    T0, Tf = 5_000.0, 0.5
    decay = (Tf / T0) ** (1.0 / max(steps, 1))
    temp = T0

    for _ in range(steps):
        col = int(rng.integers(T))
        ones = np.where(m[:, col] == 1)[0]
        zeros = np.where(m[:, col] == 0)[0]
        if not len(ones) or not len(zeros):
            temp *= decay
            continue

        i_out, i_in = int(rng.choice(ones)), int(rng.choice(zeros))

        dov = m[i_in, :].astype(np.int32) - m[i_out, :].astype(np.int32)
        dov[col] = 0
        new_ov_col = ov[col, :] + dov
        overlap_delta = int(np.maximum(0, new_ov_col - 1).sum()) - int(
            np.maximum(0, ov[col, :] - 1).sum()
        )

        rs = row_sums.copy()
        rs[i_out] -= 1
        rs[i_in] += 1
        delta = overlap_delta * 1_000.0 + float(rs.var()) - float(row_sums.var())

        if delta <= 0.0 or rng.random() < np.exp(-delta / temp):
            m[i_out, col], m[i_in, col] = 0, 1
            row_sums = rs
            ov[col, :] = new_ov_col
            ov[:, col] = new_ov_col

        temp *= decay

    return m


def _balance_fix(matrix: np.ndarray, T: int, max_iters: int = 200) -> np.ndarray:
    """Move a 1 from a high-degree neuron to a low-degree one whenever that does
    not increase overlap violations."""
    m = matrix.copy()
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = m.T.astype(np.int32) @ m.astype(np.int32)
    np.fill_diagonal(ov, 0)

    for _ in range(max_iters):
        best_gain, best_swap = 1, None  # gain >= 2 guarantees variance decreases

        for col in range(T):
            ones = np.where(m[:, col] == 1)[0]
            zeros = np.where(m[:, col] == 0)[0]
            if not len(ones) or not len(zeros):
                continue

            top_ones = ones[np.argsort(row_sums[ones])[::-1]][:3]
            bot_zeros = zeros[np.argsort(row_sums[zeros])][:3]

            for i_out in top_ones:
                for i_in in bot_zeros:
                    gain = int(row_sums[i_out]) - int(row_sums[i_in])
                    if gain <= best_gain:
                        continue
                    dov = m[i_in, :].astype(np.int32) - m[i_out, :].astype(np.int32)
                    dov[col] = 0
                    new_ov_col = ov[col, :] + dov
                    viol_delta = int(np.maximum(0, new_ov_col - 1).sum()) - int(
                        np.maximum(0, ov[col, :] - 1).sum()
                    )
                    if viol_delta <= 0:
                        best_gain = gain
                        best_swap = (col, int(i_out), int(i_in), new_ov_col.copy())

        if best_swap is None:
            break

        col, i_out, i_in, new_ov_col = best_swap
        m[i_out, col], m[i_in, col] = 0, 1
        row_sums[i_out] -= 1
        row_sums[i_in] += 1
        ov[col, :] = new_ov_col
        ov[:, col] = new_ov_col

    return m


def _total_score(m: np.ndarray) -> float:
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = m.T.astype(np.int32) @ m.astype(np.int32)
    np.fill_diagonal(ov, 0)
    violations = int(np.maximum(0, ov - 1).sum()) // 2
    return violations * 1_000.0 + float(row_sums.var())
