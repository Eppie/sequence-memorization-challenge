"""Capacity search for HandCodedModel2 (top_fraction version — PRESERVED).

This is the original capacity search that sweeps (top_fraction, S). It has been
superseded by hc2_capacity_search.py, which searches integer top_n instead (the
optimum scales with S and a fixed top_fraction grid was too coarse). This file is
kept so the earlier approach and its result format remain available for
comparison. It writes the old-style grid files (hc2_sweep_d{d}_nfacts{nf}.json)
and appends to capacity_search_results.json — exactly as before.

Run it with:
    python -m modal run hand_coded_models/hc2_capacity_search_top_fraction.py
"""

import os
import json
import glob
from collections import defaultdict
from datetime import datetime

import numpy as np
import modal


# Fixed seed for generate_facts (the facts never change), and this file's dir.
seed = 42
_HERE = os.path.dirname(os.path.abspath(__file__))


# Settings used when running this file.
d = 32
accuracy_threshold = 0.9
any_all_most = "all"

n_attempts = 11
# Binary search stops when the bracket is within this fraction of its top end
# (hi - lo < PRECISION_FRACTION * hi), giving ~2% relative resolution on max_facts.
PRECISION_FRACTION = 0.02

if d == 16:
    S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif d == 32:
    S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
elif d == 64:
    S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
elif d == 128:
    S_sweep = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

top_fraction_sweep = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


app = modal.App("hc2-capacity-search-top-fraction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
)


@app.function(image=image, timeout=86400)
def _run_one(args: dict) -> dict:
    """Evaluate ONE model: a single (n_facts, S, top_fraction, attempt) point."""
    import sys
    import warnings
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    import torch
    from hc2 import (
        HandCodedModel2,
        HandCodedModel2Settings,
        make_connection_matrix,
    )

    S = args["S"]
    conn_seed = args["conn_seed"]
    torch.manual_seed(conn_seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conn = make_connection_matrix(D=args["d_ff"], T=args["output_vocab_size"],
                                      S=S, seed=conn_seed)

    settings = HandCodedModel2Settings(
        input_vocab_size=args["input_vocab_size"],
        output_vocab_size=args["output_vocab_size"],
        n_facts=args["n_facts"],
        seed=args["seed"],
        d_ff=args["d_ff"],
        n_neurons_per_label=S,
        use_top_n_or_top_fraction="top_fraction",
        top_fraction=args["top_fraction"],
    )
    model = HandCodedModel2(settings, precomputed_conn=conn)
    accuracy, _, _ = model.evaluate()
    return {
        "n_facts": args["n_facts"],
        "S": S,
        "top_fraction": args["top_fraction"],
        "attempt": args["attempt"],
        "conn_seed": conn_seed,
        "accuracy": accuracy,
    }


RESULTS_DIR = os.path.join(_HERE, "hc2_sweep_results")
# top_fraction grid files live in their own subfolder; capacity_search_results_*
# logs sit directly in RESULTS_DIR. See hc2_sweep_results/README.md.
GRIDS_DIR = os.path.join(RESULTS_DIR, "top_fraction_grids")


def _load_cached_records(d, n_facts):
    """Pool every grid record for (d, n_facts) from the old-style grid files."""
    records = []
    for path in sorted(glob.glob(os.path.join(GRIDS_DIR, "hc2_sweep_d*_nfacts*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("settings", {})
        if s.get("d") == d and s.get("n_facts") == n_facts:
            records.extend(payload["results"])
    return records


def _covers(records, S_sweep, top_fraction_sweep, n_attempts):
    """True if `records` already hold >= n_attempts runs for every (S, tf) wanted."""
    counts = defaultdict(int)
    for r in records:
        counts[(r["S"], r["top_fraction"])] += 1
    for S in S_sweep:
        for tf in top_fraction_sweep:
            if counts[(S, tf)] < n_attempts:
                return False
    return True


def _build_cells_for_n_facts(d, n_facts, n_attempts, top_fraction_sweep, S_sweep,
                             seed_offset=0):
    """One work unit per (S, top_fraction, attempt); each gets a unique conn_seed."""
    cells = []
    idx = seed_offset
    for S in S_sweep:
        for top_fraction in top_fraction_sweep:
            for attempt in range(n_attempts):
                cells.append({
                    "n_facts": n_facts,
                    "S": S,
                    "top_fraction": top_fraction,
                    "attempt": attempt,
                    "conn_seed": seed + 1 + idx,
                    "d_ff": d,
                    "seed": seed,
                    "input_vocab_size": 2 * d,
                    "output_vocab_size": d,
                })
                idx += 1
    return cells


def _save_records(d, n_facts, records, n_attempts, top_fraction_sweep, S_sweep):
    """Write a grid result file (one per n_facts) into top_fraction_grids/."""
    os.makedirs(GRIDS_DIR, exist_ok=True)
    payload = {
        "settings": {
            "d": d,
            "n_facts": n_facts,
            "n_attempts": n_attempts,
            "top_fraction_sweep": top_fraction_sweep,
            "S_sweep": S_sweep,
            "input_vocab_size": 2 * d,
            "output_vocab_size": d,
            "d_ff": d,
            "seed": seed,
            "metric": "accuracy",
        },
        "results": records,
    }
    out_path = os.path.join(GRIDS_DIR, f"hc2_sweep_d{d}_nfacts{n_facts}.json")
    base, ext = os.path.splitext(out_path)
    i = 1
    while os.path.exists(out_path):
        out_path = f"{base}_({i}){ext}"
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _evaluate_n_facts(d, n_attempts, n_facts, top_fraction_sweep, S_sweep,
                      accuracy_threshold=1.0, any_all_most="any", verbose=True):
    """Decide whether n_facts is storable by sweeping (top_fraction, S)."""
    if any_all_most not in ("any", "all", "most"):
        raise ValueError("any_all_most must be 'any', 'all', or 'most'")

    records = _load_cached_records(d, n_facts)
    if _covers(records, S_sweep, top_fraction_sweep, n_attempts):
        if verbose:
            print(f"    n_facts={n_facts}: loaded grid from cache ({len(records)} records)")
    else:
        cells = _build_cells_for_n_facts(d, n_facts, n_attempts, top_fraction_sweep, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: running {len(cells)} evaluations on Modal ...")
        records = list(_run_one.map(cells))
        path = _save_records(d, n_facts, records, n_attempts, top_fraction_sweep, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: wrote {len(records)} records to {path}")

    wanted_S = set(S_sweep)
    wanted_tf = set(top_fraction_sweep)
    cells = defaultdict(list)
    for r in records:
        if r["S"] in wanted_S and r["top_fraction"] in wanted_tf:
            cells[(r["S"], r["top_fraction"])].append(r.get("accuracy", r.get("best_guess_accuracy")))

    if any_all_most == "any":
        reduce_fn = np.max
    elif any_all_most == "all":
        reduce_fn = np.min
    else:
        reduce_fn = np.median

    best_score = -1.0
    best_combination = None
    for (S, tf), accs in cells.items():
        score = float(reduce_fn(accs))
        if score > best_score:
            best_score = score
            best_combination = (tf, S)

    success = best_score >= accuracy_threshold
    if verbose:
        tf, S = best_combination
        print(f"    n_facts={n_facts}: best {any_all_most} cell (top_fraction={tf}, S={S}) "
              f"score={best_score:.4f} {'>=' if success else '<'} {accuracy_threshold} "
              f"-> {'PASS' if success else 'fail'}")
    return success, best_combination


def find_max_facts(d, n_attempts, top_fraction_sweep, S_sweep,
                   accuracy_threshold=1.0, any_all_most="any", verbose=True):
    """Binary-search the maximum storable n_facts. max_possible = 4*d**2.

    Stops when the bracket is within PRECISION_FRACTION of its top end
    (hi - lo < PRECISION_FRACTION * hi)."""
    max_possible = 4 * d ** 2
    lo, hi = 1, max_possible
    best = 0
    best_combo = None

    if verbose:
        print(f"Searching for max storable facts in [{lo}, {hi}]  (d={d})\n")

    while hi - lo >= PRECISION_FRACTION * hi:
        mid = (lo + hi) // 2
        if verbose:
            print(f"Trying n_facts = {mid} ...")
        success, combo = _evaluate_n_facts(
            d, n_attempts, mid, top_fraction_sweep, S_sweep,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = mid, combo
            lo = mid + 1
            if verbose:
                print(f"OK  stored {mid} facts. Now searching: {lo} - {hi}\n")
        else:
            hi = mid - 1
            if verbose:
                print(f"X   failed at {mid} facts. Now searching: {lo} - {hi}\n")

    if hi == max_possible:
        if verbose:
            print(f"Trying n_facts = {max_possible} (max possible) ...")
        success, combo = _evaluate_n_facts(
            d, n_attempts, max_possible, top_fraction_sweep, S_sweep,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = max_possible, combo

    if verbose:
        print(f"\nMax storable facts: {best}  (best top_fraction, S = {best_combo})")
    return best, best_combo


CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, "capacity_search_results.json")


def _append_capacity_result(max_facts, best_combo):
    """Append one JSON line summarising this run; never touches previous lines."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_tf, best_S = (best_combo if best_combo is not None else (None, None))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d": d,
        "max_facts": max_facts,
        "best_top_fraction": best_tf,
        "best_S": best_S,
        "accuracy_threshold": accuracy_threshold,
        "any_all_most": any_all_most,
        "n_attempts": n_attempts,
        "precision_fraction": PRECISION_FRACTION,
        "S_sweep": S_sweep,
        "top_fraction_sweep": top_fraction_sweep,
    }
    with open(CAPACITY_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return CAPACITY_RESULTS_PATH


@app.local_entrypoint()
def main():
    best, best_combo = find_max_facts(
        d=d,
        n_attempts=n_attempts,
        top_fraction_sweep=top_fraction_sweep,
        S_sweep=S_sweep,
        accuracy_threshold=accuracy_threshold,
        any_all_most=any_all_most,
        verbose=True,
    )
    print(f"\nd={d}: max_facts={best}, best (top_fraction, S)={best_combo}")
    path = _append_capacity_result(best, best_combo)
    print(f"Appended result to {path}")
