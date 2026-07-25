"""Capacity search for HybridModel2 (hand-coded up matrix + TRAINED down matrix), on Modal.

Mirror of hc2_capacity_search.py, with one change to what a grid cell does: the
model is a HybridModel2 — its up matrix is generated exactly like
HandCodedModel2's (and frozen), while its down matrix + bias are randomly
initialised and trained with full-batch CE loss (plain Adam, lr=1e-2, up to
5000 epochs, early stopping — see hc2_hybrid.py). The cell's score is the best
accuracy observed during training.

Everything else mirrors hc2_capacity_search.py: the same (S, knob) hyper-
parameter grids (knob = top_n or top_fraction, selected by use_top_fraction),
the same 11 attempts per cell with the same per-attempt column permutation and
tie-seeding, the same any/all/most reduction, and a binary search over n_facts
that stops at a 2%-relative bracket (hi - lo < 0.02 * hi). The connection-matrix
Volume (hc2-conn-cache) is shared with the original search — the matrices are
identical, so nothing is rebuilt.

Cost note: a cell is now a training run (up to 5000 epochs) instead of one
forward pass, so this sweep is FAR more expensive than the hand-coded one —
especially at large d and large n_facts. Cells still fan out one Modal task
each, so wall-clock time is bounded by Modal's container fan-out. Because the
up matrix is frozen, training precomputes the hidden activations once and each
epoch is just a (n_facts x d_ff) @ (d_ff x d) linear layer.

Outputs go to SEPARATE hybrid-suffixed locations so they never mix with the
hand-coded runs: grids in hc2_sweep_results/{mode}_grids_hybrid/ and the
capacity log in capacity_search_results_{mode}_hybrid.json.

Run it with:
    python -m modal run hand_coded_models/hc2_hybrid_capacity_search.py
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
RESULTS_DIR = os.path.join(_HERE, "hc2_sweep_results")

use_top_fraction = True

# The search knob is either integer top_n or float top_fraction (set by
# use_top_fraction above). Each mode writes to its own grid subfolder and capacity
# log so the two never mix; KNOB is the per-record/settings field name for the knob.
_mode_tag = "topfrac" if use_top_fraction else "topn"
KNOB = "top_fraction" if use_top_fraction else "top_n"

# Grid files live in their own hybrid-suffixed subfolder; only the
# capacity_search_results_* logs sit directly in RESULTS_DIR.
GRIDS_DIR = os.path.join(RESULTS_DIR, f"{_mode_tag}_grids_hybrid")
CONN_CACHE_DIR = os.path.join(_HERE, "conn_cache")


# ── Settings ──────────────────────────────────────────────────────────────────
n_attempts = 11

# Training recipe for the down matrix (see hc2_hybrid.train_down_matrix).
N_EPOCHS = 5000
LR = 1e-2
PATIENCE = 100

# Each capacity search runs one (d, accuracy_threshold, any_all_most) config.
CONFIGS = [
    dict(d=d, accuracy_threshold=thr, any_all_most=aam)
    for d in [16, 32, 64, 128, 256]
    for thr in [0.9, 1.0]
    for aam in ["any", "all", "most"]
]

testing = False  # small/cheap end-to-end validation run

# Binary search stops when the bracket is within this fraction of its top end
# (hi - lo < PRECISION_FRACTION * hi), giving ~2% relative resolution on max_facts.
PRECISION_FRACTION = 0.02


def S_sweep_for(d):
    """Which n_neurons_per_label values to sweep for a given model size d."""
    if d <= 16:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif d <= 32:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
    elif d <= 64:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif d <= 128:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    else:
        return list(range(1, 25))


def top_n_sweep_for(S):
    """Integer top_n grid for a given S: 0 .. round(2.2*S) (covers observed optima)."""
    return list(range(0, round(2.2 * S) + 1))

top_frac_sweep = [0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.18,
                  0.2 , 0.22, 0.24, 0.26, 0.28, 0.3 , 0.32, 0.34, 0.36, 0.38]


def knob_sweep_for(S):
    """The grid of suppression-strength values to sweep for a given S."""
    return top_frac_sweep if use_top_fraction else top_n_sweep_for(S)


# All connection matrices use this fixed seed, so a given (d, S) maps to ONE matrix.
MATRIX_SEED = seed


def _tie_seed(S, attempt):
    """Per-(S, attempt) seed for the per-attempt column permutation and the
    constructor's random tie-breaking."""
    return 10_000 * S + attempt


# ── Modal app + image ─────────────────────────────────────────────────────────
app = modal.App("hc2-hybrid-capacity-search")

# Self-contained image: ships hc2.py + hc2_hybrid.py + the repo-root modules they
# import. No top-level import of `hand_coded_models` (Modal re-imports THIS file
# inside every container and that package is not on the container path).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "wandb")
    .add_local_python_source("models", "device")
    .add_local_file(os.path.join(_HERE, "hc2.py"), "/root/hc2.py")
    .add_local_file(os.path.join(_HERE, "hc2_hybrid.py"), "/root/hc2_hybrid.py")
)

# Connection matrices live on the SAME Modal Volume as the hand-coded search
# (hc2-conn-cache): the matrices depend only on (d, S, MATRIX_SEED), which are
# identical here, so every previously built matrix is reused as-is.
conn_volume = modal.Volume.from_name("hc2-conn-cache", create_if_missing=True)
CONN_VOL_MOUNT = "/conn_cache"


def _conn_vol_name(D, S):
    return f"d{D}_s{S}.npy"


@app.function(image=image, timeout=86400, volumes={CONN_VOL_MOUNT: conn_volume})
def _build_conn(arg):
    """Build the ONE connection matrix for a (D, S) and persist it to the Volume.

    Idempotent: returns immediately if the file already exists. arg = (D, S)."""
    import os
    import sys
    import warnings
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from hc2 import make_connection_matrix

    D, S = arg
    path = os.path.join(CONN_VOL_MOUNT, _conn_vol_name(D, S))
    if os.path.exists(path):
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_connection_matrix(D=D, T=D, S=S, seed=MATRIX_SEED)
    np.save(path, m)
    conn_volume.commit()


@app.function(image=image, timeout=86400, volumes={CONN_VOL_MOUNT: conn_volume})
def _run_one(cell):
    """Train + evaluate one (S, attempt, knob) cell for a given n_facts.

    Builds a HybridModel2 (hand-coded frozen up matrix, random down matrix) and
    trains the down matrix with CE loss; returns the best accuracy
    seen during training. Per-attempt variety comes from permuting the matrix's
    label columns, re-seeding the constructor's tie-breaking, and the down-
    matrix init — no connection matrix is rebuilt."""
    import os
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    import torch
    from hc2 import HandCodedModel2Settings
    from hc2_hybrid import HybridModel2, train_down_matrix

    d = cell["d"]
    S = cell["S"]
    # (d_ff, n_labels) base matrix for this (d, S), read from the Volume not the input.
    path = os.path.join(CONN_VOL_MOUNT, _conn_vol_name(d, S))
    try:
        conn = np.load(path)
    except FileNotFoundError:
        conn_volume.reload()
        conn = np.load(path)
    attempt = cell["attempt"]
    n_facts = cell["n_facts"]
    tie_seed = cell["tie_seed"]
    use_top_fraction = cell["use_top_fraction"]
    knob_value = cell["knob_value"]
    knob_index = cell["knob_index"]

    # Per-attempt randomness without rebuilding: permute the label COLUMNS.
    perm = np.random.default_rng(tie_seed).permutation(conn.shape[1])
    conn = conn[:, perm]

    # Reproducible random tie-breaking inside the constructor (varies per attempt).
    torch.manual_seed(tie_seed + knob_index)
    if use_top_fraction:
        mode, top_n_arg, top_fraction_arg = "top_fraction", 0, knob_value
    else:
        mode, top_n_arg, top_fraction_arg = "top_n", knob_value, 0.2
    settings = HandCodedModel2Settings(
        input_vocab_size=cell["input_vocab_size"],
        output_vocab_size=cell["output_vocab_size"],
        n_facts=n_facts,
        seed=cell["seed"],
        d_ff=cell["d_ff"],
        n_neurons_per_label=S,
        use_top_n_or_top_fraction=mode,
        top_n=top_n_arg,
        top_fraction=top_fraction_arg,
    )
    model = HybridModel2(settings, precomputed_conn=conn,
                         init_seed=tie_seed + knob_index)
    accuracy, epochs_run = train_down_matrix(
        model, n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE)
    knob_key = "top_fraction" if use_top_fraction else "top_n"
    return {
        "n_facts": n_facts,
        "S": S,
        knob_key: knob_value,
        "attempt": attempt,
        "tie_seed": tie_seed,
        "accuracy": accuracy,
        "epochs_run": epochs_run,
    }


# ── Connection-matrix cache (Modal Volume, seeded from legacy local conn_cache/) ──

def _conn_path(D, S):
    return os.path.join(CONN_CACHE_DIR, f"d{D}_s{S}.npy")


def _ensure_conn(d, S_sweep, verbose=False):
    """Ensure every (d, S) connection matrix exists on the Volume (one per S)."""
    # Reuse prior local builds by uploading them to the Volume (idempotent overwrite).
    to_upload = [(p, _conn_vol_name(d, S)) for S in S_sweep
                 if os.path.exists(p := _conn_path(d, S))]
    if to_upload:
        if verbose:
            print(f"  Uploading {len(to_upload)} cached matrices for d={d} to the Volume ...")
        with conn_volume.batch_upload(force=True) as batch:
            for local_path, remote_name in to_upload:
                batch.put_file(local_path, remote_name)

    # Build any with no local copy; _build_conn writes them to the Volume directly.
    to_build = [(d, S) for S in S_sweep if not os.path.exists(_conn_path(d, S))]
    if to_build:
        if verbose:
            print(f"  Building {len(to_build)} connection matrices on Modal "
                  f"(d={d}, one per S); written to the Volume, reused everywhere ...")
        list(_build_conn.map(to_build))
    elif verbose and not to_upload:
        print(f"  All connection matrices for d={d} already present.")


# ── Result caching (grid files: hc2_hybrid_sweep_{mode}_d{d}_nfacts{nf}.json) ──

def _grid_glob():
    return os.path.join(GRIDS_DIR, f"hc2_hybrid_sweep_{_mode_tag}_d*_nfacts*.json")


def _load_cached_records(d, n_facts):
    """Pool every grid record for (d, n_facts) from disk."""
    records = []
    for path in sorted(glob.glob(_grid_glob())):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        s = payload.get("settings", {})
        if s.get("d") == d and s.get("n_facts") == n_facts:
            records.extend(payload["results"])
    return records


def _covers(records, S_sweep, n_attempts):
    """True if `records` already hold >= n_attempts runs for every (S, knob) wanted."""
    counts = defaultdict(int)
    for r in records:
        counts[(r["S"], r[KNOB])] += 1
    for S in S_sweep:
        for knob in knob_sweep_for(S):
            if counts[(S, knob)] < n_attempts:
                return False
    return True


def _build_cells_for_n_facts(d, n_facts, n_attempts, S_sweep):
    """One Modal task per (S, attempt, knob) training run.

    Cells carry only (d, S) to locate the matrix on the Volume — never the matrix
    itself — so each input stays tiny and Modal can fan out freely."""
    cells = []
    for S in S_sweep:
        for a in range(n_attempts):
            tie_seed = _tie_seed(S, a)
            for knob_index, knob_value in enumerate(knob_sweep_for(S)):
                cells.append({
                    "d": d,
                    "S": S,
                    "attempt": a,
                    "n_facts": n_facts,
                    "tie_seed": tie_seed,
                    "use_top_fraction": use_top_fraction,
                    "knob_value": knob_value,
                    "knob_index": knob_index,
                    "input_vocab_size": 2 * d,
                    "output_vocab_size": d,
                    "d_ff": d,
                    "seed": seed,
                })
    return cells


def _save_records(d, n_facts, records, n_attempts, S_sweep):
    """Write a grid result file (one per n_facts) into the mode's grid subfolder."""
    os.makedirs(GRIDS_DIR, exist_ok=True)
    payload = {
        "settings": {
            "d": d,
            "n_facts": n_facts,
            "n_attempts": n_attempts,
            "S_sweep": S_sweep,
            "input_vocab_size": 2 * d,
            "output_vocab_size": d,
            "d_ff": d,
            "seed": seed,
            "metric": "accuracy",
            "search_mode": KNOB,
            "model": "HybridModel2",
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "patience": PATIENCE,
        },
        "results": records,
    }
    out_path = os.path.join(GRIDS_DIR, f"hc2_hybrid_sweep_{_mode_tag}_d{d}_nfacts{n_facts}.json")
    base, ext = os.path.splitext(out_path)
    i = 1
    while os.path.exists(out_path):
        out_path = f"{base}_({i}){ext}"
        i += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ── Grid evaluation ───────────────────────────────────────────────────────────

def _evaluate_n_facts(d, n_attempts, n_facts, S_sweep,
                      accuracy_threshold=1.0, any_all_most="any", verbose=False):
    """Decide whether n_facts is storable, by sweeping (S, knob).

    Returns (success, best_combination) where best_combination is the (knob, S)
    whose any/all/most score is highest. Uses the disk cache if a covering grid
    already exists; otherwise runs the training cells on Modal and saves.
    """
    if any_all_most not in ("any", "all", "most"):
        raise ValueError("any_all_most must be 'any', 'all', or 'most'")

    records = _load_cached_records(d, n_facts)
    if _covers(records, S_sweep, n_attempts):
        if verbose:
            print(f"    n_facts={n_facts}: loaded grid from cache ({len(records)} records)")
    else:
        cells = _build_cells_for_n_facts(d, n_facts, n_attempts, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: running {len(cells)} training cells on Modal ...")
        records = list(_run_one.map(cells))
        path = _save_records(d, n_facts, records, n_attempts, S_sweep)
        if verbose:
            print(f"    n_facts={n_facts}: wrote {len(records)} records to {path}")

    # Collect each (S, knob) cell's runs, restricted to the wanted grid.
    wanted = {(S, knob) for S in S_sweep for knob in knob_sweep_for(S)}
    cells = defaultdict(list)
    for r in records:
        key = (r["S"], r[KNOB])
        if key in wanted:
            cells[key].append(r.get("accuracy", r.get("best_guess_accuracy")))

    # Reduce each cell to one score matching the rule, so success <=> best >= threshold.
    #   any -> max over runs ; all -> min over runs ; most -> median over runs
    reduce_fn = {"any": np.max, "all": np.min, "most": np.median}[any_all_most]

    best_score = -1.0
    best_combination = None
    for (S, knob), accs in cells.items():
        score = float(reduce_fn(accs))
        if score > best_score:
            best_score = score
            best_combination = (knob, S)

    success = best_score >= accuracy_threshold
    if verbose and best_combination is not None:
        knob, S = best_combination
        print(f"    n_facts={n_facts}: best {any_all_most} cell ({KNOB}={knob}, S={S}) "
              f"score={best_score:.4f} {'>=' if success else '<'} {accuracy_threshold} "
              f"-> {'PASS' if success else 'fail'}")
    return success, best_combination


# ── Binary search over n_facts ────────────────────────────────────────────────

def find_max_facts(d, n_attempts, S_sweep,
                   accuracy_threshold=1.0, any_all_most="any", verbose=False):
    """Binary-search the maximum storable n_facts. max_possible = (2*d)^2 = 4*d**2.

    Stops when the bracket is within PRECISION_FRACTION of its top end
    (hi - lo < PRECISION_FRACTION * hi). Returns (best_n_facts, best_combination)
    with best_combination = (knob, S).
    """
    # Ensure the connection matrices for this d are on the Volume ONCE (one per S).
    _ensure_conn(d, S_sweep, verbose=verbose)

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
            d, n_attempts, mid, S_sweep,
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

    if hi == max_possible:  # loop never tests the top end
        if verbose:
            print(f"Trying n_facts = {max_possible} (max possible) ...")
        success, combo = _evaluate_n_facts(
            d, n_attempts, max_possible, S_sweep,
            accuracy_threshold=accuracy_threshold,
            any_all_most=any_all_most, verbose=verbose,
        )
        if success:
            best, best_combo = max_possible, combo

    if verbose:
        print(f"\nMax storable facts: {best}  (best {KNOB}, S = {best_combo})")
    return best, best_combo


# ── Result logging + driver ───────────────────────────────────────────────────
# Separate log from every hand-coded run: the hybrid (trained-down) results live
# in their own _hybrid-suffixed file, so nothing old is mixed with or overwritten.
CAPACITY_RESULTS_PATH = os.path.join(RESULTS_DIR, f"capacity_search_results_{_mode_tag}_hybrid.json")


def _append_capacity_result(d, max_facts, best_combo, accuracy_threshold,
                            any_all_most, n_attempts, S_sweep):
    """Append one JSON line summarising a run; never touches previous lines."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_knob, best_S = (best_combo if best_combo is not None else (None, None))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d": d,
        "max_facts": max_facts,
        f"best_{KNOB}": best_knob,
        "best_S": best_S,
        "accuracy_threshold": accuracy_threshold,
        "any_all_most": any_all_most,
        "n_attempts": n_attempts,
        "precision_fraction": PRECISION_FRACTION,
        "S_sweep": S_sweep,
        "search_mode": KNOB,
        "model": "HybridModel2",
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "patience": PATIENCE,
    }
    with open(CAPACITY_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return CAPACITY_RESULTS_PATH


@app.local_entrypoint()
def main():
    configs = CONFIGS
    attempts = n_attempts
    if testing:
        # Cheap end-to-end validation: smallest d, few attempts.
        configs = [dict(d=16, accuracy_threshold=1.0, any_all_most="any")]
        attempts = 2

    print(f"Running {len(configs)} capacity search config(s).")
    for cfg in configs:
        d = cfg["d"]
        thr = cfg["accuracy_threshold"]
        aam = cfg["any_all_most"]
        S_sweep = S_sweep_for(d)

        print(f"\n===== d={d}, accuracy_threshold={thr}, any_all_most={aam} =====")
        best, combo = find_max_facts(
            d, attempts, S_sweep,
            accuracy_threshold=thr, any_all_most=aam, verbose=False,
        )
        print(f"d={d}: max_facts={best}, best ({KNOB}, S)={combo}")
        path = _append_capacity_result(
            d, best, combo, thr, aam, attempts, S_sweep)
        print(f"Appended result to {path}")
