"""Reproduce Figure 5 of the post: max facts vs model dimension, per condition.

For every (condition, d, accuracy threshold) we binary-search the largest
storable fact count, then fit max_facts = a * d^b / ln(d) and compare the
fitted (a, b) with the authors' published values.

    uv run python run_scaling.py --ds 16 32 64 --n-attempts 3
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from handcode.capacity import CONDITIONS, OUR_CONDITIONS, find_max_facts, fit_scaling, predict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# The authors' best-fit lines from Figure 5, as a * d^b / ln(d).
PUBLISHED = {
    ("trained", 1.0): (5.66, 2.06),
    ("trained", 0.9): (9.42, 1.97),
    ("hybrid", 1.0): (1.56, 2.07),
    ("hybrid", 0.9): (2.20, 2.02),
    ("hand-coded", 1.0): (2.17, 1.55),
    ("hand-coded", 0.9): (1.16, 1.93),
    ("rand-emb", 1.0): (0.151, 2.24),
    ("rand-emb", 0.9): (0.278, 2.21),
}


def _run(job: tuple) -> dict:
    import torch

    condition, d, threshold, n_attempts, threads = job
    # One BLAS thread per worker is right when jobs outnumber cores, but wrong
    # in the tail of a run: with two jobs left on a 16-core box it pins 14 cores
    # idle while the batched solves run single-threaded.
    torch.set_num_threads(threads)
    print(f"  start {condition:11s} d={d:<4d} acc>={threshold}", flush=True)
    result = find_max_facts(condition, d, threshold, n_attempts, verbose=False)
    print(
        f"  DONE  {condition:11s} d={d:<4d} acc>={threshold}  "
        f"max_facts={result.max_facts:<6d} ({result.seconds:.0f}s)",
        flush=True,
    )
    return result.to_json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 0.9])
    parser.add_argument("--n-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "scaling.json"))
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="re-print the comparison from an existing results file",
    )
    args = parser.parse_args()

    if args.report_only:
        with open(args.out) as f:
            payload = json.load(f)
        # `--ds` selects which sizes to tabulate; the file may hold others.
        report(payload["records"], args.ds)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    records = _load(args.out)
    done = {(r["condition"], r["d"], r["threshold"]) for r in records}

    pending = [
        (c, d, t)
        for c in args.conditions
        for t in args.thresholds
        for d in args.ds
        if (c, d, t) not in done
    ]
    # Split the cores over however many jobs will actually run concurrently.
    concurrent = max(1, min(args.workers, len(pending)))
    threads = max(1, (os.cpu_count() or 1) // concurrent)
    jobs = [(c, d, t, args.n_attempts, threads) for c, d, t in pending]
    print(f"{len(jobs)} searches to run ({len(done)} already in {args.out}) "
          f"on {args.workers} workers x {threads} BLAS threads\n")

    # Longest jobs first so the tail of the run is not one straggler.
    jobs.sort(key=lambda j: -j[1])  # longest (largest d) first

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            _save(args.out, records, args.n_attempts, args.ds)  # checkpoint as we go
    print(f"\nall searches done in {time.time() - started:.0f}s; wrote {args.out}\n")

    report(records, args.ds)


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["records"]


def _key(record: dict) -> tuple:
    return (record["condition"], record["d"], record["threshold"])


def _save(path: str, records: list[dict], n_attempts: int, ds: list[int]) -> None:
    """Checkpoint, merging with whatever is already on disk.

    Two runs covering different conditions may be in flight at once (searches
    are cheap enough that it is natural to launch a follow-up before the first
    has drained), and a blind overwrite would drop the other run's results.
    """
    merged = {_key(r): r for r in _load(path)}
    merged.update({_key(r): r for r in records})
    ordered = sorted(merged.values(), key=lambda r: (r["condition"], -r["threshold"], r["d"]))
    with open(path, "w") as f:
        json.dump({"n_attempts": n_attempts, "ds": ds, "records": ordered}, f, indent=2)


def report(records: list[dict], ds: list[int]) -> None:
    by_key: dict[tuple, dict[int, int]] = {}
    for r in records:
        by_key.setdefault((r["condition"], r["threshold"]), {})[r["d"]] = r["max_facts"]

    for threshold in (1.0, 0.9):
        label = "acc = 1" if threshold == 1.0 else "acc >= 0.9"
        print(f"\n=== {label} " + "=" * 62)
        header = "condition    " + "".join(f"{'d=' + str(d):>9s}" for d in ds)
        print(f"{header}   {'ours':>22s}   {'published':>22s}")
        for condition in CONDITIONS:
            got = by_key.get((condition, threshold))
            if not got:
                continue
            xs = [d for d in ds if d in got]
            ys = [got[d] for d in xs]
            counts = "".join(f"{got[d]:>9d}" if d in got else f"{'-':>9s}" for d in ds)
            if len(xs) >= 2:
                a, b = fit_scaling(xs, ys)
                ours = f"{a:.3g}*d^{b:.2f}/ln d"
            else:
                ours = "-"
            published = PUBLISHED.get((condition, threshold))
            pub = f"{published[0]:.3g}*d^{published[1]:.2f}/ln d" if published else "(no baseline)"
            print(f"{condition:<13s}{counts}   {ours:>22s}   {pub:>22s}")

        print("\n  per-d ratio to the published fit (1.00 = exact match):")
        for condition in CONDITIONS:
            got = by_key.get((condition, threshold))
            published = PUBLISHED.get((condition, threshold))
            if not got or not published:
                continue
            ratios = "".join(
                f"{got[d] / predict(*published, d):>9.2f}" if d in got else f"{'-':>9s}"
                for d in ds
            )
            print(f"{condition:<13s}{ratios}")

        # Our readouts are modifications of the hand-coded construction, so the
        # meaningful baseline for them is that construction, not a published fit.
        baseline = by_key.get(("hand-coded", threshold))
        if baseline and any(by_key.get((c, threshold)) for c in OUR_CONDITIONS):
            print("\n  ratio to the hand-coded construction (>1.00 = we store more):")
            for condition in OUR_CONDITIONS:
                got = by_key.get((condition, threshold))
                if not got:
                    continue
                ratios = "".join(
                    f"{got[d] / baseline[d]:>9.2f}" if d in got and d in baseline
                    else f"{'-':>9s}"
                    for d in ds
                )
                print(f"{condition:<13s}{ratios}")

    # Appendix D: the post reports best_S ~ sqrt(d) for the hand-coded model.
    print("\n=== winning hyperparameters " + "=" * 47)
    print(f"{'condition / acc':<22s}" + "".join(f"{'d=' + str(d):>12s}" for d in ds))
    for condition in ("hand-coded", "hybrid"):
        for threshold in (1.0, 0.9):
            cells = {
                r["d"]: (r["best_S"], r["best_top_fraction"])
                for r in records
                if r["condition"] == condition and r["threshold"] == threshold
            }
            if not cells:
                continue
            row = "".join(
                f"{f'{cells[d][0]}, {cells[d][1]:g}':>12s}" if d in cells else f"{'-':>12s}"
                for d in ds
            )
            print(f"{condition + ' acc=' + str(threshold):<22s}{row}")
    print(f"{'(sqrt(d) for ref)':<22s}" + "".join(f"{round(d**0.5, 1):>12g}" for d in ds))


if __name__ == "__main__":
    main()
