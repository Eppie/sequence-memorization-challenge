"""Precompute the connection matrices the sweep needs, in parallel.

A connection matrix depends only on (D, T, S, seed) -- not on n_facts or
top_fraction -- so every one of them can be built once and reused across the
whole binary search. Building one takes ~3s (simulated annealing), and the
search needs a few hundred, so doing this up front across cores turns the
dominant cost of the experiment into a one-off minute.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from handcode.capacity import SweepGrid
from handcode.connection import CACHE_DIR, get_connection_matrix


def build(job: tuple[int, int, int, int]) -> tuple[tuple, float]:
    d, _t, s, seed = job
    started = time.time()
    get_connection_matrix(D=d, T=_t, S=s, seed=seed)
    return job, time.time() - started


def jobs_for(ds: list[int], n_attempts: int) -> list[tuple[int, int, int, int]]:
    wanted = set()
    for d in ds:
        for condition in ("hand-coded", "hybrid"):
            for s in SweepGrid.for_d(d, condition).S_values:
                for attempt in range(n_attempts):
                    wanted.add((d, d, s, 1000 + attempt))
    return sorted(wanted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--n-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    args = parser.parse_args()

    all_jobs = jobs_for(args.ds, args.n_attempts)
    todo = [
        j
        for j in all_jobs
        if not os.path.exists(
            os.path.join(CACHE_DIR, f"conn_D{j[0]}_T{j[1]}_S{j[2]}_seed{j[3]}.json")
        )
    ]
    print(f"{len(all_jobs)} matrices needed, {len(todo)} missing -> building on "
          f"{args.workers} workers")

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (job, seconds) in enumerate(pool.map(build, todo), start=1):
            d, _t, s, seed = job
            print(f"  [{i}/{len(todo)}] d={d} S={s} seed={seed}  {seconds:.1f}s "
                  f"(elapsed {time.time() - started:.0f}s)", flush=True)

    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
