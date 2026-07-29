# Handoff — the replication guard: parameterization, an exhausted optimization search, and seed 43 in flight

**Date:** 2026-07-27
**Repo:** `/Users/eppie/claude_projects/handcode`, worktree
`.claude/worktrees/replication-guard` on branch `worktree-replication-guard`
(branched from `main` @ `2ef6b67`).
**Local state (updated 2026-07-28): MERGED INTO `main`** at `2ab3e31` — the two
probe patches, the result JSONs, `make_replication_table.py` and the draft
section, plus a merge of main's softseed work (`0c3c4a3`). 69 tests green.
`main` is 3 commits ahead of `origin/main`; **nothing has been pushed.**
**Supersedes nothing.** Extends `2026-07-26-001-gate-quality-program.md` item 2
("Replication guard"), which is the task this session took on.

---

## What this session established

### 1. The probes could not run any cell but the published one

`d`, `n_facts`, and the fact seed were module constants
(`probe_gatequality.D/N_FACTS`, the literal `42` inside `setup()` and
`curve_ascend_worker`), not flags. **No replication was runnable as-is** —
"run it at d=64" was not a command, it was a code change. Now:

    uv run python probe_gatequality.py --phase predict --fact-seed 43
    uv run python probe_flippolicy.py  --phase stride --fact-seed 43 --stride-seed edge

`configure(d, n_facts, fact_seed)` in `probe_gatequality` sets the cell and
re-derives every cache/output path. **The default cell (32, 1584, 42) keeps the
original paths byte-for-byte**, so published caches and JSONs still resolve;
off-default cells get a `_d{d}_n{n}_s{seed}` suffix on `GATES_DIR`,
`gatequality.json`, and `flippolicy.json`. `probe_flippolicy.refresh_paths()`
mirrors this.

### 2. THE TRAP THAT WOULD HAVE FAKED THE WHOLE RESULT

`ProcessPoolExecutor` uses **spawn** on macOS. A spawned child re-imports the
probe module and sees the *module-level defaults* — not whatever `configure()`
set in the parent. Both ascent pools (`probe_flippolicy.run_ascents`,
`probe_gatequality`'s curve phase) therefore had to pass
`initializer=configure, initargs=(D, N_FACTS, FACT_SEED)`. Without it every
worker silently ascends the **d=32/seed-42** gate and the driver files the
result under seed 43 — a replication that always "reproduces" because it never
changed cell. This is now done and commented in-source; **do not remove it, and
add it to any new pool.**

### 3. The optimization search is exhausted — ten levers, one modest win, declined

LP solve time is ~99% of runtime (assembly is 0.1–1.5%). All timings d=32,
n=1584, high power mode (`pmset powermode 2`; low power is ~1.37× slower and
*mid-run mode switches invalidate timings* — a real trap).

| lever | measured | verdict |
|---|---|---|
| vectorize LP assembly | 99–99.9% of time is inside HiGHS | dead, 1% ceiling |
| lazy pattern rows (004's own suggestion) | 3.4%/2.2% of rows bind, but at 2 nnz each = 13% of nnz → **1.17× / 0.81×** | dead |
| `--ipm-tol` (exists; commit f079e67 claims 2–4×) | **1.02×**; direction identical (cos 1.000000, same 253 flips) | does not reproduce |
| scipy `threads` option | **not in scipy's recognised list** — passed verbatim, silently ignored | invalid |
| highspy `threads` > 1 | returns in 0.00s with `kNotset` + garbage objective; wheel has no parallel build | hard no |
| highspy 1.15.1 vs scipy's vendored 1.12.0 | 27.7s vs 26.1s | slower |
| `run_crossover=off` (highspy only) | 21.30s vs 26.07s, obj matches to 7e-9 | **1.22×, the only win** |
| k0 12→8 | 1.68× (trained) / 1.06× (digit) but γ 11.593 → 11.674 | unreliable |
| k0 12→6 | 1.83× but γ 11.593 → **9.642** | no |
| warm-start simplex | cold simplex **times out at 900s**; warm 752.87s vs **7.69s** cold IPM | dead |

**Recommendation: do not adopt highspy.** A ~1.15× net (crossover's 1.22× minus
the version's 0.94×) is not worth swapping the solver binary underneath a study
whose purpose is trusting the numbers — every published figure came from
scipy's vendored **HiGHS 1.12.0**. `highspy` was installed into the venv for
testing only (`uv pip install`, `pyproject.toml` untouched); uninstall or leave,
nothing depends on it.

**Dynamic programming does not apply.** The stride loop is a fixed-point
iteration (round r+1 consumes round r's pattern) with no overlapping
subproblems; the ascents are already independent. The one DP-adjacent idea —
reuse an adjacent round's solution — is the warm-start row above, and it is
~100× *worse* than solving cold.

### 4. Process-level parallelism is sublinear — bandwidth, not cores

HiGHS IPM is single-threaded here by every route, so the only lever left was
running independent runs concurrently. Measured with 6–8 concurrent stride
processes on 16 cores: **each process holds a full core at 100% CPU, yet
per-round wall time rose from 31.4s to ~5 min (~10×).** They are contending for
memory bandwidth (each solve holds 1.3–7 GB and IPM is bandwidth-bound), not
CPU. **An earlier estimate in this session claimed ~6× from process
parallelism; that is wrong and is corrected here.** Budget concurrency at
roughly 2–4 simultaneous LP processes, not 16, and re-measure before trusting
any wall-clock plan.

### 5. Cost anchors (d=32, high power, uncontended)

| kernel | rows | nvar | time |
|---|---|---|---|
| stride emb-LP (`lp_spread`, no pattern rows) | 19,008 | 5,680 | 24.6s |
| stride readout-LP (spread) | 19,008 | 2,608 | 6.8s |
| **one stride round, `--oracle lp`** | | | **31.4s** |
| ascent emb-LP (max-min + pattern rows) | 69,696 | 4,097 | 10.0s |
| ascent readout-LP | 19,008 | 1,025 | 7.1s |
| one `ascend_best` (3 rounds) | | | ~0.9 min |

Do **not** use §21's "~45 s per round" as a cost cross-check: it reproduces only
under low power mode. Training is free at both sizes (180 epochs: 0.2s at d=32,
0.7s at d=64).

### 6. d=64 is not merely slow — it is intractable under the current formulation

At d=64, n=6336 (load-fraction matched, 38.7% of `4d²`): the ascent emb-LP is
**481,536 rows** (76,032 margin + 405,504 pattern), matching 004's "~550k rows,
compute-bound". **One stride emb-LP ran >18 minutes without converging** against
24.6s at d=32 — ≥45×, and it **blew through its `time_limit: 900`** (HiGHS
checks the limit between IPM iterations, and at nvar=22,720 an iteration is tens
of seconds; crossover is the likely tail). 440 stride rounds is >130 h before
ascents. **The 900s guard does not reliably stop a d=64 solve, so an unattended
d=64 run can hang indefinitely.**

Two semantic problems must be settled before any d=64 number means anything:

* **Epoch 180 is not the same fitting state across d.** d=32 → train acc 0.867
  (just under the feasibility edge); d=64 → **0.698**. §19's "epoch-180 seed"
  and §20's epoch ladder are defined by position *relative to the edge*, not by
  epoch number. Copying epoch numbers replicates a different object.
* **n=1584 is "the digit code's capacity point"**, a *measured* quantity. At
  d=64 it must be re-measured, not scaled by 4×. This session used 6336
  (load-fraction matched) for costing only — not as a scientific choice.

### 7. A free result from a failed optimization experiment

Testing warm starts, a perturbed instance was built by flipping **0.5% of the
trained gate's bits at random**. That made the LP **infeasible** (γ = 0, the
Prop 1(a) do-nothing point). The stride process flips the *same fraction* per
round via near-ties and stays feasible while improving. Independent
corroboration of §17's "flips are extreme near-ties": at matched flip count,
random flips destroy feasibility, near-tie flips do not. Worth a proper
experiment (n random-flip trials vs the stride's own flips) — it would sharpen
§17 from "GD chooses near-ties" to "near-tieness is *necessary*, not incidental".

---

## Replication results — three fact seeds, two dimensions

**The protocol gate passed exactly**: under the patched code the published cell
reproduces with sigma90 ratio **1.0000** on all four gates. Every number below
comes from code verified against the original.

| row | published (d=32,s42) | seed 43 | seed 44 | d=64 s42 | verdict |
|---|---|---|---|---|---|
| S14 trained ceiling | 4.40e-2 | 4.41e-2 | 4.36e-2 | 3.17e-2 | reproduces |
| S14 digit m=2 | 2.85e-3 | 2.52e-3 | 2.81e-3 | not run | reproduces |
| S14 random / init | infeasible | infeasible | infeasible | **infeasible** | reproduces |
| S17 dense curve ep200-280 | 1.39-3.7e-2 | 1.51-3.73e-2 (ratios 1.01-1.09) | — | — | reproduces |
| S19 monotone climb | ->1.80e-2 @r40 | ->1.66e-2 | ->1.95e-2 | — | reproduces |
| S19 crossing from infeasible | ~4 rounds (ep180) | **ep150 crosses @r36** | — | — | reproduces (different seed epoch) |
| S20 ep50 dies | dies | dies | dies | — | reproduces |
| S20 ep100 crosses | ~r75 | **crosses @r104** | **crosses @r92** | — | reproduces, slower |
| S20 softness (GD side) | 0.5%q = 0.0011 | 0.0010 | — | — | reproduces |
| S21 matvec >= exact LP | 2.16 vs 1.80e-2 | 2.09 vs 1.66e-2 | — | — | reproduces |
| S21 0.2% step | 1.66-1.76e-2 | 1.70-1.76e-2 | — | — | reproduces |
| S21 5% step | destroyed 1 round | destroyed 1 round | — | — | reproduces |

**The three headline conclusions all survive.** The trained ceiling is
4.40/4.41/4.36e-2 across three seeds (<1% spread); random and init are
infeasible in *every* cell including d=64; the matvec-over-exact-LP ordering
and the 0.2%-works/5%-destroys contrast both hold.

**One thing does not transfer, and it is about epoch numbers, not mechanisms.**
(This section originally listed two. The second — S20's bracket — was written
while the ep100 runs were still short of their crossing; both have since
completed and both cross. Corrected below.)

1. **The epoch-180 seed.** At seeds 43 and 44 epoch 180 is *already feasible*
   (7.6e-3), so S19's "infeasible seed crosses in ~4 rounds" cannot be tested
   from it. Worse (and this is the sharper finding): within **one** seed, two
   independently trained micro-realizations disagree — `window_checkpoints` has
   seed 43's ep180 infeasible, `edge_state` has it feasible. Torch CPU training
   is not bit-deterministic, so "epoch 180" does not name a fixed state even at
   fixed seed. Run from seed 43's *actual* edge (ep150) the crossing reproduces
   cleanly: infeasible through r32, crosses r36, monotone to 1.05e-2 by r60.
2. **S20's bracket (50, 100] does transfer** — both edges, both seeds. ep50
   dies at both. ep100 is infeasible for a long prefix and then crosses: seed 43
   at **r104** (2.27e-3 -> 6.84e-3 by r120), seed 44 at **r92** (3.19e-3 ->
   1.11e-2 by r120), against the published ~r75. Only the rate moves, 1.2-1.4x
   slower, matching S19's rate-not-mechanism pattern. Internal ordering also
   holds: at seed 43, ep150 crosses at r36 and ep100 at r104 — nearer the edge,
   sooner across. **Lesson for the next agent: do not read a stride run that has
   not exhausted its round budget as a negative result.** The earlier "does not
   transfer" was called at r92 of 120, twelve rounds before the crossing.

**New result (S17 strengthened).** Flip 0.5% of the trained gate's bits — the
stride's own quota — two ways. The 0.5% *closest to zero*: ceiling 4.4008e-2,
accuracy 1.000, gate intact. The same count *at random*: **infeasible**, three
draws for three, at both seed 42 and seed 43 (6/6). S17 showed GD only flips
near-ties; this shows near-tie-ness is **necessary**, not incidental. Found by
accident while testing LP warm starts.

**Artifacts.** `results/gatequality_d32_n1584_s4{3,4}.json`,
`results/gatequality_d64_n6336_s42.json`,
`results/flippolicy_s4{3,4}_*.json`, and `make_replication_table.py`
(regenerates the whole table from the JSONs). Draft section in
`FINDINGS_22_DRAFT.md` — since folded into `FINDINGS.md` as **§24**
(2026-07-28; the draft file is gone).

## Traps added this session

* **spawn + module globals** (§2 above). The single most dangerous thing here.
* **`from probe_gatequality import GATES_DIR` binds the value at import time**,
  so the name never follows a later `configure()`. This is why the patched
  probes import the module (`import probe_gatequality as pgq`) and reference
  `pgq.GATES_DIR` at use sites instead. `probe_ordering.py` (added on main in
  `5e20f6d`) still does the from-import; harmless today because it takes no cell
  flags and only ever runs the default cell, but it will silently measure d=32 /
  seed 42 the moment anyone gives it `--fact-seed`. Convert it at that point,
  not after.
* **`pgrep -f <pattern>` matches the waiting shell's own command line**, so
  `until ! pgrep -qf 'foo.py'; do sleep 5; done` can spin forever and a chained
  launch never fires. Cost one silently-never-started experiment. Wait by PID
  (`while kill -0 $PID`) or use a marker file.
* **A stride run exits 0 after breaking out early.** An LP failure prints
  `[stride] LP failed at round N` and `break`s; the ascents then run on whatever
  snapshots exist and the process exits cleanly. **Check the round count in the
  JSON, never the exit code.** This is the same failure mode that produced the
  earlier false "S20 does not transfer" — a short run read as a negative result.
  Seed 44's ep50 arm is the standing example: 900s `time_limit` exhausted at
  round 4, reproducibly, because the collapsed state's spread LP is degenerate
  (`mean_m = -0.0000`).
* **Power mode invalidates timings.** Re-measure anchors after any change;
  a mid-run switch corrupts the run that straddles it.
* **`time_limit` is not a reliable guard at d=64** — exceeded by >3.5 min and
  still running when killed.
* **Concurrent stride runs race on `window_checkpoints.npz`** (each checks
  `if not os.path.exists(CKPT_PATH)` then trains and writes). Harmless for the
  stride phase, which reads `curve_checkpoints`/`edge_state`, but pre-build
  checkpoints serially before fanning out, or a reader can hit a partial write.
* **`--out` must be applied after `refresh_paths()`** or the cell-derived path
  clobbers it and parallel runs collide on one JSON. Correct in-tree; preserve
  the ordering.
* Solver settings (`ipm_tol`, `k0`) are **not recorded in the result JSONs**, so
  it is unrecoverable whether the published §20/§21 runs used `--ipm-tol`.
  Worth fixing: persist the solver config next to every result.

## Open, in priority order

*(Items 1-3 below are done as of 2026-07-28; kept for the record with their
outcomes. 4 and 5 are the live ones.)*

1. ~~Finish the seed-43 runs and complete the replication table.~~ **Done.**
   Both ep100 arms ran their full 120 rounds and both crossed; the table above
   is regenerated by `make_replication_table.py`. The one arm still short of the
   seed-43 protocol is seed 44's ep50 (3 rounds, 1 ascent vs 44 rounds, 12
   ascents); a matched run is in flight.
2. ~~Confirm the baseline gate.~~ **Done** — sigma90 ratio 1.0000 on all four
   gates under the patched code.
3. ~~§21 oracle arms at seed 43.~~ **Done** — `flippolicy_s43_fw.json`,
   `_fwfull02`, `_fwfull5pct`; all three rows reproduce.
4. **d=64**: needs a reformulation, not tuning. The pattern rows are 84% of the
   d=64 LP's rows but only ~13% of its nnz, so row reduction will not help
   (measured at d=32). Either find a formulation with fewer *embedding
   variables*, or accept a multi-day run with a working time guard.
5. The random-flip vs near-tie-flip experiment from §7 above.
