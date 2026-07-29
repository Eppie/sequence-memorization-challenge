# Handoff — the ceiling is a ten-kilobit co-adaptation; forward design goes 0-for-7

**Date:** 2026-07-28
**Repo:** `/Users/eppie/claude_projects/handcode`, `main`, all committed and
pushed. 69 tests pass.
**Extends:** `2026-07-27-001-seed-falls-ordering-refuted.md` (the seed/flow
arc) and `2026-07-27-001-replication-guard.md` (whose draft this session
folded into `FINDINGS.md` as §24).

## What this session established

1. **Rate–distortion of the trained gate** (`probe_ratedistortion.py`,
   FINDINGS §25). Compress the pre-activation embeddings, hand the induced
   pattern to §14's ascent, measure the ceiling against description length.
   Per-parameter quantization keeps almost everything: the feasibility edge is
   between 1.0 and 1.6 bits/param, a three-level alphabet (~6.5 kb, 20×
   compression) names a pattern **above the construction record** (4.28e-3 vs
   2.85e-3), and 3 bits/param recovers 96% of the trained ceiling. Every
   structural compression — SVD rank to 24/32, k-means token codebooks to
   64/128, sparsity at half — is infeasible at matched or lower flip
   fractions. Perturbation *direction* is what matters, not flip count:
   nearest-zero flips tolerate 8% (3.19e-2) and die by 16%; random flips die
   at 0.5%. Net: ~4,096 scores × ~2.5 irreplaceable bits ≈ 10 kb of
   co-adaptation with no cross-parameter redundancy found.
2. **Forward-designed gates all fail** (`probe_designedgates.py`, FINDINGS
   §26). Designing in score space makes realizability free (Theorem 2), so
   storability is the only question. Hadamard signatures, modular-hash
   staircases, thermometer digit codes (the inequality-native digit code the
   rank barrier does not touch): all γ\*=0. The control carries it: the
   trained gate's own per-neuron score columns with tokens permuted — every
   first-order statistic matched — is infeasible at both seeds. 6‴ now has a
   fourth leg (design-space) and a measured size; `docs/theory.md` updated.
3. **Housekeeping:** the replication draft became FINDINGS §24 (renumbered
   from its colliding "§22"); the two "replication guard gates publication"
   caveats in §§21–22 now point at §24; the reply draft's STATUS reflects the
   landed guard (posting remains the maintainer's call) and its
   "process-shaped" paragraph carries the fourth leg and the 10 kb number.

## Traps (beyond the standing ones)

* **A first-round emb-LP timeout records as infeasible with nothing proven.**
  `linear_k6` was mis-recorded this way under 3 concurrent LP processes
  (HiGHS `time_limit` is a literal 300s in `probe_geometry_ascent.py`);
  re-run alone with the limit raised it is feasible at 3.27e-2. Negative
  verdicts are trustworthy only when round 0 completed with γ\*=0 — check
  `n_lp_steps >= 1` in the result JSON. `probe_ratedistortion.py
  --lp-time-limit N` raises the limit without touching the solver source
  (patches `scipy.optimize.linprog`; the probe imports it function-locally).
* IPM is slowest exactly on the near-edge instances, so timeouts cluster on
  the scientifically interesting cells. Keep concurrency ≤2 for near-edge
  sweeps, or raise the limit.
* Designed gates with density much below 0.5 can leave facts with empty
  active sets — trivially unstorable before any LP. Pure-AND Hadamard leaves
  44; only one of four mixed variants avoids empties at every column roll.
  `probe_designedgates.py` prints `empty_facts` per config; check it before
  spending an ascent.

## Open, in priority order

1. **Co-adapted declarative design** — the one door §26 leaves open: scores
   chosen as a *function of the specific fact table* (not fact-blind
   families). §25's edge (~6.5–8.2 kb) sitting just above the fact table's
   own ~7.9 kb of label content suggests a counting argument for why
   fact-blind design must fail; making that argument is the 6‴ lower bound in
   embryo.
2. **Replication of §§22–23 and §§25–26 rows** at seed 43 (the probes take
   `--fact-seed`; the ridge seed builder + fw flow is ~3–4 h at d=32, the
   rate–distortion sweep ~2 h uncontended).
3. Finer quantizers (per-column, vector quantization on *score pairs*) to
   push the 1.0–1.6 bits/param edge down; each point is one ascent.
4. The reply draft is content-complete and gated only on the maintainer's
   decision to post.
