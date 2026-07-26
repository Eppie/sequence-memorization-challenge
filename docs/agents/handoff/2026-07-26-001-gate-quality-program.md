# Handoff — the gate-quality program: three refutations and what remains

**Date:** 2026-07-26
**Repo:** `/Users/eppie/claude_projects/handcode` → published at
<https://github.com/Eppie/sequence-memorization-challenge> (public, `main`).
**Local state: everything committed and pushed** through `33e4721`; tree clean, 57
tests pass.
**Supersedes:** the open-questions list of
`2026-07-25-004-robustness-and-what-gd-builds.md` — its gate program has now *run*
and the candidate metrics it proposed are refuted. Its traps and machinery notes
remain authoritative.

---

## What this session established

**The question "what makes a gate good?" was answered three ways, all negative, all
measured** (`probe_gatequality.py`, `FINDINGS.md` §14, results in
`results/gatequality.json`). Six-gate zoo at one load: d=32, n=1584 (the digit code's
capacity point; trained σ90 there is 4.5e-2). Gates: trained (full budget), trained
at first acc=1, twosided, digit m=2 (pedestal-optimized), density-matched random
additive, and the training run's own init.

1. **Not an intrinsic pattern statistic (phase `metrics`).** Handoff 004's candidates
   — per-token design conditioning and same-token active-set decorrelation — do not
   separate the trained gate from the *random* gate or from the trained run's own
   init (same-token corr 0.323 vs 0.334 vs 0.332; every additive gate sits at the
   ~1/3 a Gaussian calculation predicts; smin/κ likewise overlapping). Refuted, not
   just unsupported: `smin_p10` ranks random and init *above* trained. The 41%-of-bits
   drift is invisible to family statistics. Conditioning detects exactly one thing:
   pedestal degeneracy (twosided smin ≈ 0, κ ≈ 257; digit intermediate).

2. **Not compensable by readout freedom (phase `predict`).** The two-LP max-margin
   ascent (readout co-optimized — the fair test `probe_maxmargin`'s mixed conditions
   never gave, since they froze the readout) on every gate: trained → σ90 **4.40e-2**
   (its own model: 4.55e-2; LP margin 17.3 vs model's 7.6); digit → **2.85e-3** (the
   best-snapshot fix recovered more than 004's 1.7e-3); twosided → 1.41e-5 (its
   emb-LP still fails on the rank-2 ladder readout, HiGHS status 4 — known wart);
   random and init → **infeasible outright, acc 0.03**. So gate quality is
   *relational*, not combinatorial: whether the cone of embeddings realizing the
   pattern's signs contains high-margin value embeddings. Two gates with identical
   statistics sit at opposite extremes of storage capability.

3. **Not buildable by exact-solve imitation of GD (phase `drift`).** New
   construction: pattern drift under max-min-margin pressure — emb-LP with the
   sign-consistency rows dropped (`solve_max_margin(..., pattern_rows=False)`, new
   flag), step toward the LP point under twosided's 2% flip cap (order-statistic
   step, `capped_step`, unit-checked), re-read pattern, refit readout-LP, repeat.
   Seeded from the digit gate: **one step produces the best constructed gate to date
   (σ90 1.27e-3 → 2.96e-3, above the frozen-pattern ceiling), then iteration
   destroys it** — monotone σ90 decline over 8 rounds to 1.7e-4, accuracy collapse
   (0.73, 0.60) at 7.9% cumulative drift, masked-LP γ itself decaying to 0. The flip
   ledger is the diagnosis: every round spends its full 2% quota while cumulative
   drift advances ~0.6%/round — churn, not consolidation (GD: 0.5%/epoch compounding
   to 41% with accuracy intact, §12).

**Net state of the reduction:** values/readout/margins are LPs; the gate is
everything; gate quality is operationally defined (cone richness), located, and
fenced by three refutations; the best constructed gate stands at 2.96e-3, a measured
**~15×** from trained at matched load (was "~25–30×" in handoff 004 — the
best-snapshot and the one good drift step tightened it). Docs updated to match:
`FINDINGS.md` §14 (new), `docs/what-gd-builds.md` (status line, §3 quantitative
decomposition, §4 item 1 rewritten), README (probe listed in run/layout).

**Also this session:** a new comment on the post (Jesse Li, 2026-07-26 11:00 UTC)
proposes squeezing Appendix A into width d by silencing all tokens with id ≥ d−1 in
both positions → (d−1)² facts. Analysis (delivered in chat, not yet in any doc): the
idea is sound but 2× conservative — only *first*-position tokens need selectors, so
discarding only those reaches (d−1)·2d ≈ 2d², i.e. exactly the built-gate bound the
README derives; it is another member of the 2d² family (twosided exists to break it,
3168 vs 961 at d=32 acc=1). Robustness-wise it is a magnitude code but with *no
accumulation pedestal* (one selector neuron carries ℓ+1), so plausibly less fragile
than linsolve — measuring it for the frontier plot is a ~1-hour task if wanted.
Fetch method that worked: LessWrong GraphQL (`POST https://www.lesswrong.com/graphql`,
view `postCommentsTop`, postId `KWtchKwwnJkd4bwCi`); GreaterWrong 429'd, WebFetch of
the LW page doesn't render comments.

## Key decisions

* **One load for the whole zoo** (n=1584) so every σ90 is comparable and the trained
  reference (4.5e-2) is already known. Phase results merge into one JSON
  (`results/gatequality.json`) keyed `metrics` / `predict` / `drift`.
* **Gate cache** (`results/gatequality_gates/*.npz`, patterns + start weights +
  drift-best snapshot) is gitignored like `conn_cache` — deterministic from seeds,
  ~10 min to rebuild. The JSON carries all numbers.
* **`ascend_best`** snapshots the best-true-margin full model at every half-step; the
  ascent oscillates and the last state is often not the best (this alone improved the
  digit ceiling 1.7e-3 → 2.85e-3 over handoff 004's last-state number).
* **Drift changed exactly one thing at a time**: same flip cap, same no-memory cut
  handling, same readout-LP as the ascent; only the sign-consistency rows dropped.
  That keeps the collapse attributable to the pressure structure, not to a bundle of
  changes.
* γ = 0.0000 from the LPs is the degenerate "do-nothing" answer (W=0 / u=v=0 are
  always feasible), i.e. *infeasibility* of any positive margin — not a numerical
  quirk.

## Traps

* All of handoff 004's LP traps still apply (ipm not simplex; logits hints; the
  twosided ladder readout degenerates the cutting planes — it did again here).
* `probe_gatequality.py` phases assume the gate cache exists or is buildable;
  `--gates` subsets. Building `digit_m2` takes ~2–4 min (1500 rounds), `twosided`
  similar, the trained gates ~1–2 min each.
* The drift run writes its JSON entry and best-state npz **only at completion** —
  don't kill it mid-run expecting partial output (history is printed to the log
  either way).
* Foreground `sleep` is blocked in this environment; use `run_in_background` +
  Monitor with an until-loop.

## Open at session end — in priority order

1. **The spread-pressure drift experiment — designed, announced, NOT run** (the
   /handoff arrived first). This is the one experiment between the current state and
   a defensible closure of the constructive question. Spec: replace the drift's
   emb-LP objective with the capped-sum LP — per-fact auxiliary variables m_f,
   maximize Σ_f m_f subject to the same margin rows written against m_f instead of a
   shared γ, bounds m_f ≤ τ (free below), τ ≈ 0.5; everything else (2% flip cap,
   readout-LP refit, no memory) identical, so the outcome is attributable to
   pressure structure alone. ~5.7k variables at n=1584 — same solve class, ~3.5
   min/round, 30 rounds. If it also collapses → the process thesis ("the good gate
   exists only as the fixed point of a coupled dynamics; the final ~15× is
   irreducibly a process") has survived its natural alternative and is ready to
   write up. If it climbs → the missing ingredient was all-facts pressure, and the
   construction is live again. Either outcome is a strong §15.
2. **Replication guard:** §14 is d=32, n=1584, one seed throughout. A d=64 (or
   second-seed) spot check of the headline rows (trained vs random feasibility;
   drift collapse) before publishing claims from it.
3. **`docs/reply-to-post.md`** still awaits user review and now understates the
   result: its point (3) says residual ~30× — today's number is ~15× with the
   three-refutation story behind it. Fold in before posting (user's call).
4. **Jesse Li reply** (user's call): the 2×-improvement observation (discard
   first-position tokens only → 2d²) plus the built-gate-bound framing would be a
   good-faith thread contribution; optionally implement their construction for
   `results/frontier.png`.
5. Unchanged from 004: exact Gauss-Newton/K-FAC; quantization-vs-d law; best-S ≈ √d.
6. Housekeeping: `trained_early` was measured in `metrics` but never LP-ascended
   (expected ≈ trained; cheap to add for completeness).
