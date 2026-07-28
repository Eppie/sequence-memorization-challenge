# Handoff — the seed problem falls; the ordering invariant refuses to exist

**Date:** 2026-07-27
**Repo:** `/Users/eppie/claude_projects/handcode` → public at
<https://github.com/Eppie/sequence-memorization-challenge>, `main`.
**Local state:** committed and pushed through `3289da7`; 69 tests pass. Two
runs may still be in flight when you read this (see *In flight* below); their
scratch JSONs and logs are session-scratchpad files, their state npzs are in
the gate cache.
**Supersedes:** the seed/softness reading of `2026-07-26`'s work (FINDINGS §20)
— revised by §22, which this session added. FINDINGS §§22–23, `theory.md`
6′/6″/6‴, and `what-gd-builds.md` are current and authoritative.

## What this session established

1. **The seed problem (theory 6′) is resolved constructively** (FINDINGS §22).
   `handcode/softseed.py` builds ridge-only seeds (freeze-pattern rounds:
   readout ridge to one-hot + per-token embedding sweeps + flip-capped step).
   Fit plateaus at 0.63; no ridge knob builds a near-tie reservoir (all configs
   sit at the Gaussian-init ratio 6–9e-3). Two post-passes: `soften`
   (per-column ridge compressing the smallest-|pre| band; sign-preserving,
   hits GD's 1.1e-3 ratio on demand) and `w_rms` (rescale readout to GD's rms
   1.28 — argmax-invariant but changes the τ-saturation regime).
2. **The `w_rms`/τ regime was §20's real gatekeeper.** The raw ridge readout is
   19× under GD scale ⇒ zero facts above τ = 0.5 ⇒ fit facts never saturate ⇒
   §14 churn. Without it every seed collapses under fw-full in one round; with
   it every seed climbs. Softness is a second-order stabilizer only
   (longer feasible windows under fw-full; twins identical under fw).
3. **The fw flow (matvec direction + exact readout refit) consolidates from
   constructed seeds**: crossing at round 25 at 1.45e-2 (= §19's GD crossing
   value), monotone to 3.00e-2/3.21e-2 (stiff/softened) at round 400, **no
   plateau**, past the GD-seeded fw numbers by round 250–300, ≤1.4× from the
   trained ceiling 4.40e-2. The 0.52-fit seeds also cross (round ~100,
   1.8–1.9e-2 by 200). Ledger: best GD-free artifact 3.15e-3 → **3.21e-2**;
   construction record unchanged at 2.85e-3 (the flow optimizes task margins =
   training; process-class analysis, not an entry).
4. **The ordering invariant (6″) is first-pass refuted** (FINDINGS §23,
   `probe_ordering.py`): ten rank-space statistics + the per-fact
   near-boundary-exposure joint statistic (all zoo gates match the
   independence null) fail on the matched pairs (ridge seed vs its own fw
   product; ep180 vs trained). Provenance-detecting statistics exist; the
   seed row kills them as quality metrics. The seed→product delta reproduces
   §17's profile (boundary cells, uniform, error-agnostic). 6‴
   (incompressibility) now has three legs.
5. **`what-gd-builds` item 2 (codebook targets) closed on theory grounds:**
   an equality code with m channels ⇒ rank-m readout ⇒ capacity ≤ 4d²/m; the
   trained readout's rank (~d/2) is bought with inequality slack, which only
   the process class spends. No build needed.

## In flight at handoff time

* **Scratch cell** (task `bjvqtk60d`): seed tag `mu0.01-lam0.1-r0-c0.02-w1.28`
  (random init, ridge W at GD scale, acc 0.126, zero embedding fit) through
  fw-full (400r filter) then fw (200r + ascents). If it crosses, §20's scratch
  failure was also the readout regime and the flow needs *no* seed at all —
  §22 gets an addendum either way.
* **r800 resume** (task `bod9v2flj`): softened twin fw continued 400→800 to
  chase the plateau. If it approaches ~4e-2, the residual "integrand edge" is
  fully iteration budget.

## Key decisions

* **Legality line, stated in §22 and used consistently:** iterating solves
  against a coder-declared equality system (twosided/digit) is construction;
  iterating them against the task's own margins is training. All flows stay
  process-class analysis regardless of how GD-free they are.
* Seeds are cached as `softseed_<tag>.npz` with self-describing tags
  (`mu…-lam…-r…-c…[-s…][-w…]`); stride runs reference `--stride-seed
  soft:<tag>`. The sweep accumulates per-tag in `results/flippolicy.json`
  under `softseed` (merge-friendly across invocations).
* Flow states worth keeping are copied `r200_*/r400_*` in the gate cache
  before any `--resume` (resume overwrites the live state npz).
* `probe_ordering.py` pins `FW_STATE` to the preserved r200 product so §23's
  numbers stay reproducible regardless of later resumes.

## Traps

* **The machine is shared with the replication agent**
  (`.claude/worktrees/replication-guard`, long-lived HiGHS jobs; its
  `--fact-seed` flag exists only in its worktree copy). More than ~5–6
  concurrent LP processes total collapses throughput (measured: 15 s/round →
  3–4 min/round at 8). `pgrep -fl probe_flippolicy` before launching; chain
  sequentially.
* A killed-then-relaunched `--resume` run **appends duplicate round entries**
  to the JSON history (state resumes from the last snapshot, history from the
  file). Dedupe by round (keep last) when merging — the merge script in this
  session's history under `results/flippolicy.json` did this.
* zsh does not word-split `$var` — passing `"--flag value"` via a loop
  variable sends one argument and argparse dies silently inside a pipe to
  grep. Expand explicitly or use `${=var}`.
* `--fw-tw` default 0.05 is §21's known demolition setting; runs use 0.002.
* The softseed CLI grid rebuilds seeds unconditionally (no cache reuse) —
  deterministic, cheap (~0.2 s each at d=32).

## Open at handoff — in priority order

1. Read out the two in-flight runs; add the scratch-cell verdict and the r800
   number to §22 (a sentence or two each) and commit.
2. **Replication now covers §§22–23 too**: the seed builder + fw flow at a
   second fact seed / d=64 belongs on the replication agent's list (its
   current scope is §§14–21 rows). Coordinate before posting anything.
3. `docs/reply-to-post.md` is now two stories behind; rewrite after
   replication lands (user's call to post).
4. The declarative program's only remaining line: higher-order invariants
   nothing first-pass sees, or the 6‴ lower bound. The best available test
   pair (seed vs product, same ancestry, 7.6% apart) is cached for whoever
   tries next.
5. If the r800 run keeps climbing: run the flow to its actual plateau and/or
   larger round budgets from the 0.52 seed — the "GD ceiling = process ceiling
   + budget" hypothesis is now live and quantitative.
