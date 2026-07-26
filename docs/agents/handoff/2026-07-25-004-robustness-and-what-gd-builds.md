# Handoff — the robustness reframe, and most of a constructive account of trained storage

**Date:** 2026-07-25
**Repo:** `/Users/eppie/claude_projects/handcode` → published at
<https://github.com/Eppie/sequence-memorization-challenge> (public, `main`).
**Local state: a full session of work is UNCOMMITTED** — new probes, new module, new
docs, one `git mv` staged (handoff 001 renamed). The user has not asked for a commit.
**Supersedes:** `2026-07-25-003-published-and-why-gd-fails.md` — its open questions
(the noise experiment, why GD falls short) are now answered. Its "ruled out" list and
002's remain authoritative for `twosided.py` internals.

---

## What this session established

**1. The capacity headline was audited and survives, but the benchmark itself broke.**
The audit (protocol vs. `reference/`, spot-checks at d=16, fact seeds 43/44) confirmed
everything is like-for-like and reproducible; ratios are robust to fact seed, exact
integers are seed-42-specific. Then `probe_robustness.py` measured the axis the metric
ignores: **at matched load a trained model tolerates ~1300× more relative weight noise
than the value codes** (σ90 2.0e-2 vs 1.5e-5 at d=32 n=2080; ≥1200× at d=64). In
absolute units the construction dies at ~1/9 of one Adam step; trained solutions carry
5–20 steps of cushion. **The authors' own construction is within ~2× of trained
robustness** — their instinct about exact solves is now a number. Figure:
`results/robustness.png`. Recommended metric fix (in docs): report σ90 next to
max_facts, or score under noise ≈ one optimizer step.

**2. Why gradient descent falls short — fully decomposed** (`FINDINGS.md` §5–7):

* *Walk-away is arithmetic*: one optimizer step exceeds the construction's noise
  budget ~9×. Casualties are margin-random (rank corr 0.05, `probe_walkaway.py`) —
  interference returns globally, not selectively.
* *The capacity gap is the optimizer, not the objective*: fix the readout to the
  quadratic decode (kills the scale loophole that made margin caps untestable) and
  Adam still reaches 2080 in full — **the unembedding's d² parameters are an
  optimization aid, not storage** — then stalls at the converged-GD ceiling (0.86 at
  2560 after 200k epochs, 0.39 flat at 3168). Under the post's patience-100 recipe the
  fixed-decode model gets 4%: the free readout buys *trainability*.
* *No standard optimizer does better* (`probe_optimizers.py`, §10): converged Adam >
  SGD+momentum > L-BFGS at every load. (torch L-BFGS is a weak second-order stand-in —
  K-FAC/Gauss-Newton untried.)

**3. Most of a constructive account of what GD builds** (`docs/what-gd-builds.md` —
the flagship doc):

* *Structure measurements* (`probe_structure.py`, §8): trained per-fact weight-space
  radii have a high floor that continued training raises 4.4× faster than the median
  (implicit-bias/max-margin signature); the trained readout is **high-rank
  (rank90 ≈ d/2·…: 12 at d=32, 24 at d=64) and ladder-free** (graded fraction 0.03).
  The exact solve's radii are uniformly thin (CV 0.13).
* *Digit-code family* (`handcode/digitcode.py`, `probe_digitcode.py`, §9): label
  digits carried by neuron groups, m equations/fact, ridge-only; at m=1 it IS
  twosided. Redundancy alone buys only ~20× robustness. The dominant fragility is the
  **pedestal** (t0 inflates ‖h‖; readout noise scales with it); shrinking t0 to ~d/2
  buys another ~10× and drops weights to trained scale (`probe_pedestal.py`). Fully
  optimized: **n=1584 (76% of trained) at σ90 1.3e-3 — residual 30× from trained.**
* *The residual is inequality packing*, and the max-margin LP (`probe_maxmargin.py`,
  scipy/HiGHS added to deps) tests it: at d=16 the max-min-margin solution of the
  trained model's own frozen (pattern, readout) **reproduces trained robustness**
  (σ90 4.6e-2 vs 4.8e-2), and both mixed conditions are **infeasible** — random
  codebook can't decode the trained pattern, trained readout can't separate a random
  pattern. **Capacity lives in the joint adaptation of gate and codebook.**
  d=32 replication was running at session end (see below).

**4. The authors' appendix-B mystery answered** (`probe_badcombo.py`, §11): the
MLP+Norms+NoRes+NoBias+ReLU deficit is real, budget-insensitive (12× epochs moves
0.68→0.71 at n=768), not dead units at scoring (but whole-fact death *grows* with
training, 1.8%→12% — a ratchet). Every one-setting flip restores capacity by adding a
slack mechanism (GELU leak / bias threshold / residual bypass / free scale). Answer:
training dynamics of a structural kind budget does not cure.

**5. Deliverables written**: `docs/reply-to-post.md` (draft comment for the authors —
ready for user review), `docs/what-gd-builds.md`, FINDINGS §5–11, README updated
(honest counterweight + file inventory), `tests/test_digitcode.py` (57 tests pass).
Everything converted to **American English** except `reference/`, `reference_post.md`,
and the authors' quoted title (user asked; chat responses also American — see user
memory).

---

## Traps and decisions this session added

* **HiGHS simplex thrashes on the max-min-margin LP at d=32** (80+ CPU-min, no
  solution; degenerate optimal face). `method="highs-ipm"` solves d=16 in 1.5s.
  IPM returns a central (non-vertex) point, so its γ is conservative; the
  cutting-plane wrapper (`solve_with_cuts`) reports the TRUE min margin of the
  returned point, so truncated constraint sets can't overstate. k0=8 confusable
  classes + verification round is the working recipe.
* **`| tail` buffers background output** — pipe to a file instead
  (`> log 2>&1; tail log`), or nothing shows until exit.
* The digit-code convergence needed two things twosided didn't: `sweeps=8` and, for
  m>1, the label carried *inside* each group's own sum (signed random projections do
  NOT converge — that was the first, failed design; base-p digit codebooks on neuron
  groups are the design that works).
* Trained margins depend on when you stop: ~3.3 at first acc=1, 7.77 after the full
  5000-epoch budget. The robustness baselines use first-acc=1 (conservative).
  `probe_reachability.run_adam` runs the full budget with no early stop.
* d=64 structure numbers (rank90=24, radii) were measured ad hoc and are cited in
  `what-gd-builds.md` but not persisted in `structure.json` — regenerate via
  `probe_structure.py` logic at d=64 if needed.

## Open at session end

* ~~The d=32 max-margin run~~ **Completed and folded in.** d=32 confirms d=16:
  σ90(LP) 1.57e-2 vs trained 1.62e-2; γ* = 9.40 vs trained min margin 5.84 (trained
  sits at ~62% of the margin optimum at ~100% of the robustness optimum — robustness
  saturates first); all mixed and scratch conditions infeasible, including random
  geometry at n=992. One methods note: the readout-similarity confusability hint fails
  at d=32 (3 cut rounds leave violations); hinting with the trained model's own
  logits (k0=12) converges in one round, 36s. `results/maxmargin.json` holds the
  refined numbers; `probe_maxmargin.py`'s built-in hint is still readout-similarity —
  worth upgrading to accept a logits hint if rerun.
* **The construction problem is now "gate discovery" and nothing else** (FINDINGS
  §13, run after this handoff was first written). Ridge-bootstrap coordinate ascent
  dies (ridge readouts support zero margin even on the trained gate); two-LP
  max-margin ascent from a feasible start raises the digit gate's min margin 1000×
  while σ90 moves 1.3e-3 → 1.7e-3 only. Same LP machinery on the trained gate hits
  trained σ90. So: characterize what makes the trained gate good (it drifts 41% of
  bits from init, slowly; candidate metrics: per-token design conditioning,
  active-set decorrelation among facts sharing a token) and construct gates with that
  property. Engineering notes: carry LP cut sets across ascent rounds (rebuilding
  them thrashes, especially under rank-2 readouts); the d=64 reconstruct LP is
  compute-bound at ~550k rows — add cutting planes on pattern-consistency rows too.
* Ideas ranked next: gate-quality metric + gate construction (above); exact
  Gauss-Newton/K-FAC to complete the optimizer story; the quantization-vs-d law
  (bits/neuron ≈ log d?); their best-S ≈ √d question.
* All committed and pushed through the geometry-ascent results;
  `docs/reply-to-post.md` awaits user review before posting.
