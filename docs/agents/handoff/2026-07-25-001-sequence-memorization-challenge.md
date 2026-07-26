# Handoff — hand-coded weights for sequence memorization

**Date:** 2026-07-25
**Repo:** `/Users/eppie/claude_projects/handcode` (not a git repo)
**Subject:** Reproduction of, and an entry to, [Linsefors & Bushnaq, *Challenge: Hand coding
weights for efficient sequence memorisation*](https://www.lesswrong.com/posts/KWtchKwwnJkd4bwCi/challenge-hand-coding-weights-for-efficient-sequence-1)

---

## What was accomplished

**1. Faithful reproduction of the post's Figure 5.** All four of their conditions across
d ∈ {16,32,64,128}, plus `hand-coded` at d=256. Fitted curves land within 0.95–1.05× of
their published ones per point:

| condition | ours | published |
|---|---|---|
| trained, acc≥0.9 | 8.49·d^2.00/ln d | 9.42·d^1.97/ln d |
| hand-coded, acc≥0.9 | 1.17·d^1.91/ln d | 1.16·d^1.93/ln d |
| hand-coded, acc=1 | 2.23·d^1.50/ln d | 2.17·d^1.55/ln d |

Verified against the authors' own source (fetched into `reference/`): fact generation is
byte-identical; construction weights are bit-identical where the frequency ranking is
unambiguous; where it ties, both pick tokens of the same frequencies and the accuracy
distributions agree to ~1 SE. Tests in `tests/test_matches_reference.py`.

**2. Three new constructions** (all rule-legal, no gradient descent):

| condition | idea | acc≥0.9 at d=64 |
|---|---|---|
| `hc-tiebreak` | their embedding + a bounded ridge tie-breaker | 1008 |
| `coin-tiebreak` | active-pattern AND-rectangle embedding | 928 |
| `linsolve` | **value code solved as a linear system** | **8704** |
| (their hand-coded) | | 768 |
| (trained) | | 8320 |

`linsolve` stores 6–13× their construction and overtakes the trained model above d≈60
(1.26× at d=128, verified in both float32 and float64).

**3. A mechanism finding** (`FINDINGS.md`) — the part most relevant to the authors' actual
goal. Near capacity the trained model does **not** use a pattern code:

| load | binary/full (d=64) |
|---|---|
| 7% | 1.00 |
| 41% | 0.25 |
| 82% | 0.11 |

At low load, binarizing the activations costs nothing. At capacity, the pattern alone
retains ~10% and ~32 magnitude levels are needed. The transition is universal in
load-fraction across d=32/64/128. Density stays at 0.53–0.60 of `d` throughout.

**4. A ~9× solver speedup** (`handcode/fastsolve.py`) with identical results, and a runner
that splits BLAS threads across concurrently-running jobs.

---

## Answer to "what next, and how do you beat the prefactor/exponent while meeting their goals?"

`linsolve` beats the *exponent* (2.24 vs 2.00) but not the *prefactor* (3.11 vs 8.49), so
the curves cross at d≈60 rather than coinciding. It also fails their research goal: its
weights reach 2.5e6 where trained weights max out at **6.9**, and it stores facts in the
low-order bits of large numbers, which makes it precision-limited (float32 dies at d≈290).

The measurements now specify what a construction must look like to do both. **Target
spec, all measured on trained models at capacity, d=64:**

| property | trained | linsolve | implication |
|---|---|---|---|
| density | **0.53** | 0.11 | too sparse; need ~half the neurons on |
| coding | magnitude, ~5 bits/neuron | magnitude | ✓ right family |
| max abs weight | **6.5 / 6.9** | 7.3e3 / 2.5e6 | drop the `T0` mega-offset |
| params carrying facts | all 5d² | 2d² (second embedding only) | **this is the prefactor gap** |

### The concrete proposal

The prefactor gap is a degrees-of-freedom gap. `linsolve` puts all fact information in
one embedding (`(d-1)·n_input_vocab ≈ 2d²` values, one equation per fact), so it is capped
at `2d²·(1/0.9)`. Trained models use all `5d²`. At d=16 that is 533 vs the 760 needed —
the shortfall is structural, not tuning.

To use all the parameters, both embeddings must carry values, which requires a gate
depending on both tokens. **This was tried and failed — but the failure mode is now
diagnosed, and it is fixable.** The two-sided AND gate `h_i = 1[m_i[a] ∧ n_i[b]]·(p_i[a] +
q_i[b])` requires `p+q > 0` on gated triples for its hard decode to work, and that forces
gate density `ρ² > 0.75`. **The trained model sits at density 0.53, i.e. `ρ = 0.73` —
below the threshold my construction could reach.**

The reason it needs the threshold at all is the *hard decode*: `linsolve` insists the
activation sum land within ±½ of an integer, so every gated neuron must fire exactly as
planned. The trained model has no such constraint — it lets the ReLU zero whatever it
zeroes and the readout adapts.

So: **drop the exact-gate requirement and alternate two closed-form solves.**

```
1. init two-sided graded embeddings at density ~0.53 (rho ~ 0.73), small weights
2. H = relu(u[a] + v[b])                       <- whatever it actually is
3. solve readout W closed-form (ridge) on H     <- d^2 params
4. fix the activation pattern from current weights; solve (u, v) closed-form
   for the now-linear system given W            <- 4d^2 params
5. repeat 3-4 for 2-4 rounds
```

Each step is a closed-form linear solve, which the rules permit ("closed-form
computations, greedy algorithms ... a ridge regression would be allowed"). Keep the round
count small and fixed, and document it — many rounds of alternation shades toward the
"generic black-box optimizer" the rules forbid, and a referee could reasonably object.

Why this should close the prefactor gap: it uses `5d²` parameters instead of `2d²`, which
at d=16 raises the ceiling from 533 to ~1100 against the 760 needed. Why it should satisfy
their goal: density, coding scheme, and weight scale all match trained models by
construction, and dropping `T0` removes the precision wall.

### Secondary directions

* **E3 (unrun, cheap):** is the trained `W_U` closer to `HᵀY` (prototype) or `(HᵀH)⁻¹HᵀY`
  (whitened)? I showed a prototype readout provably cannot break the ties their
  construction leaves while a whitened one can; knowing which one gradient descent picks
  is directly informative.
* **E4 (unrun, cheap):** measure neurons-per-label and labels-per-neuron in *trained*
  models. Their Appendix D reports best `S ≈ √d` for the hand-coded model and explicitly
  says it may say nothing about trained ones.
* **Report the load-fraction law to the authors.** The pattern→magnitude transition is a
  new empirical regularity about their own setup, and it reframes their Figure 8.
* Replicate the transition with more seeds (currently one per cell; trend is monotone
  across three sizes but individual numbers are single measurements).

---

## Key decisions and things ruled out

**Ruled out with derivations, do not re-litigate:**

* **Constant-gate two-sided designs cannot beat `2d²`.** Constant intersection means
  `M Nᵀ = r·J`, which is rank 1, so valid masks solve `N·1_{m[a]} = r·1` and live in a
  space of dimension `nv − rank(N)`. Mask diversity trades against `rank(N)` one-for-one;
  the total caps back at `V·nv = 2d²`. Hash-based designs collapse to exactly this.
* **Fluctuating-gate two-sided caps at ≈2.48d²** given the positivity constraint
  (`rank = 2Vρ·nv − nv²` at `ρ = 0.866`), matching the ~1.2× measured. The proposal above
  escapes this by removing the positivity constraint, not by beating the bound.
* **Projection methods lose to the greedy drop.** Kaczmarz to exact targets restores an
  inconsistent equality system and cycles; POCS to interval edges is still worse, because
  least squares optimizes L2 while the metric is L0. This is *why* the greedy drop works.
* **A Hebbian/prototype tie-breaker resolves ties at chance** — on a tied fact `h` is
  already zero on every tied label's neurons, so `⟨h, μ_c⟩` collapses to the same value.
  Whitening is the entire difference; Hebbian is ridge's λ→∞ limit.
* **Graded values inside a rectangle hurt monotonically** — a real fact and a cross-pair
  are not additively separable, the post's `x,z→l; p,q→l; x,q→not l` obstruction.
* **Per-neuron readout gains** don't help; conditioning was not the binding constraint.

**Deliberate choices:**

* 3 seeds per cell, not the post's 11. Costs the `trained` baseline ~20% at d=16 (496 vs
  600); from d=32 up the two agree exactly.
* `hybrid` at d=128 was killed mid-run (5–10h, not needed for any claim). It is the one
  gap in `results/scaling.json`.
* Gradient descent **is** used for analysis probes in `FINDINGS.md`. That is legitimate —
  the no-GD rule constrains challenge entries, not measurement.

---

## Context a fresh agent needs

**Layout**

```
handcode/data.py        fact generation (verbatim from the post)
handcode/model.py       Figure-4 architecture + trained baselines
handcode/connection.py  S-neurons-per-label design (greedy + SA + balance)
handcode/handcoded.py   the authors' silence construction
handcode/readouts.py    closed-form unembeddings (ridge, bounded tie-breaker)
handcode/coincidence.py active-pattern AND-rectangle embedding
handcode/linsolve.py    the value code (reference implementation)
handcode/fastsolve.py   batched solver, ~9x, used by the capacity search
handcode/capacity.py    binary search, sweeps, scaling fit
reference/              the authors' fetched source
reference_post.md       full post text (byte-identical to live as of 2026-07-25)
results/scaling.json    all capacity measurements (resumable)
FINDINGS.md             the mechanism work
```

**Commands**

```bash
uv run pytest                                          # 38 tests
uv run python precompute_conn.py --ds 16 32 64 128     # ~1 min, cached
uv run python run_scaling.py --ds 16 32 64 128         # resumable; skips done cells
uv run python run_scaling.py --ds ... --report-only    # re-print tables
uv run python plot.py                                  # results/scaling.png
```

**Traps that cost time this session**

* `model.train()` **clones its inputs and returns only the accuracy** — the trained weights
  are discarded. Any analysis of trained models needs a local loop that keeps them. This
  silently produced a whole table of random-init results before it was caught.
* Ridge's primal and dual forms are equal in exact arithmetic but **not** in floating
  point with tiny `mu`; the choice must be made *per token*, not globally from the padded
  width. Getting this wrong is silent and large (~1e2 against a ±½ tolerance).
* Weight-level comparisons between solvers are meaningless where a token's system is
  under-determined — there is a null space, and solvers land in different places.
  **Compare decoded sums, not weights.**
* `pkill -f run_scaling.py` kills the parent but orphans its `multiprocessing.spawn`
  children, which keep burning CPU. Kill children first.
* `run_scaling.py` writes `results/scaling.json` by merging with what is on disk, so two
  concurrent runs no longer clobber each other — but anything started before that fix did.

**Open verification items**

* d=256 `linsolve` acc≥0.9 (126976) is **float64-verified only**; float32 came in at 0.8979
  under a reduced sweep. Flagged with a dagger in the README.
* `trained` at d=256 was never measured (hours per point). The 13.1× ratio at d=256 is
  against their `hand-coded`, not against trained.
