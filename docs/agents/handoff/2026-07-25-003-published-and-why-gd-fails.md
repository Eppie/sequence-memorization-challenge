# Handoff — final results, publication, and the best account of why GD falls short

**Date:** 2026-07-25
**Repo:** `/Users/eppie/claude_projects/handcode` → **published** at
<https://github.com/Eppie/sequence-memorization-challenge> (public, `main`, in sync
with `origin`)
**Supersedes:** `2026-07-25-002-two-sided-value-code.md` — that doc was written mid-session
before the last measurements landed. Its numbers have since been patched to final, and its
"ruled out — do not re-litigate" list is still the authoritative one. **Read 002 before
changing anything in `twosided.py`.**

---

## Final state

Everything the previous handoff set out to do is done, measured, documented and published.
All 8 `twosided` capacity cells completed; `results/scaling.json` is current; 53 tests pass.

| max facts | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|
| their hand-coded, acc=1 | 53 | 130 | 216 | 776 |
| trained, acc=1 | 496 | 2080 | 7296 | 25088 |
| **twosided, acc=1** | **696** | **3168** | **12800** | **51200** |
| ratio to trained | 1.40× | 1.52× | 1.75× | **2.04×** |
| trained, acc≥0.9 | 760 | 2528 | 8320 | 27648 |
| **twosided, acc≥0.9** | **928** | **3904** | **15872** | **63488** |
| ratio to trained | 1.22× | 1.54× | 1.91× | **2.30×** |
| whole fact space `4d²` | 1024 | 4096 | 16384 | 65536 |

At acc≥0.9 the construction stores **90–97% of every fact that exists** at every size.

Plain power laws on the complete data (more legible than the post's `a·d^b/ln d`, in which a
pure `C·d²` law fits as `b ≈ 2.28` — so `twosided`'s `b = 2.30` is **not** super-quadratic
and must not be reported as such):

| | `p`, acc=1 | `p`, acc≥0.9 |
|---|---|---|
| their hand-coded | 1.23 | 1.64 |
| trained | **1.88** | **1.73** |
| linsolve | 2.01 | 2.02 |
| twosided | 2.06 | 2.03 |

The value codes are clean `d²` laws; the trained model is measurably sub-quadratic. The
widening ratio is an exponent difference, not a constant — falsifiable at d=256.

---

## What changed since handoff 002

* **d=128 acc≥0.9 = 63488** (of a possible 65536) and the **d=64 long-training check**
  landed. The latter matters: it *narrowed* a caveat 002 had stated too broadly. The post's
  ≤5000-epoch recipe under-trains at d=32 (converged GD reaches ~2560 not 2080, so that
  ratio is ~1.24× not 1.52×) but **is already converged at d=64** — 7296 is reached by epoch
  ~1600 at any budget and 9600 plateaus at 0.78 with 12× the epochs. So only the d=16/d=32
  acc=1 points are budget-inflated; **d=64's 1.75× stands against converged gradient
  descent**. Raw logs kept in `results/longtrain.log`, `results/longtrain64.log`.
* **Published to GitHub**, public, at the user's explicit direction. Before pushing I added
  `reference/README.md` attributing `reference/` and `reference_post.md` to Linsefors &
  Bushnaq with the source URL, since publishing redistributes their work and the repo
  previously carried only a one-line note. It offers to swap the vendored copy for a fetch
  script if either author objects.
* **`results/conn_cache/` is gitignored** (29 MB, regenerable in ~1 min via
  `precompute_conn.py`). So is `.pytest_cache/` and `reference/__pycache__/`.

---

## The current best account of *why* gradient descent falls short

This was the session's closing discussion and is **not yet written into the repo** — the
strongest remaining piece of work is to test it and fold it into
`docs/twosided-construction.md` §8.

**Established:** the construction is not a stationary point of cross-entropy (Adam placed on
it trades 17% of the facts for 18% less loss); at d=64 the baseline is converged, so this is
not a budget story; both are magnitude codes at capacity, so the old pattern-vs-value framing
is not the difference; margins at 100% accuracy are 0.42 (construction) vs 7.77 (trained).

**The trap to avoid — 002 records this and it bears repeating.** "Cross-entropy wants margin"
cannot on its own explain lower capacity, because **margin is free**: scaling `W_U` by 10
gives 10× the margin, identical accuracy and lower loss, costing no facts. That scale
invariance is also exactly why the natural experiment is unfalsifiable, and why both attempts
(hinge, CE on `β·logits`) were confounded.

**Best current hypothesis: cross-talk, with margin as its visible shadow.** Each fact's logit
is a sum over ~half the neurons, and other facts contribute interference. The construction
drives interference to *exactly zero* — that is what solving the linear system means — so an
arbitrarily thin margin suffices. A trained model retains residual cross-talk, so it needs a
real margin to sit above the noise, and the fact count it can hold with signal > noise is
correspondingly lower. This explains both halves at once: Adam *leaves* because any step of
size ~`lr` reintroduces interference far exceeding a 0.42 tolerance, and it cannot climb back
because climbing back means exactly solving a `4d²`-variable linear system; and Adam never
*reaches* comparable capacity because first-order methods solve ill-conditioned linear
systems slowly, while this algorithm is Newton's method on the same equations.

**The wrinkle that argues against pure slow-convergence:** at d=64 the trained model
*plateaus* (0.78 at 9600 facts, flat from 5k to 60k epochs) rather than crawling upward.
That suggests GD settles into a genuinely different solution type — distributed, redundant,
noise-tolerant — rather than approaching this one slowly.

### The prediction worth testing first

**The benchmark has no robustness requirement, so it is maximised by maximally fragile
codes.** This construction is correct by 0.42 out of a ±500 logit range; a trained model by
7.77. Predicted: adding Gaussian weight noise should shatter the construction while barely
moving a trained model at the same load.

If that holds, part of the hand-coded/trained gap is not the trained model being *worse* —
it is the trained model solving a harder problem (store facts *robustly*) that the metric
does not score. That would make "close the gap to trained" partly the wrong target and
argue for a noise-robustness term in the metric. It is cheap to run and it is the single
most valuable next experiment, both for the repo and as something to say to the authors.

### Ranked ideas for making a trained model match

1. **Fix the readout to the quadratic decode, train only the embeddings.** The
   construction's `W_U` is essentially rank-2 (labels ⊗ ones, plus the bias column) and GD
   has no reason to find a low-rank readout. This is the post's `hybrid` with the roles
   reversed, and the best-conditioned version of the problem.
2. **Remove the scaling escape, then cap the margin** — constrain `‖W_U‖` or normalise the
   logits, so confidence cannot be bought by scaling and a margin cap becomes meaningful.
   This is the experiment this session could not run, and it directly tests the cross-talk
   story.
3. **Second-order / Gauss–Newton training** (K-FAC, or full-batch CG at these sizes). If the
   bottleneck really is "solve a linear system by first-order descent", curvature should
   close much of the gap.
4. **Initialise at the construction and fine-tune with something that will not destroy it** —
   very low `lr`, or a loss with zero gradient on already-correct facts. Separates stability
   from reachability.

---

## Important context

**Known limitations, all recorded in the repo — do not rediscover them:**

* The acc≥0.9 numbers at large `d` are **round-budget-limited, not capacity-limited**. At
  d=32, n=4096 the `keep=0.92` schedule reaches 0.905 at 400 rounds versus 0.840 at 150.
  `rounds=150` is fixed deliberately; raising it per point would be tuning the metric.
* **The margin gap is a description, not a demonstrated cause.** See above.
* **The retrained-readout probe is invalid for value codes** — it scores 0.07–0.10 on
  features whose own readout scores 1.000. `probe_coding.py` prints `n/a` for those rows.
* **`twosided`'s 150-round iteration is the part a referee should push on.** Every weight
  comes from a ridge regression and no gradient is computed, but it is an iterative
  procedure with a step rule. The argument for it, and the reasons it might not persuade,
  are in `docs/twosided-construction.md` §9.

**A number worth explaining to anyone new:** run the construction at exactly `n = 4d²` and
accuracy is 0.8516 / 0.8518 / 0.8520 / 0.8508 at d=16/32/64/128 — the `keep=0.85` schedule
returning precisely what it was asked to keep. Four sizes, four decimals. Log in
`results/ceiling.log`.

**Traps that cost time (also in 002):**

* `torch.linalg.lu_factor` raises on an exactly singular block, and in float32 a `mu` of
  `1e-8` does not perturb a Gram with entries of order `width` — the ridge silently
  vanishes. Factorisations are done in float64 for this reason; keep the split.
* The capacity search's ramp assumes monotonicity in `n_facts`, but this construction
  converges *more slowly at low load*. A spurious first-probe failure aborts the search and
  reports nonsense. The `rho` sweep covers it; re-check the low-load end if the sweep is
  ever trimmed.
* `probe_coding.py` / `probe_reachability.py` need their own training loops — `model.train`
  clones its inputs and returns only the accuracy.

**Commands**

```bash
uv run pytest                                                  # 53 tests
uv run python run_scaling.py --ds 16 32 64 128 --conditions twosided --workers 8
uv run python run_scaling.py --ds 16 32 64 128 --report-only
uv run python probe_coding.py --d 64
uv run python probe_reachability.py --d 32
```

**Layout added this session**

```
handcode/twosided.py             the construction
probe_coding.py                  density, weight scale, pattern-vs-magnitude
probe_reachability.py            does gradient descent stay at the construction?
docs/twosided-construction.md    the write-up, aimed at the post's authors
reference/README.md              provenance for the vendored third-party source
results/coding.json              probe_coding output
results/reachability.json        probe_reachability output
results/{ceiling,longtrain,longtrain64}.log   raw logs behind the caveats
tests/test_twosided.py           15 checks, incl. the freeze-first ablation
```
