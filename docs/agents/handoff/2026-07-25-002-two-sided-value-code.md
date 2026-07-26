# Handoff — the two-sided value code, and why gradient descent won't go there

**Date:** 2026-07-25
**Repo:** `/Users/eppie/claude_projects/handcode` (now a git repo — `main`)
**Previous handoff:** `2026-07-25-001-sequence-memorization-challenge.md`
**Subject:** Implementing that handoff's "concrete proposal", and what the result says about
the authors' actual question

---

## What was accomplished

**1. The proposed construction works, and beats the trained model at every size.**
`handcode/twosided.py`, condition `twosided` in the capacity harness.

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

The previous session's `linsolve` beat the *exponent* but lost 2× on the *prefactor*, so its
curve only crossed the trained one at d≈60. This one is ahead everywhere.

**2. The mechanism, which is the part worth keeping.** Written up for a general reader in
`docs/twosided-construction.md`, intended to be usable as a contribution back to the post.

The `2d²` ceiling the last session derived is a bound on *built gates*, not on value codes.
If the active set `S_a` depends on one token only,

```
s = Σ_{i∈S_a} (u_i[a] + v_i[b]) = [Σ_{i∈S_a} u_i[a]] + [Σ_{i∈S_a} v_i[b]]
                                   one number per first token
```

so the entire first embedding contributes `2d` numbers rather than `2d·d`, whatever its
entries are. Building a *two*-token gate explicitly hits a different wall: a gated neuron
must emit a positive value to be seen, and that positivity constraint costs enough rank to
cap the budget near `2.48d²`.

So build no gate. Take `A(a,b) = {i : u_i[a] + v_i[b] > 0}` — the ReLU's own sign pattern.
It depends on both tokens by construction, and the positivity constraint evaporates because
a neuron that would go negative simply switches off. Both embeddings then carry facts:
`4d²` unknowns, one equation per fact.

**3. The step rule is the one non-obvious ingredient.** Freezing the pattern makes the
system linear and it decomposes into one ridge regression per token (block Gauss-Seidel).
But taking the full solve flips ~43% of the pattern and collapses everything to chance, via
a **one-sided ratchet**: a neuron that ends up on the wrong side of zero always makes the
true ReLU sum *larger* than the equation's, so every decode overshoots, the next solve
shrinks the weights, more neurons fall below zero, and density falls 0.53 → 0.05.

Capping the step so at most 2% of the pattern flips fixes it, and needs no search: each unit
is affine in the step length, so it crosses zero at exactly one `α`, and the longest
admissible step is the 2% order statistic of those crossing times. It is self-scheduling —
tiny early, 1.0 once the pattern settles, at which point the fixed point is reached and the
loop exits.

**4. Gradient descent does not merely fail to find this — it walks away from it.**
`probe_reachability.py`, and §4 of `FINDINGS.md`. At d=32, n=3168:

| | accuracy | cross-entropy |
|---|---|---|
| the construction | 1.0000 | 0.898 |
| + 2000 Adam epochs from there | 0.834 | 0.737 |

Adam lowered its loss 18% while giving up 17% of the facts. **The construction is not a
stationary point of the training objective**, so no initialization or schedule would settle
on it. Margins at 100% accuracy: construction median 0.42 (min 0.02), trained model median
7.77 (min 5.93).

**5. Qualitative match improved a lot over `linsolve`.** `probe_coding.py`, d=64, each at
~90% of its own acc=1 capacity:

| | trained | their hand-coded | linsolve | twosided |
|---|---|---|---|---|
| density | 0.53 | 0.82 | 0.08 | **0.44** |
| max abs embedding weight | 6.4 | 1 | 4.7e2 | 5.4e2 |
| max abs unembedding weight | 6.3 | 2 | 4.2e4 | **6.4e1** |
| params carrying facts | `5d²` | — | `2d²` | `4d²` |

**6. Performance.** A `twosided` solve at d=128, 49k facts takes ~8s. Factorizing the
per-token normal equations once per round instead of once per sweep, replacing the
12-point step search with one order statistic, carrying the masked decode forward
incrementally, and running the bulk arithmetic in float32.

---

## Key decisions and things ruled out

**Ruled out, with evidence — do not re-litigate:**

* **`t0 ~ d²` is wrong here**, even though it is what `linsolve` needs. It costs 4% accuracy
  at d=128 in float32 while float64 still reads 1.000 — catastrophic cancellation in the
  readout, which forms logits of size `d(t0+d)` by subtracting two nearly equal numbers.
  `t0 = 16d` holds the sign pattern still just as well and keeps two orders of float32
  margin. `t0 = 0` fails outright (0.03–0.07 accuracy).
* **The bias neuron's height `β` is a pure gauge** — it cancels from every logit, so it
  cannot change accuracy, only which matrix carries the offset's magnitude. Setting it to
  twice the largest solved value takes `max|W_U|` from 1056 to 64 at d=64.
* **Fixed damping is worse than the flip cap** (d=16, n=640: 0.934 vs 1.000) and costs 3×
  the block solves. A line search on the residual is worse than both (0.863).
* **The margin hypothesis is untestable in this architecture.** The natural test — retrain
  under an objective that stops caring past a small margin — fails because the loss is
  scale-invariant: the unconstrained unembedding meets any fixed margin target by scaling
  up. A hinge at margin 0.5/1.0 did not train at all (0.15 where CE reaches 1.000), and
  CE on `β·logits` for β up to 100 left capacity unchanged. **Do not report the margin gap
  as the cause of the capacity gap** — it is a description of where the two solutions sit.
* **The retrained-readout probe is not a valid instrument for a value code.** It scores
  0.07–0.10 on `linsolve` and `twosided` features whose own readouts score 1.000, even when
  started from a closed-form ridge fit. That is a failure to re-derive the decode, not an
  absence of information; `probe_coding.py` now marks those cells `n/a` rather than printing
  a meaningless ratio.

**Deliberate choices:**

* `rounds = 150`, fixed, not tuned per point. More rounds do help near the ceiling (at d=32,
  n=4096 the `keep=0.92` schedule reaches 0.905 at 400 rounds versus 0.840 at 150), so the
  reported acc≥0.9 numbers at large `d` are budget-limited. Reported at a fixed budget on
  purpose; raising it per point would be tuning the metric.
* Working precision float32, with the per-token *factorizations* in float64. A ridge of
  `1e-8` is below float32 epsilon against Gram entries of order `width`, so adding it was a
  silent no-op and singular blocks crashed `lu_factor`. This bit once; keep the split.

---

## Important context

**Caveat that qualifies the headline — carry this forward.** The trained baseline uses the
post's recipe (≤5000 epochs, patience 100) and reproduces their published curves to
0.95–1.05, so the comparison is like-for-like *against the post*. But that recipe is not
converged. At d=32, single seed:

| n_facts | 5k epochs | 50k | 200k |
|---|---|---|---|
| 2080 | 0.9995 | 1.0000 | 1.0000 |
| 2560 | 0.8148 | 0.9969 | 0.9980 |
| 2880 | 0.6347 | 0.8382 | 0.8278 |
| 3168 | 0.6133 | 0.7080 | 0.7465 |

So converged gradient descent reaches ~2560 at acc=1, not 2080, and the honest d=32 acc=1
ratio is **~1.24×, not 1.52×**. acc≥0.9 barely moves (2528 → ~2700).

**This is a small-`d` artifact — the same check at d=64 clears the baseline:**

| n_facts | 5k epochs | 60k |
|---|---|---|
| 7296 (reported capacity) | 1.0000 (@1613) | 1.0000 (@1429) |
| 9600 | 0.7168 | 0.7765 |
| 11200 | 0.5380 | 0.6300 |
| 12800 (the construction's) | 0.4237 | 0.4909 |

At d=64 the recipe is already converged: 7296 is reached by epoch ~1600 whatever the budget,
and 9600 plateaus at 0.78 with twelve times the epochs. So **the d=64 ratio of 1.75× stands
against converged gradient descent**; only the d=16 and d=32 acc=1 points are inflated. The
d=128 version was not run (each point is hours).

**A number worth explaining.** Run the construction at exactly `n = 4d²` — every pair that
exists — and the best accuracy is 0.8516 / 0.8518 / 0.8520 / 0.8508 at d=16/32/64/128. That
is the `keep=0.85` drop schedule returning precisely what it was asked to keep: drop 15% and
the remaining `3.4d²` equations fit inside `4d²` unknowns. Four sizes, four decimals.

**Plain power laws are more legible than the post's `a·d^b/ln d` form.** Fitting `C·d^p`
directly: `twosided` 2.05–2.10, `linsolve` 1.99–2.05, **trained 1.73–1.88**, their
hand-coded 1.35–1.71. The value codes are clean `d²` laws; the trained model is measurably
sub-quadratic, which is why the ratio grows with `d`. In the `/ln d` form a pure `C·d²` law
fits with `b ≈ 2.28`, so `twosided`'s `b = 2.3` is *not* super-quadratic — do not report it
as such.

**Open items**

* All eight `twosided` cells completed; `results/scaling.json` is current.
* The long-training check was not run at d=128 (hours per point). Given that the effect
  shrinks from d=32 to d=64 rather than growing, the d=128 ratio of 2.04× is unlikely to be
  budget-inflated, but that is an inference, not a measurement.

**Layout additions**

```
handcode/twosided.py             the construction
probe_coding.py                  density, weight scale, pattern-vs-magnitude
probe_reachability.py            does gradient descent stay at the construction?
docs/twosided-construction.md    the write-up, aimed at the post's authors
results/coding.json              probe_coding output
results/reachability.json        probe_reachability output
tests/test_twosided.py           15 checks, incl. the freeze-first ablation
```

**Commands**

```bash
uv run pytest                                                  # 53 tests
uv run python run_scaling.py --ds 16 32 64 128 --conditions twosided --workers 8
uv run python probe_coding.py --d 64
uv run python probe_reachability.py --d 32
```

**Traps**

* `torch.linalg.lu_factor` raises on an exactly singular block. In float32 a `mu` of `1e-8`
  does not perturb a Gram whose entries are `O(width)`, so the ridge silently vanishes. The
  factorizations are done in float64 for this reason.
* The capacity search's ramp assumes monotonicity in `n_facts`. This construction converges
  *more slowly at low load* (the per-token systems are heavily under-determined and the flip
  cap pins the step), so a spurious failure at the first probe aborts the whole search and
  reports a nonsense capacity. The `rho` sweep covers it; if the sweep is ever trimmed,
  re-check the low-load end first.
* `probe_coding.py` and `probe_reachability.py` train models, so they need their own
  training loop — `model.train` clones its inputs and returns only the accuracy.
