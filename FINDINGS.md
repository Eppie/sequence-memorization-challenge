# What the trained model actually does

Notes toward the post's stated goal — *"a clearer understanding of how look-ups are
encoded"* — rather than toward the capacity benchmark. All measurements are on the
Figure-4 toy model, `handcode/` for the code.

Method throughout: take a model's hidden activations `H` on its fact set, coarsen the
**magnitudes** while keeping the **pattern** (`h → 1[h>0]`, or quantized to `L` levels),
and retrain a linear readout on each variant. Retraining matters — ridge under-reads
badly here (0.479 where the model itself gets 1.000) because ridge optimizes L2 while the
metric is argmax. Gradient descent is used for these *probes*; that is analysis, not a
challenge entry, so the no-GD rule does not apply.

## 1. The trained model switches coding scheme with load

d=64, trained to convergence at each fact count, readout retrained on each variant:

| n_facts | % of capacity | full | binary | 4 levels | 8 levels | binary/full |
|---|---|---|---|---|---|---|
| 500 | 7% | 1.000 | **1.000** | 1.000 | 1.000 | 1.00 |
| 1500 | 21% | 1.000 | 0.692 | 1.000 | 1.000 | 0.69 |
| 3000 | 41% | 1.000 | 0.249 | 1.000 | 1.000 | 0.25 |
| 6000 | 82% | 0.999 | 0.113 | 0.271 | 0.508 | 0.11 |
| 8000 | 110% | 0.953 | 0.089 | 0.125 | 0.216 | 0.09 |

**At low load the trained model is a pattern code — binarizing its activations costs
nothing. At capacity it is a magnitude code**, needing ~32 levels (~5 bits/neuron) to
retain full accuracy, with the pattern alone retaining 9%.

Sparsity barely moves across this range (37.8 → 33.3 active neurons of 64). The model is
not changing *which* neurons fire; it is changing how much it says with *how loudly* they
fire.

## 2. The transition is universal in load-fraction

binary/full, at matched fractions of each size's measured capacity:

| load | d=32 | d=64 | d=128 |
|---|---|---|---|
| 10% | 1.00 | 1.00 | 1.00 |
| 30% | 0.45 | 0.35 | 0.30 |
| 60% | 0.24 | 0.16 | 0.11 |
| 90% | 0.16 | 0.10 | 0.07 |

The same curve at every size, slightly steeper as `d` grows. Active neurons per fact stay
at 0.57–0.60 of `d` throughout — density is a constant of the architecture, not a knob
the model uses.

## 3. The authors' construction is stuck in the low-load regime

Each construction at ~90% of *its own* acc=1 capacity, d=64:

| construction | n | density | full | binary | binary/full | coding |
|---|---|---|---|---|---|---|
| trained | 6566 | 0.53 | 1.000 | 0.105 | 0.11 | **magnitude** |
| their hand-coded | 194 | 0.82 | 1.000 | 1.000 | 1.00 | **pattern** |
| linsolve | 5414 | 0.08 | — | — | — | magnitude (by construction) |
| twosided | 11520 | 0.44 | — | — | — | magnitude (by construction) |

Their silence code is a *pure* pattern code: binarizing is free, because the code is
literally "are these neurons off". That is exactly the regime the trained model occupies
at ≤10% load and abandons thereafter.

`linsolve` needs no probe: its decoded sum *is* the label, so it is a magnitude code by
construction. (A probe would mislead — a randomly-initialized readout scores 0.035 on its
features, because the decode needs weights of ~2.5e6 that Adam will not find from a small
init. That is an optimization failure, not an information one.)

## 4. The best construction is not a stationary point of the training objective

`probe_reachability.py`. The `twosided` construction (see
[docs/twosided-construction.md](docs/twosided-construction.md)) stores 1.4–2.0× what a
trained model of the same size does, so those weights exist inside the architecture and
gradient descent does not reach them. There are two ways that can be true: gradient descent
cannot get there, or it would not stay. It is the second.

Start Adam exactly at the construction, at d=32 and n=3168 — the construction's own acc=1
capacity, 1.52× the trained model's:

| | accuracy | cross-entropy |
|---|---|---|
| the construction | **1.0000** | 0.898 |
| + 2000 Adam epochs from there | 0.834 | **0.737** |

It walks off within 200 epochs and settles near 0.83, **lowering its loss by 18% while
giving up 17% of the facts**. Nothing is malfunctioning: it is minimizing its objective, and
the objective disagrees with the metric. So no initialization and no schedule would settle
gradient descent on this solution — it is not a place the loss wants to be.

The visible difference is decision margin. At 100% accuracy on their own fact sets:

| | median best-minus-second logit | min |
|---|---|---|
| the construction | **0.42** | 0.02 |
| trained model | **7.77** | 5.93 |

The quadratic decode's gap between adjacent labels is exactly 0.5 and the construction
spends all of it, so every fact is correct by a hair against logits spanning ±500. (The
trained margin depends on when training stops: 7.77 after the full 5000-epoch budget,
~3.3 at the first epoch that reaches 100%, which is the snapshot §5 uses.)

**That the margin difference *causes* the capacity gap is a hypothesis this section could
not test.** The natural test — retrain under an objective that stops caring past a small
margin — does not work here, because the loss is scale-invariant: the unembedding is
unconstrained, so the network meets any fixed margin target by scaling itself up.
Accordingly, a hinge loss at margin 0.5 and 1.0 failed to train at all (0.15 where
cross-entropy reaches 1.000), and cross-entropy on `β·logits` for `β` up to 100 left
capacity unchanged. §7 later closes the loophole by fixing the readout, and the margin
hypothesis does not survive it: with relative margins pinned, Adam stalls at the same
capacity anyway. §§5–6 locate the real difference in noise tolerance.

A second observation points the same way. The probe in §3 hands gradient descent the
construction's *own activations* and asks only for the readout — the hard half already
solved. Starting from a closed-form ridge fit, it reaches 0.07–0.10 where the construction's
readout scores 1.000 on those identical features.

## 5. The benchmark does not charge for fragility, and the value codes are maximally fragile

`probe_robustness.py`. Perturb each weight matrix with Gaussian noise scaled to its own
RMS (`W → W + σ·rms(W)·ε`, so the comparison is invariant to the constructions' very
different weight scales), 20 trials per noise level, and report **σ90**: the relative
noise at which mean accuracy first falls below 0.9. Every construction is built at its
own acc=1 capacity and a model is *trained* to acc=1 at the same fact count, so each
comparison is load-matched. Trained baselines are snapshotted at the first epoch that
reaches 100% — the moment the benchmark would score them; training past that point only
fattens margins, so the trained numbers below are conservative.

| solution | n (d=32) | σ90 weight | n (d=64) | σ90 weight |
|---|---|---|---|---|
| their hand-coded | 130 | 2.4e-1 | 216 | 2.7e-1 |
| trained, same n | 130 | 4.6e-1 | 216 | 6.2e-1 |
| linsolve | 1408 | 1.0e-5 | 6016 | ≤1.0e-5 |
| trained, same n | 1408 | 6.3e-2 | 6016 | 3.3e-2 |
| trained at its capacity | 2080 | **2.0e-2** | 7296 | **1.3e-2** |
| twosided, same n | 2080 | **1.5e-5** | 7296 | **≤1.0e-5** |
| twosided at its capacity | 3168 | 1.6e-5 | 12800 | ≤1.0e-5 |

Full curves in `results/robustness.png` (drawn by `plot_robustness.py`).

Two facts, consistent across both sizes:

* **At matched load, the trained model tolerates ~3 orders of magnitude more relative
  weight noise than the value codes** (1300× at d=32; ≥1200× at d=64, where the value
  codes sit at the floor of the swept range). Perturbing hidden activations instead of
  weights gives the same ordering.
* **The authors' construction is in the trained model's robustness class.** At its own
  capacity it holds σ90 = 0.24–0.27 against the load-matched trained model's 0.46–0.62 —
  a factor of ~2, not ~1000. Their silence code plays the same robust game gradient
  descent plays, just less efficiently. The value codes win capacity by leaving that
  game entirely.

The absolute numbers make the walk-away of §4 arithmetic rather than mysterious. In
absolute units the construction's tolerance is σ90·rms(W) ≈ 1.1e-3 per weight, while one
Adam step at the post's lr moves every weight by ~1e-2 — **a single optimizer step is
roughly 9× the construction's entire noise budget**. The trained solutions sit on the
other side of the same line: their absolute tolerance is 5–20 Adam steps. A solution can
only be found *and held* by an optimizer whose own churn it can absorb; the value codes
are three orders of magnitude below that threshold, which is to say their extra capacity
is purchased in exactly the currency an optimizer cannot spend.

## 6. When Adam walks off the construction, it does not choose its casualties by margin

`probe_walkaway.py`, d=32, n=3168. If cross-entropy were selectively trading the
thinnest-margin facts for margin on the rest, the facts lost in §4's walk-away should be
the ones that started with the smallest margins. They are not:

* survival is nearly flat across starting-margin deciles (0.73 → 0.86 from thinnest to
  fattest), rank correlation **0.048**;
* survivors' margins fatten collectively, median 0.42 → 1.02.

So the walk-away is not a negotiated sale of individual facts. The first steps
re-introduce interference across *all* facts at once — consistent with §5's finding that
one step exceeds the entire noise budget — and which facts break is close to random.
What survives is then re-margined as a group. (Final accuracy after 2000 epochs is 0.76
here versus the 0.83 best-en-route reported in §4; both runs walk off within 200 epochs.)

## 7. The unembedding is an optimization aid, not storage

`probe_fixed_readout.py`, d=32. Fix the unembedding to the construction's quadratic
decode — a rank-2 ladder that carries no fact information and pins the *relative* margin
geometry, closing the scaling loophole that made margin-cap experiments untestable (§4)
— and train only the embeddings with Adam:

| n_facts | post's recipe (5k, patience 100) | 50k epochs | 200k epochs | frozen random readout, 5k |
|---|---|---|---|---|
| 1408 | 0.04 | **1.000** | — | 0.56 |
| 2080 (trained capacity) | 0.04 | **1.000** | — | 0.36 |
| 2560 (converged-GD capacity) | 0.04 | 0.78 | 0.86 | 0.28 |
| 3168 (construction's capacity) | 0.04 | 0.39 | 0.39 | 0.23 |
| 3904 | 0.04 | 0.26 | — | 0.19 |

Three separate conclusions:

* **Storage.** Adam reaches 100% at the trained model's full capacity through a readout
  with zero fact information. The unembedding's `d²` parameters were never load-bearing
  for *capacity* — everything the trained model stores fits in the embeddings alone,
  exactly as the construction's parameter counting assumes.
* **Optimization.** With the decode fixed, relative margins are architecturally capped —
  cross-entropy can no longer buy margin with facts — and Adam *still* tops out at the
  free-readout ceiling, not the construction's: 2080 in full, 0.86 at 2560 after 200k
  epochs (converged free-readout training reaches ~0.997 there), and a hard stall at
  0.39 at 3168 that more budget does not move. The remaining gap to 3168 therefore is
  not the objective preferring fat margins; it is the optimizer. The construction gets
  there by solving the frozen-pattern linear system exactly (a Newton step); first-order
  descent on the same equations does not, at any budget tried.
* **Trainability.** Under the post's own recipe the fixed-decode model reaches 4% — the
  landscape through the rigid decode has plateaus the patience-100 early stop cannot
  cross. What the free readout buys gradient descent is not capacity but *speed*: the
  freedom to rotate the readout early is what makes the problem trainable in 5000 epochs.

Finally, the robustness of the solution Adam finds under the fixed decode
(`results/fixed_readout_robustness.json`, n=2080): σ90 = 3.3e-4 — 60× more fragile than
the free-readout trained model at the same load, but 20× more robust than the Newton
construction. Fragility decomposes cleanly: most of it is intrinsic to the quadratic
value *code*, and the exact solve contributes the final order of magnitude by leaving
zero slack. The trained model's robustness lives substantially in its freedom to choose
the readout.

## 8. What the trained solution looks like at the weight level

`probe_structure.py`, d=32, all models at (or within one fact of) 100% accuracy. Two
measurements that any constructive account of the trained code has to reproduce.

**Per-fact weight-space radius** — `margin_f / ‖∇_W margin_f‖`, the first-order distance
to fact f's decision boundary in weight space; σ90 is the aggregate of these, this is the
distribution (512-fact sample):

| model | median | p10 | min | CV |
|---|---|---|---|---|
| twosided @ 3168 | 4.3e-4 | 3.6e-4 | 2.3e-4 | **0.13** |
| fixed-quad GD @ 2080 | 9.8e-3 | 8.4e-3 | 4.4e-6 | 0.65 |
| trained @ 2080, first acc=1 | 9.4e-2 | 6.5e-2 | 2.4e-2 | 0.85 |
| trained @ 2080, post's full budget | **1.8e-1** | **1.4e-1** | **1.1e-1** | 0.68 |

Two signatures. The construction's radii are *uniformly thin* — every fact equally close
to failure, which is what "solved the equations exactly, spent no slack anywhere" looks
like. The trained model's are two hundred times larger at the minimum, and continued
training past 100% accuracy raises the **floor** (min 2.4e-2 → 1.1e-1, a 4.4× lift) much
faster than the median (1.9×) — cross-entropy keeps working on the *worst* margins, which
is the classic implicit-bias-toward-max-margin signature made visible at the weight
level. What gradient descent builds is, to first order, a margin-floor-equalized
solution of the same inequalities the constructions treat as equalities.

**Readout structure** — singular spectrum of the unembedding, and the fraction of each
neuron's class profile `down[:, i]` explained by a graded (linear-in-class) ladder, the
structure every value code here reads out with:

| model | participation ratio | rank for 90% energy | graded-ladder fraction |
|---|---|---|---|
| twosided / fixed-quad | 1.0 | 1 | 1.00 |
| trained @ 2080 (either budget) | **18** | **12** | **0.03** |

The trained readout is not a ladder at all: essentially zero linear-in-class structure,
with label information spread over ~12–18 effective directions of the 32 available. So
the trained code is a **magnitude code read through a high-rank, unstructured codebook**
— graded activations (§1–3), but nothing like the single-coordinate decode any of the
constructions use. A constructive account has to produce that shape, not just that
capacity.

## 9. Redundancy is constructible; the last factor of ~30 is not (yet)

`handcode/digitcode.py`, `probe_digitcode.py`, `probe_pedestal.py`. The digit-code
family spreads a label's log₂(d) bits over `m` neuron groups (base-`p` digits, `p =
⌈d^(1/m)⌉`, one two-sided-style sum equation per group; still ridge-only), trading
capacity `4d²/m` for per-coordinate precision. Sweeping `m` and the stabilizer:

* redundancy alone buys only ~20× robustness across the whole family (m=1→5) while
  capacity falls 8× — the frontier stays 400–700× below trained at every load;
* the dominant fragility is the **pedestal**: the stabilizer `t0` every frozen-pattern
  equality solve needs inflates ‖h‖, and readout noise scales with it. Shrinking `t0`
  to the edge of solvability (~d/2) buys another ~10× and drops the weights to trained
  scale;
* fully optimized, the family reaches **n = 1584 (76% of trained capacity) at σ90 =
  1.3e-3** — a residual **30×** from the trained model at the same load, attributable
  to margin-inequality packing (§8's floor-equalization), which no equality solve in
  this family performs.

The assembled account, and what would finish it, is in
[docs/what-gd-builds.md](docs/what-gd-builds.md).

## 10. No standard optimizer packs in more facts than converged Adam

`probe_optimizers.py`, d=32, all full-batch on cross-entropy from the standard init,
best over seeds (and over learning rates for SGD); the post's recipe, a converged Adam
budget, SGD with momentum, and L-BFGS with strong-Wolfe line search:

| n_facts | adam (post recipe) | adam (converged) | sgd+momentum | L-BFGS |
|---|---|---|---|---|
| 2560 | 0.849 | **0.998** | 0.939 | 0.824 |
| 2880 | 0.661 | **0.808** | 0.769 | 0.707 |
| 3168 | 0.589 | **0.730** | 0.686 | 0.622 |

Converged Adam wins at every load; L-BFGS — the curvature-aware candidate that the
"first-order vs Newton" story might have suggested would close the gap — does *worse*
than the post's plain recipe at 2560. So the 2560 → 3168 headroom is not reachable by
swapping in a better generic trainer: the only method in this repo that reaches 3168 is
the frozen-pattern Newton solve, which is not a gradient trainer at all, and §5 says the
facts it gains live in a fragility regime no optimizer that moves in ~lr-sized steps
could hold anyway. (Caveat: torch's L-BFGS on a nonsmooth ReLU objective is a weak
stand-in for serious second-order methods — K-FAC, Shampoo, or exact Gauss-Newton were
not tried.)

## 11. Their outlier: the architecture with no slack mechanism

`probe_badcombo.py`, `results/badcombo*.json`. The post's appendix B flags one
combination — MLP=✅, Norms=✅, Res=❌, Bias=❌, ReLU — as doing far worse than its
parts predict, and asks whether that is training dynamics or architectural capacity.
Rerunning their own vendored model at their scale (V=32, d=16, 2Emb mixing; best of 2
seeds, lr 1e-2), with each setting flipped one at a time:

| accuracy at n_facts | 512 | 640 | 768 | 896 | 1000 |
|---|---|---|---|---|---|
| **the bad combo** | 1.00 | 0.94 | **0.68** | 0.67 | 0.60 |
| bad combo, 12× budget, 100× patience | 1.00 | 0.95 | **0.71** | 0.72 | 0.66 |
| flip act → GELU | 1.00 | 1.00 | 0.89 | 0.77 | 0.68 |
| flip bias → on | 1.00 | 0.98 | 0.86 | 0.80 | 0.71 |
| flip res → on | 1.00 | 1.00 | **1.00** | 0.95 | 0.79 |
| flip norms → off | 1.00 | 0.99 | 0.90 | 0.66 | 0.61 |

Three observations that narrow their question:

* **The deficit is not the training budget.** A 60k-epoch, patience-10k run recovers
  three points of accuracy where the controls are 15–30 ahead. Whatever is wrong, more
  Adam does not fix it.
* **It is not dead units at the moment of scoring** (0–2% of facts have an all-dead
  hidden layer in the short runs) — but extended training *raises* whole-fact death to
  12% at n=768, so the failure deepens with optimization rather than easing, a ratchet
  rather than a plateau.
* **Every single-setting flip restores most of the capacity, and each flip adds a
  different escape valve.** GELU leaks gradient through inactive units; a bias shifts
  thresholds without rotating hyperplanes; a residual bypasses the MLP; dropping the
  norm lets overall scale grow. The bad combo is precisely the architecture with *no*
  slack mechanism: ReLU's hard zero, thresholds pinned through the origin of a
  normalized sphere, no bypass, no scale. In that geometry the only way to fix one
  fact is to rotate hyperplanes that other facts already depend on.

So the answer to their "training dynamics or architectural capacity?" is: an
optimization-geometry problem — dynamics, but of a structural kind that budget does not
cure. It also rhymes with §§5–9: the slack mechanisms this combination removes are
exactly the currencies gradient descent normally spends to buy robust storage, and with
all of them gone its capacity falls by roughly the same ~30% that separates the
equality-constrained codes from trained models. (Protocol caveat: their appendix used
an lr ladder and 11 attempts per point; this is single-lr, best-of-2 — the comparison
across rows is internal and like-for-like, the absolute numbers are not theirs.)

## 12. The trained gate settles early, and its stability is co-sized with its margins

`probe_patterns.py`. The account's last unmeasured clause was that trained models need
no stabilizer pedestal because gradient descent stabilizes the ReLU sign pattern
*dynamically*. Two direct measurements, d=32:

**Churn.** Training at n=2080 under the post's recipe, the fraction of (fact, neuron)
pre-activation signs flipping per epoch collapses from 100% (init) through 1.3% by
epoch 10 to ~0.5% by epoch 30 — while accuracy is still at 21%. The per-epoch rate is
calm early, but it does not mean the gate is final: the finished gate differs from its
random init in **41% of bits** (`probe_remargin.py`), so ~0.5%/epoch compounds into a
long, slow consolidation. The right statement is that gradient descent rebuilds the
gate gradually and substantially while never churning fast — the opposite of the
frozen-pattern solves, which hold the gate rigid and move everything else.

**Co-sizing.** Under weight noise, compare where accuracy fails (σ90) with where the
pattern starts moving (1% of sign bits flipped):

| model | σ(acc < 0.9) | σ(1% pattern flips) | ratio |
|---|---|---|---|
| trained @ 2080 | 3.2e-2 | 1.0e-2 | **0.3×** |
| digit m=2, pedestal-optimized @ 1584 | 3.2e-3 | 3.2e-2 | 10× |
| twosided @ 3168 | 3.2e-5 | 1.0e-2 | **316×** |

The trained model's gate stability and decision margins sit at the same noise scale —
it even keeps decoding with ~1% of its gate bits flipped, so the pattern is itself a
robust code rather than brittle bits. The exact solve's accuracy dies 316× before its
gate moves at all: its pattern stability (bought with the pedestal) is irrelevant to
its failure mode, which is decode precision. That mismatch — slack spent where the
metric looks instead of where failure comes from — is the equality-code design in one
number.

One picture of the whole session: `results/frontier.png` (`plot_frontier.py`), the
(capacity, σ90) plane with every construction, the trained frontier, and the
max-margin LP landing on the trained point.

## 13. Two construction attempts at geometry discovery, and where they narrow to

`probe_remargin.py`, `probe_geometry_ascent.py`. With margins reduced to LPs, the open
constructive problem is discovering the adapted geometry. Two attempts:

**Ridge bootstrap — fails at the first joint.** Take a ridge-only gate (the twosided
solve's own sign pattern), fit a ridge readout, alternate with the embeddings-LP. It
never gets started: a ridge-fit readout supports *no positive margin even on the
trained gate* (γ* ≈ 0, against 9.4 for the trained gate with its own co-adapted
readout; ridge on the trained activations decodes only 47%). Coordinate ascent cannot
bootstrap from an infeasible point — the (gate, readout) pair is co-adapted or it is
nothing.

**Two-LP max-margin ascent — stays feasible, and hits the real wall.** Start from a
construction that already decodes at thin margins (its gate + its own readout) and
alternate two exact solves: max-min-margin over embeddings (readout frozen) and over
the readout (activations frozen). On the pedestal-optimized digit gate at n=1584 the
mechanics work — accuracy stays 1.0 and the minimum margin climbs **1000×** (3e-4 →
0.32) — but σ90 moves only 1.3e-3 → **1.7e-3**, still ~25× short of trained at the
same load. Margins per unit weight box grow enormously; robustness barely follows.
Since the same LP machinery on the *trained* gate reproduces trained σ90 exactly (§ in
`docs/what-gd-builds.md`), the deficit is now located to one object: **the gate**. The
sign patterns that frozen-pattern ridge solves discover lack some property the trained
gate has — the property gradient descent builds during its slow 41%-of-bits
consolidation — and no exact margin ascent on top of a deficient gate can buy it back.

So the constructive program reduces to a single question: *what makes a gate good, and
can that be constructed?* Everything else — values, readout, margins — is an LP away.
(Engineering notes for whoever picks this up: the twosided ladder readout is rank-2
and makes the cutting-plane hints degenerate — thrashing, negative interim margins;
carry cut sets across rounds instead of rebuilding, and prefer starting readouts with
spread spectra. The d=64 reconstruct LP is compute-bound at ~550k rows; cutting planes
on the pattern-consistency rows, not just the margin rows, is the likely fix.)

## 14. Gate quality is not a pattern statistic, and margin pressure cannot build it

`probe_gatequality.py`. Section 13 reduced the constructive program to one question —
what makes a gate good? This section answers what it is *not*, three ways, on a
six-gate zoo at one load (d=32, n=1584, where the trained model's own σ90 is 4.5e-2).

**Not an intrinsic statistic (phase `metrics`).** The two candidate metrics from §13 —
per-token design conditioning and same-token active-set decorrelation — were computed
for every gate. They fail, decisively: the trained gate is indistinguishable from a
*density-matched random additive gate* on all of them (same-token correlation 0.323 vs
0.334, both at the 1/3 a Gaussian calculation predicts for any additive threshold
gate; median per-token-design smin 0.29 vs 0.26; κ 59 vs 68), and indistinguishable
from **its own random init** despite having drifted 41% of its bits away from it.
Worse than unsupported — refuted: the tail statistic `smin_p10` ranks the random and
init gates *above* the trained gate. The one thing conditioning does detect is
pedestal degeneracy (twosided: smin 0.06, κ 257; digit m=2 intermediate), which
orders the two solve gates correctly and nothing else.

**Not readout freedom (phase `predict`).** The two-LP max-margin ascent — embeddings
and readout both exactly optimized, a test no generic gate had been given (the
`probe_maxmargin` mixed conditions all froze the readout) — was run on every gate:

| gate | best LP point | σ90 |
|---|---|---|
| trained | acc 1.0, min margin 17.3 | **4.40e-2** (its own model: 4.55e-2) |
| digit m=2 | acc 1.0, min margin 0.99 | 2.85e-3 |
| twosided | LP fails (ladder-readout degeneracy); construction kept | 1.41e-5 |
| random additive | **acc 0.03** — no positive margin exists | — |
| init | **acc 0.03** — no positive margin exists | — |

The random and init gates are not fragile — they are *infeasible*: the fact set
cannot be stored on them at any positive margin even with both matrices LP-optimal.
Two patterns with identical family statistics — trained and random — sit at opposite
extremes of storage capability, so gate quality is relational, not combinatorial:
what makes a gate good is whether the cone of embeddings realizing its signs also
contains embeddings whose values decode the facts. Co-adaptation, one level further
down than §13 put it.

**Not margin-pressure co-adaptation (phase `drift`).** If gates are only ever good by
co-adaptation with a value scheme, the constructive move is to imitate the mechanism:
let the pattern drift under *max-margin* pressure (gradient descent's implicit
objective) instead of the ridge solves' equality pressure. The iteration: solve the
max-min-margin LP over embeddings with the sign-consistency rows dropped, step toward
the LP point under twosided's 2% flip cap, re-read the pattern, refit the readout-LP,
repeat — exact solves everywhere, no gradients. Seeded from the digit gate's feasible
point, the first step is the best constructed gate to date — σ90 1.27e-3 → **2.96e-3**,
above the frozen-pattern ceiling (2.85e-3) — and every subsequent round makes it
worse: σ90 declines monotonically through eight rounds (2.34e-3 → 1.7e-4) while
accuracy erodes, then collapses outright (0.73, 0.60) at 7.9% cumulative drift, by
which point even the masked-linear LP margin has decayed to zero — the drifted
pattern is no longer feasible *as a mask*, let alone as a gate. The flip ledger says
why: every round spends its full 2% quota but cumulative drift advances ~0.6% —
the flips churn back and forth rather than consolidate. Max-min margin pressure has
no memory: it re-targets a distant LP vertex each round, and the order-statistic step
keeps flipping whichever bits cross first. Gradient descent's consolidation — 0.5%
churn per epoch compounding to 41% drift with accuracy intact (§12) — moves pattern
and values together, continuously, under pressure from every fact at once rather
than the worst one; the exact-solve imitation of it has none of those properties and
destroys what it touches.

The state of the reduction, then: gate quality cannot be scored by pattern statistics
(refuted), cannot be substituted by readout freedom (refuted), and cannot be built by
flip-capped exact-solve margin ascent (refuted — one step helps, iteration collapses).
What remains is the single mechanism that demonstrably builds it — gradient descent's
own coupled small-step dynamics — and the residual ~15× robustness gap between the
best constructed gate (2.96e-3) and the trained gate (4.40e-2) at matched load is now
the measured price of not having that mechanism in constructive form.

## 15. Spread pressure collapses the same way, and the gate is built during fitting

`probe_gatequality.py --phase drift --pressure spread` (and `predict` for
`trained_early`). §14's drift experiment left one natural alternative open: its
max-min-margin objective concentrates all pressure on the single worst fact, where
gradient descent's softmax loss pulls on *every* fact at once with saturating
strength. The spread-pressure variant makes that exact-solve-precise: per-fact margin
variables m_f, maximize Σ m_f subject to the same margin rows, bounded m_f ≤ τ = 0.5
so satisfied facts stop pulling (`solve_max_margin(..., spread_tau=τ)`). Everything
else — the 2% flip cap, the readout-LP refit, no memory across rounds — is identical
to §14, so the outcome is attributable to pressure structure alone.

Seeded from the digit gate at n=1584, the first step is again the best constructed
gate to date — σ90 1.27e-3 → **3.18e-3**, beating minmax drift's 2.96e-3 — and
iteration again destroys it: σ90 declines monotonically every round, accuracy erodes
from round 4, and the run early-stops at round 12 with accuracy 0.983, min margin
−1.68, σ90 5.9e-5, at 8.6% cumulative drift. The ledger shows the same churn
signature: every round spends its full 1013-flip quota, and the LP reports *every*
fact at the τ cap every round — mean m_f pinned at 0.5 from round 0 to the end —
while the realized model's minimum margin falls from +0.86 to −1.68. The solve's
belief and the stepped-and-re-read reality diverge immediately; spread pressure
degrades more gracefully than minmax (accuracy 0.983 at 8.6% drift vs 0.73 at 7.9%)
but the shape is identical. **All-facts saturating pressure was not the missing
ingredient.** What separates gradient descent from every exact-solve imitation tried
is not *what* it pressures but *how*: its direction is recomputed continuously, and
its flips happen only where a pre-activation happens to cross zero — near-ties,
locally almost free — where the capped LP step forces exactly the flips a distant
vertex wants most, valid only under the stale signs the step itself invalidates.

The second result relocates where that process does its work. `trained_early` — the
gate snapshotted at the *first* epoch training reaches 100% — was the zoo's one
unascended member; its two-LP ceiling comes out at **σ90 4.07e-2** (its own model
directly: 4.30e-2), against 4.40e-2 for the full-budget trained gate. The init gate
is infeasible (§14); the first-acc=1 gate is already at ~full quality. **The gate's
quality is built almost entirely during the fitting phase**, and the remaining ~4500
epochs of margin growth and continued drift — the long consolidation §12 measured —
add only ~8% in σ90 terms. Since the two-LP ascent defines gate quality as a function
of the pattern alone, gradient descent's quality trajectory moves only at flip
events; both §14's and this section's constructions failed while imitating the *late*
phase, which builds almost nothing. The object of study is the flip sequence of the
fitting phase — which flips the error-driven dynamics make between infeasibility and
first-acc=1, and why those flips enrich the cone.

Net: the residual between the best constructed gate (3.18e-3) and the trained ceiling
(4.40e-2) at matched load stands at a measured **~14×**, and the process thesis — the
good gate exists only as the fixed point of a coupled dynamics — has survived its
natural alternative. (Same caveat as §14: one seed, one d, one load; the headline
rows want a d=64 or second-seed spot check before anything is published from them.)

## 16. The gate becomes good in a narrow window at the end of fitting

`probe_gatequality.py --phase curve`. §15 located gate quality in the fitting phase
by its endpoints (init infeasible, first-acc=1 ≈ full). The curve phase resolves the
inside: re-run the post's recipe with the zoo's seed, snapshot 23 checkpoints (dense
early), and give every checkpoint's frozen pattern the same two-LP ascent the zoo got
— the σ90 ceiling of that gate with embeddings *and* readout exactly optimal. The
checkpoints are independent, so the ascents run in parallel worker processes; the
endpoint rows reproduce the zoo's numbers (first acc=1 at epoch 355 → ceiling
4.06e-2 vs the independently built trained_early's 4.07e-2; epoch 5000 → 4.40e-2).

| epoch | train acc | gate ceiling σ90 | net drift from init |
|---|---|---|---|
| 0–150 | 0.03–0.77 | **infeasible** (LP acc 0.032, γ = 0) | 0 → 0.389 |
| 200 | 0.90 | 1.44e-2 | 0.391 |
| 300 | 0.99 | 3.84e-2 | 0.395 |
| 355 (first acc=1) | 1.00 | 4.06e-2 | 0.396 |
| 500–5000 | 1.00 | 4.21e-2 → 4.40e-2 | 0.396 → 0.400 |

Three readings:

**Feasibility is won in a ~50-epoch window at the end of fitting.** Through epoch 150
— train accuracy already 0.77 — the gate still cannot store the fact set at any
positive margin, full stop. By epoch 200 (train accuracy 0.90) the ceiling is
1.44e-2: the gate becomes good *before the model does* — while ~150 facts are still
misclassified, the pattern already supports acc=1 at 4.5× the robustness of the best
constructed gate (3.18e-3). By epoch 300 it is at 87% of its final ceiling.

**The decisive bits are few.** The infeasible → feasible transition crosses only 4.6%
of pattern bits (epochs 150→200) — while the immediately preceding 50 epochs moved
*more* bits (6.9%, epochs 100→150) with the ceiling pinned at infeasible. Bit count
is not quality; §14's negative result (family statistics blind to quality) has a
dynamic counterpart here: the cone becomes rich through a small set of decisive,
relational flips, invisible in the aggregate flip budget.

**Net drift is a fitting-phase phenomenon; the margin phase polishes.** 39.1 of the
final 40.0 points of net drift-from-init are in place by epoch 200. After first
acc=1, 4800 epochs of ~0.55%/epoch churn — gross flipping totaling 28× the pattern
size over the full run — move the pattern only 5.6% net while the ceiling rises
4.06e-2 → 4.40e-2 (+8%). §12's long consolidation is real, but it is polish;
construction happens in the window where gradient descent is fighting for its last
~360 facts.

So the object of study narrows once more: epochs ~150–355, train accuracy 0.77 → 1.0,
where the error-driven dynamics make the flips that turn an unusable pattern into a
near-optimal one. The open questions are the flip-selection policy inside that window
and whether a fit-pressure construction started from infeasibility (rather than
margin pressure from feasibility, which §§14–15 refuted) can reproduce it. Caveats as
§§14–15: one seed, one d, one load.

## 17. The flip policy is trivial; the direction is everything

`probe_flippolicy.py`. Three phases inside §16's window: a dense ceiling curve
(checkpoints every 20 epochs, 140–360, each LP-ascended in parallel), per-flip
statistics for every consecutive pair against a churn-regime null pair (500→700),
and an embedding-space interpolation across the feasibility edge. One methods note
with scientific content: torch CPU training is not bit-deterministic (parallel
reduction order) and the trajectory is chaotic, so the dense run is an independent
micro-realization — its patterns diverge from §16's run at the ~1% bit level by
epoch 200, yet its ceiling curve lands on §16's almost exactly (ep200 1.39e-2 vs
1.44e-2; ep300 3.83e-2 vs 3.84e-2). The window phenomenon is a property of the
dynamics, not of one lucky trajectory.

**Dense curve.** Infeasible through epoch 180 (train acc ~0.85); feasible at 1.39e-2
by epoch 200; then a smooth fast climb (2.0e-2, 2.8e-2, 3.4e-2, 3.7e-2 at 220–280)
into the polish regime. The infeasible → feasible edge sits inside a single
20-epoch interval.

**Flip statistics.** Three facts, none of which §16's "error-driven flips" framing
predicted:

* *Flips are extreme near-ties.* The median |pre-activation| of a bit that flips in
  the next 20 epochs is 0.002–0.03, against 0.85–1.6 for the population — gradient
  descent flips only bits already 50–500× closer to zero than typical, paying ~zero
  margin cost per flip. (The failed LP drifts of §§14–15 do the opposite: the
  order-statistic step forces whichever flips the distant vertex wants most.)
* *Flips are error-agnostic in location.* Wrong-fact enrichment is 0.93–1.71 across
  the window — even at the feasibility edge, flips land on correctly-classified
  facts' bits nearly in proportion to their share. The errors drive the *embedding
  motion* globally through the loss; they do not select the flips locally.
* *The edge has no bit-level signature.* Net flip counts decline smoothly through it
  (1458 → 627 per 20 epochs; ~15× the churn regime's net rate), and near-tie-ness,
  direction (~50/50 on↔off), survival to epoch 5000 (~⅔), and neuron concentration
  (~uniform) are all flat across the transition. §14's lesson — quality is
  relational, invisible to statistics — holds at flip granularity too.

**Interpolation across the edge.** Per-bit pattern hybrids are not additively
realizable, so interpolate where every point is a real model: u_t = (1−t)·u_180 +
t·u_200. Every interpolate is feasible, and the first one is the result: at
t = 0.125 the pattern differs from the *infeasible* epoch-180 pattern by **0.35% of
bits (~180 bits)** and already ascends to σ90 9.7e-3 — 3× the best gate any
construction has produced. The ceiling then rises smoothly in t (1.04e-2 … 1.35e-2
at t=0.875, endpoint 1.39e-2). The feasibility edge is razor-thin, and crossing it
buys more than every exact-solve construction combined.

The synthesis, and the sharpest statement of the reduction to date: the interpolation
*is* a flip-capped step — near-ties cross first, exactly like the drift experiments'
`capped_step` — but pointed along gradient descent's own realized direction instead
of at an LP vertex. Same step mechanics, opposite outcome: the LP-vertex direction
destroys the gate (§§14–15), the training direction crosses from infeasible to 3× the
constructed record within 0.35% of bits. **The flip policy is trivial — cross
whatever is nearest zero along the motion. All of gate quality lives in the direction
field of the embedding dynamics.** The constructive question is now: what
characterizes that direction, and can an exact solve point along it from an
infeasible ~90%-fit start? The ~180 decisive bits at the edge are a small, explicitly
enumerable object for that study. Caveats: one seed family, one d, one load;
interpolation granularity t = 0.125.

## 18. No single direction crosses the edge — only the integrated path

`probe_flippolicy.py --phase direction`. §17 asked what characterizes the crossing
direction. The test: from one infeasible edge state (epoch 180 of a fresh
realization, train acc 0.867), take the *same* order-statistic step — near-ties
cross first, budget fixed at k flips — along seven directions, and LP-ascend every
stepped gate. Only the direction differs; the step mechanics, the budget, and the
ascent are identical.

Calibration first, and it is itself a finding: on this realization the edge along
the realized 180→200 path lies in t ∈ (0.125, 0.25] — §17's realization crossed by
t = 0.125. The edge's *location* varies between micro-realizations (its ep200
ceiling, 1.43e-2, matches the others); the *scale* — a fraction of a percent of
bits — is the robust object. The budget was therefore set to this realization's
measured crossing: k = 294 flips (0.58% of bits) at t = 0.25.

| direction (same k = 294, same step rule) | stepped acc | overlap w/ realized flips | ceiling |
|---|---|---|---|
| realized 20-epoch training delta | 0.873 | 1.00 | **9.1e-3** (1.13e-2 with t-interpolated readout start) |
| Adam's own next step (up_181 − up_180, extrapolated 4.6×) | 0.869 | 0.35 | infeasible |
| raw loss gradient −∇L | 0.869 | 0.31 | infeasible |
| fit-pressure spread LP, τ = 0.1 | 0.869 | 0.29 | infeasible |
| fit-pressure spread LP, τ = 0.5 | 0.869 | 0.27 | infeasible |
| max-min-margin LP vertex | 0.869 | 0.26 | infeasible |
| random | 0.869 | 0.21 | infeasible |
| (unstepped epoch-180 base) | 0.867 | — | infeasible |

The headline row is the second: **gradient descent's own next move, linearly
extended to the same flip budget, does not cross the edge.** Neither does the raw
gradient, nor exact-solve pressure of either §15 flavor, nor random stirring. Every
direction leaves training accuracy untouched (near-ties are free, §17) and flips a
~30%-overlapping set of near-tie bits — but only the *integrated* 20-epoch delta,
the accumulated result of ~20 re-evaluations of the direction after each step's
flips, lands feasible. The trajectory curves, and the curvature is load-bearing.
(The readout start shifts the crossed ceiling ~25% — 9.1e-3 vs 1.13e-2 — but not
feasibility.)

The decisive flip set itself (k = 169 at t = 0.125, `decisive_bits` in
`results/flippolicy.json`) is maximally distributed: 169 bits touch 158 distinct
facts — one near-tie bit per fact, essentially — across ~154 distinct embedding
cells per side, mildly error-enriched (18% vs a 13% base), median |pre-activation|
0.002. There is no localized structure to copy; the edge is crossed everywhere at
once, a little.

This closes the direction question the hard way and states the process thesis at
its finest granularity yet: **the crossing direction is not an evaluable field at
the start point — not the gradient, not the optimizer's preconditioned step, not
any one-shot exact solve. It exists only as the integral of the coupled dynamics.**
The one constructive corner left untested is the honest one: a multi-step
exact-solve process at gradient descent's own stride (~0.03% of bits per epoch,
re-solved every stride from an infeasible fitting start) — that is, simulating the
process — which is the thesis, stated constructively. Caveats: one realization per
row (edge-location variance itself measured across two), one d, one load; the
instantaneous directions are linear extrapolations by construction — which is
precisely what makes their failure informative.

## Why this matters for the challenge

The post frames its construction and the trained model as differing in *which* neurons
carry the signal — "our hand-coded algorithm stores facts as patterns of inactive
neurons, trained models seem to use patterns of active neurons instead" (Figure 8). The
measurements above say the more consequential difference is elsewhere: **near capacity
the trained model is not using patterns at all.**

That reframes the capacity gap. A pattern code over `d` neurons has at most `d` bits per
fact to work with, and the authors' construction spends them well — it is simply playing a
game the trained model has stopped playing by the time it is loaded up. This predicts,
correctly, that a graded construction should scale better: `linsolve` stores 11–13× more
than the silence code and overtakes the trained model above d≈60.

It also suggests the authors' judgement that Appendix-A-style value codes are "probably a
bit of a dead end" for reverse-engineering may be too quick. Value codes look unlike
trained models *at low load*, which is where weights get inspected most easily — but the
loaded regime is the one that resembles them. The open question is not pattern-vs-value;
it is what graded code gradient descent picks, given that it picks one.

The robustness measurements (§5) add the counterweight, and it lands in the authors'
favor: their instinct to deprioritize Dugan-style exact-solve constructions as "very
unlike something models trained with gradient descent would or could learn" is
measurably correct. The exact-solve value codes sit three orders of magnitude outside
the robustness class that every gradient-descent product occupies — and the authors' own
construction sits *inside* that class. Read §§1–7 together and the two axes separate:
the **coding scheme** worth pursuing is graded/magnitude, because that is what loaded
trained models actually use (§1–3); the **solution style** worth pursuing is
noise-tolerant, because that is the game every optimizer-built solution plays (§5) and
the readout freedom that buys it is where trained models spend their `d²` unembedding
parameters (§7). The benchmark as stated sees only the first axis. A capacity metric
that also charged for fragility — max facts at acc ≥ 0.9 under weight noise of one
optimizer step (absolute σ ≈ lr), or simply reporting σ90 alongside max_facts — would
close the loophole the value codes exploit and make the challenge track the question the
post actually asks: what fact storage looks like when gradient descent builds it.

## Caveats

* Probes are trained linear readouts; a probe failing bounds what a *linear* readout can
  extract, not what information is present. Where a construction has a known decode
  (linsolve), the analytic answer is used instead.
* Quantization is uniform between the min and max of the positive activations. A
  quantizer matched to the activation distribution would likely need fewer levels, so
  "~32 levels" is an upper bound on what the trained model requires.
* One seed per cell. The trend is monotone and consistent across three model sizes, but
  the individual numbers are single measurements.
* Robustness numbers (§5) average 20 noise draws per level but use one base model per
  cell, and the swept noise grid floors at σ = 1e-5 — cells reported "≤1.0e-5" are
  pinned at that floor, so the ≥1200× d=64 gap is a lower bound. The trained baselines
  are snapshotted at the first epoch reaching 100%; trained past that point they only
  get more robust. The d=64 linsolve cell rebuilt at 0.99 clean accuracy rather than
  1.000 (the capacity search's winning seed was not recovered); its σ90 is at the grid
  floor either way.
* "One Adam step ≈ lr per weight" is the steady-state Adam step magnitude, a heuristic
  rather than a measured per-step displacement.
* Fact-seed dependence spot-checked at d=32 (seeds 43, 44): both `twosided` and
  `trained` land within a couple percent of their seed-42 acc=1 boundaries (0.999+ at
  the reported capacities, clear failures ~4–6% above), so capacity ratios are robust
  to the fact sample; the exact integers are seed-42-specific.
