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

These statements are made exact in `docs/theory.md`: flips are free along *any*
continuous path (Lemma 3), a straight segment to any feasible realizer crosses in
exactly Hamming-many flips (Lemma 4) — so crossing directions exist in abundance,
and this section's content is precisely that no locally computable oracle among the
natural candidates produces one. By Lemma 4, naming a crossing direction is
equivalent to naming a feasible realizable pattern near P₀ — the construction
problem itself. Theorem 2 there also characterizes realizability outright (per
column: no alternating strict cycle; total case: staircase/Ferrers) with a counting
corollary — fewer than 2^(−28000) of patterns are realizable at this scale — which
is why all interventions here move in embedding space, never on bits.

## 19. The stride process crosses: iteration, not the gradient, is the ingredient

`probe_flippolicy.py --phase stride`. The one door §18 left open: simulate the
process itself — an exact-solve direction oracle re-solved *every step* at gradient
descent's own stride (~0.5% of bits per full-batch step), from the infeasible
epoch-180 fitting state. Two runs, and the first is part of the result:

**Max-min refits are degenerate off the feasible set — observed.** With the §15
machinery unchanged (spread emb-LP, but max-min readout refit), the process
destroys itself in eight rounds (train accuracy 0.867 → 0.01): at an infeasible
state the max-min readout LP's optimum is the do-nothing W (`docs/theory.md` Prop
1a), the logits zero out, and the next emb-LP is contentless. Fit pressure is
needed in *both* blocks: the readout refit was replaced by the capped-sum
(spread) objective — `readout_lp_spread` — and nothing else changed.

**With fit pressure in both blocks, the process crosses and keeps building:**

| round | train acc | net drift | pattern ceiling σ90 |
|---|---|---|---|
| 0 (= epoch-180 state) | 0.867 | 0 | infeasible |
| **4** | 0.949 | 0.87% | **1.45e-2 — crossed** |
| 20 | 0.971 | 1.78% | 1.69e-2 |
| 40 | 0.983 | 2.57% | **1.80e-2** |

Four rounds — 0.87% of bits, the §17 edge scale — take the pattern from
storage-infeasible to a ceiling at gradient descent's own epoch-200 level
(1.39–1.44e-2), and forty rounds reach **1.80e-2 with monotone improvement
throughout**: the first exact-solve iteration in this program that consolidates
instead of churning. The gate it builds is 5.7× the previous constructed record
(3.18e-3, §15) and within **~2.4×** of the trained full-budget ceiling (4.40e-2).
Train accuracy climbs the whole way (0.867 → 0.983) — also a first: every previous
exact-solve iteration degraded the model it touched.

What this overturns, and what it doesn't. §18's strongest reading — the crossing
information lives in the loss gradient field and nothing coarser — is refuted: a
fit-pressure LP oracle, re-solved each stride, integrates to a crossing path with
no gradients anywhere. What stands, sharpened: *one-shot fails, iteration
succeeds* — the same spread LP whose single budget-matched step failed in §18
crosses in four re-solved strides. The ingredient is the re-solve loop — direction
recomputed after each step's flips — not the specific field being integrated. GD
at matched train accuracy is still ~2× ahead in ceiling (its acc-0.98 gate sits at
~3.5e-2), so the gradient field remains the better integrand; but the *class* of
processes that build gate quality just widened from "gradient descent" to
"any sufficiently fine re-solved fit-pressure dynamics."

Attribution caveats, explicitly: relative to §15's collapsing drift, this run
changed three things at once — the seed (infeasible fitting state, not a feasible
constructed gate), the stride (0.5%, not 2%), and the readout refit (spread, not
max-min). Which are necessary is unmeasured (the natural ablations: stride at 2%;
stride seeded from the digit gate). And the seed is a *gradient-descent* state:
epochs 0–180 of training produced it. §16 says those epochs build no gate quality
— the epoch-180 pattern is still infeasible — but they do build the 87%-fit
embedding structure the process starts from. **The live constructive question is
therefore no longer "can exact solves build the gate" — they can, from the right
start — but "can anything cheaper than gradient descent produce the start":
whether the stride process runs from a ridge-style 87%-fit state, or from scratch.**
Caveats as ever: one seed family, one d, one load.

## 20. The seed problem: the flow needs half a fit and a soft geometry

`probe_flippolicy.py --phase stride --stride-seed {scratch,epN,digit}`. §19's process
starts from a gradient-descent state, so it closes nothing constructively (a recipe
containing GD is GD with a post-processor). This section asks what the flow actually
requires of its seed, two ways: how much GD prefix (an analysis question — localizing
what remains un-understood), and whether a *constructed* seed serves (the
constructive question proper).

**The prefix sweep.** The same LP stride process seeded along the training
trajectory:

| seed | train acc | outcome |
|---|---|---|
| scratch (GD's own init) | 0.03 | degenerate by round 2, LP dead by round 4 |
| epoch 50 | 0.34 | dead by round 32; 14% of bits churned for nothing |
| epoch 100 | 0.58 | **crosses** ~round 75, ceiling 1.51e-2 at round 120 |
| epoch 150 | 0.77 | **crosses** ~round 25, ceiling 1.52e-2 at round 60 |
| epoch 180 (§19) | 0.87 | crosses in 4 rounds, 1.80e-2 at round 40 |

The scratch and ep50 deaths are §14 mechanized: a far-from-feasible pattern gives
fit pressure nothing to push on (the LP optimum through the mask is ≈ the
do-nothing point), and stepping toward a contentless target is churn. The bootstrap
threshold sits in **GD-epochs (50, 100]** — train accuracy between 0.34 and 0.58.
Everything after half-fit, *including the entire gate transition*, belongs to the
generic process class; gradient descent's irreplaceable contribution is at most the
first ~100 of 355 fitting epochs. Notably the ep100 and ep150 flows crossed on
their own, well outside §16's window — they did not replay gradient descent's
crossing; they built their own.

**The constructed seed refuses the cheap flows.** The pedestal-optimized digit gate
(ridge solves only, feasible at ceiling 2.85e-3) was given three versions of the
process. The first-order oracles fail in an instructive order: the fully-cheap
variant that works from the GD seed *destroys* the digit seed in one round; the
hybrid (matvec direction + exact readout refit) holds live accuracy at 1.0 for 60
rounds but its ceiling dips to 1.86e-3 and crawls back only to 2.47e-3 — sixty
rounds for a net loss. The exact-LP version builds, but at a crawl: the same dip,
then a slow climb past the seed to **3.15e-3 at round 50** (≈1.1× the seed;
plateauing by round 60 at 3.10e-3) — a real gain, and the best GD-free artifact in
the repo, yet ~6× short of what the *same flow* achieves from soft GD seeds. Stiff
geometry does not refuse the exact flow; it throttles it. The mechanism is
measurable and is the section's second finding:

**Softness.** The GD fitting state keeps 0.5% of its pre-activations within ~0.001
of zero (|pre| quantiles at epoch 180: 0.5% = 0.0011, median 0.98); the digit
construction's same quantile is ~12× farther out (0.0137, median 2.45) — the
pedestal exists precisely to stiffen signs. The stride step caps *flips*, not
travel: on soft geometry the flip quota is met with tiny motion and any decent
direction survives; on stiff geometry the same quota forces excursions an order of
magnitude longer, which only exact targets (if anything) survive. **Gradient
descent does not just build a good gate — it maintains a near-tie reservoir that
makes its own geometry cheaply adaptable.** §17's flip policy seen from the other
side, and the first measurable property separating GD-built from constructed
embedding geometry.

The honest constructive ledger, restated per the distinction this program now
insists on: the *construction record* (closed-form-ish, rules-legal in spirit)
remains the ridge-built digit gate at ceiling 2.85e-3; the best fully GD-free
artifact is that seed plus the exact-LP flow at 3.15e-3. Everything from §19 on is
*process-class analysis*, not entries: hand-specified dynamics closing the gap is
the finding, not the recipe the challenge asked for — and the 5–7× between the best
GD-free artifact and the soft-seeded flows' 1.5–2.3e-2 is now the measured value of
the soft geometry that only gradient descent's early fitting has been shown to
build.

## 21. The oracle ablation: granularity is the ingredient, exactness is not

`probe_flippolicy.py --phase stride --oracle {lp,fw,fw-full}`. §19 used exact LPs
as the direction oracle. Replacing them, from the same edge seed, identical step
mechanics throughout:

| oracle (emb / readout) | cost per round | ceiling |
|---|---|---|
| exact LP / exact LP (§19) | ~45 s | 1.80e-2 at round 40 |
| **matvec vertex / exact LP** | ~15 s | **2.16e-2 at round 40, plateau ~2.2e-2, peak 2.33e-2** |
| matvec vertex / 0.2% vertex step | **~10 ms** | 1.66–1.76e-2 plateau (400 rounds) |
| matvec vertex / 5% vertex step | ~10 ms | destroyed in one round |

The matvec oracle is the box-LP solution of the *linearized* capped-sum objective —
sign-snapping a subgradient, no solver. Three attributions fall out. (1) **Step
magnitude is the requirement**: the identical process at a 5% readout step is a
demolition; at 0.2% it matches the exact-LP process from a milliseconds-per-round
loop. (2) **LP-exactness of the direction is not**: the first-order direction beats
the LP direction outright (2.2e-2 vs 1.8e-2) — aiming at a distant LP vertex is
worse than following the local pull, §18's lesson recurring inside the successful
regime. (3) The exact readout refit is worth ~1.3× over the cheap nudge (2.2e-2 vs
1.7e-2). The fully-first-order arm is, bluntly, gradient descent with a hinge
objective and an order-statistic step rule — and that bluntness is the finding:
nothing about gradient descent's specifics (cross-entropy, softmax, Adam,
backprop-exact gradients) is load-bearing for gate quality on soft geometry.
Iterated fine-grained fit pressure is the whole mechanism. What remains
unexplained: from the same seed, gradient descent's own continuation reaches
4.06e-2 — a ~1.8× integrand advantage over the best hand-specified flow, the
process class's one remaining mystery. Caveats as ever: one seed family, one d,
one load — since narrowed by the replication guard (§24), which confirms
the headline rows at seeds 43–44 and the feasibility split at d=64.

## 22. The seed problem falls: half a fit and a GD-scale readout, no gradient descent

`probe_flippolicy.py --phase softseed`, then `--phase stride --stride-seed
soft:TAG --oracle {fw-full,fw}`. §20 left the constructive question at "the flow
needs a GD state past half-fit, and constructed geometry refuses the cheap
oracles" — with softness (the near-tie reservoir) as the candidate separating
property. Both halves of that verdict are now revised by construction.

**The seed** (`handcode/softseed.py`) is the most boring rules-legal recipe that
could work: random init at trained density, then freeze-the-pattern rounds of
ridge fits — readout to one-hot targets, per-token embedding sweeps —
under twosided's flip-capped step. Fit plateaus at train accuracy **0.63**
(above the ep100 state's 0.58, which bootstrapped in §20) and no knob in four
orders of magnitude of ridge strength moves the near-tie profile: every config
sits at q0.5%/median ≈ 6–9e-3, the Gaussian-init value and the digit
construction's class. Two post-passes complete the seed, both ridge-only and
both leaving fit, pattern, and each other untouched: **soften** compresses the
smallest-|pre| band toward zero by per-column solves until the reservoir matches
GD's (1.10e-3 vs the edge state's 1.09e-3; 0–1 of 50,688 pattern bits flipped),
and **w_rms** rescales the readout to GD scale. The sweep's product is a 2×2 of
seeds — fit {0.52, 0.63} × reservoir {stiff ~7e-3, GD-matched 1.1e-3} — with no
gradient descent anywhere in the ancestry.

**The readout scale is a regime, not a gauge.** The raw ridge readout comes out
19× smaller in rms than the ep180 state's (0.067 vs 1.277), which puts *zero*
facts' logit gaps above the flow's saturation cap τ = 0.5, where the ep180 state
has 74% above. Below τ every already-fit fact keeps pulling on the oracle —
§14's margin-churn regime — and measured, it is fatal: without `w_rms`, all six
seeds collapse under fw-full in one round (acc 0.52–0.63 → 0.22–0.30) and churn
indefinitely. With the readout at GD scale, every seed climbs. **What §20 read
as stiff embedding geometry defeating cheap oracles was, at least for
ridge-family seeds, the readout regime**: fit facts that never saturate drown
the direction in margin pressure.

**Softness buys stability margin, not crossing.** Under fw-full (cheap readout
nudges), both high-fit seeds cross into storage-feasibility around round 100 at
ceiling ~1e-2 — and both eventually lose it; the softened twin holds its
feasible window longer (rounds 100–200 vs 100–150) and peaks higher (1.13e-2 vs
1.01e-2). Under fw (exact readout refit), the twins are indistinguishable. The
near-tie reservoir is a second-order stabilizer, not the gatekeeper §20
conjectured.

**The headline: the fw flow consolidates from the constructed seed.** Matvec
direction + exact readout refit, from the 0.63-fit ridge seed, either twin:

| round | ceiling σ90 (stiff / softened) |
|---|---|
| 0 | infeasible |
| 25 | 1.45e-2 / 1.46e-2 — **crossed** |
| 100 | 1.95e-2 / 1.99e-2 (live accuracy 1.000 from here on) |
| 200 | 2.31e-2 / 2.40e-2 |
| 300 | 2.92e-2 / 2.64e-2 |
| 400 | **3.00e-2 / 3.21e-2**, ~10% net drift |
| 800 | — / **3.46e-2**, decelerating but still creeping (softened arm) |

The crossing value equals GD's own crossing gate (§19: 1.45e-2); the GD-seeded
fw numbers (§21: plateau ~2.2e-2, peak 2.33e-2) are passed by round 250–300.
**Gradient descent is not needed to make the start.** The best fully GD-free
artifact moves 3.15e-3 → **3.46e-2** — from ~14× short of the trained
full-budget ceiling (4.40e-2) to **~1.27×** — and the pipeline's every stage is
understood: ridge seed, subgradient direction, order-statistic step, LP readout
refit. Two corollaries: §21's "~1.8× integrand edge" was partly round count —
the flow keeps building long after the horizons those comparisons used — but
not entirely: at *matched* iteration budget gradient descent holds 4.28e-2
(epoch 750, §16's curve) against the flow's 3.46e-2 at round 800, a bounded
**~1.24× matched-budget edge**, shrinking slowly (+8% over rounds 400–800) with
no sign yet of closing outright. And the fit floor is looser than §20's
GD-prefix threshold suggested: the 0.52-fit seed also crosses (round ~100) and
reaches 1.8–1.9e-2 by round 200 on the same monotone climb.

**The scratch cell fails, so the fit requirement is real.** The remaining
suspicion — that §20's scratch death was also a readout artifact — was
retested under the revised regime: a rounds-0 seed (random embeddings, ridge
readout at GD scale, train accuracy 0.126). It does not bootstrap. fw-full
climbs 0.13 → 0.51 over 400 rounds without ever entering feasibility; the fw
arm is worse — the exact readout refit on a contentless pattern is destructive
(live accuracy pinned at ~0.03 for a hundred rounds, 43% churn by round 200,
infeasible throughout), §20's death mechanism reproduced. The flow's fit floor
sits between 0.13 and 0.52, the embedding half-fit is a genuine ingredient the
flow cannot make from nothing at tested budgets — and the point of this
section stands precisely because it is constructible by ridge. (One honest
hedge: at matched step counts, Adam fits from scratch far faster than the fw
flow, so "never" is untested beyond 400 rounds; what is established is that a
ridge half-fit is the cheap constructible substitute for whatever the early
fitting phase builds.)

The ledger, unchanged in kind: the flow optimizes the task objective, so this
remains **process-class analysis, not a challenge entry** — iterating solves
against a coder-declared equality system (twosided, digit) is construction;
iterating them against the task's own margins is training with different
branding. The *construction record* stands at 2.85e-3 (§14). What the result
changes is the residue: after §21 removed GD's optimizer specifics and this
section removes its seed, the unexplained remainder is a bounded ~1.2–1.3×
matched-budget ceiling edge (4.28e-2 at epoch 750 vs 3.46e-2 at round 800),
shrinking slowly with flow rounds — whether it vanishes at very large budgets
is open — plus the fact that the whole account is a process, not a
description. Caveats: one fact seed, one d, one ridge
init, and the fw-full/fw contrast measured on one seed family; the
replication guard (§24) covers §§14–21, not yet this section's rows.

## 23. Order-structure statistics are as blind as magnitude statistics

`probe_ordering.py`. Theorem 2 parameterizes every additive gate column as a
token ordering plus row thresholds, so "gate quality is declaratively
describable" (theory.md problem 6″) has a natural test bed: statistics of the
orderings and thresholds alone, invariant to monotone value changes — the
parameterization §14's refuted magnitude statistics never touched. The zoo now
contains the sharpest possible contrast: the ridge seed (infeasible, ceiling 0)
and the fw product grown from it (2.31e-2), 7.6% of bits apart, same ancestry
class throughout.

Ten statistics — rank-margin softness and quantiles, fact-boundary attention,
cross-column ordering diversity (Kendall), interleaving runs z, threshold
dispersion, per-fact active degree, same-label pattern correlation excess, and
the label share of rank-margin variance — and the result is uniform:

* **No statistic separates good from bad within matched provenance.** The seed
  (0) and its fw product (2.31e-2) agree to within noise on everything (rank
  softness 0.065/0.061, fact attention +0.53/+0.58, ordering diversity
  0.092/0.089, label-R² 0.034/0.035). ep180 (0) vs trained (4.40e-2): same.
* **What does separate is provenance.** The runs statistic detects GD-trained
  states (including the worthless ep180); label-correlation excess and label-R²
  detect fit pressure of any kind (including the worthless seed). The seed row
  is the control that kills them as quality metrics: it has the "good" label
  structure and a ceiling of zero.
* **The building delta is structureless too.** The 3,845 flipped cells between
  seed and product sit at the staircase boundary (median rank distance 3 vs 18
  for the population — the order-space restatement of §17's free flips), spread
  uniformly over columns (top-4 share 0.16 vs 0.125 uniform), labels (CV 0.10),
  and tokens, error-agnostic (35% on then-wrong facts vs a 37% base rate). The
  flow's flip policy is gradient descent's flip policy — §17 replicated for a
  GD-free process — and the choice of *which* boundary cells to flip, where all
  the quality lives, has no marginal signature a construction could imitate.

One second-order statistic was also tested — per-fact near-boundary *exposure*
(how many of a fact's d cells sit within 2 rank steps of the staircase edge),
the natural joint quantity if good gates protected facts by anti-concentrating
fragile bits. Every gate in the zoo, good and bad alike, matches the
independent-assignment null CV to two decimals (trained 0.51 vs null 0.53;
seed 0.55 vs 0.55; fw product 0.55 vs 0.56): no gate structures its fragile-bit
placement at all.

First pass over the full Theorem-2 parameterization, refuted at the same depth
as §14: gate quality is invisible to state statistics in either magnitude or
order coordinates, invisible to the delta's marginals, and invisible to the
first joint statistic with a robustness story. With §18's Remark 4.1 (naming a
crossing direction is solving the construction problem), the incompressibility
position (theory.md 6‴) now rests on three independent legs.

## 24. Replication guard: §§14–21 across three fact seeds and two dimensions

Everything in §§14–21 was measured in one cell — d=32, fact seed 42, n=1584,
one training trajectory — and every one of those sections carries the caveat
"one seed, one d, one load". This section is the guard those caveats asked for:
the headline rows re-measured at **fact seeds 43 and 44** (d=32) and, for the
rows whose cost permits it, at **d=64**.

Two things had to happen before any of it could run. The probes had no `--d`,
`--n-facts`, or fact-seed flags — those were module constants, so *no
replication was runnable as-is*; they are now flags, with the default cell
keeping the original paths so published caches still resolve. And
`ProcessPoolExecutor` uses **spawn**, so worker processes re-import the probe
module and see the *defaults* rather than the configured cell: without an
explicit pool `initializer`, every ascent silently measures the d=32/seed-42
gate and files it under the new label — a replication that always succeeds
because it never changed cell. Both pools now pass the cell explicitly.

**The protocol gate.** Under the patched code the published cell reproduces
*exactly* — σ90 ratio 1.0000 on all four gates (trained 4.39564e-2, digit
2.84530e-3, random and init infeasible). Every number below therefore comes
from code verified against the original.

### The headline rows

| row | published (d=32, s42) | d=32 seed 43 | d=32 seed 44 | d=64 seed 42 | verdict |
|---|---|---|---|---|---|
| §14 trained gate ceiling | 4.40e-2 | **4.41e-2** | **4.36e-2** | **3.17e-2** | reproduces |
| §14 digit m=2 ceiling | 2.85e-3 | **2.52e-3** | **2.81e-3** | not run | reproduces |
| §14 random additive | infeasible | **infeasible** | **infeasible** | **infeasible** | reproduces |
| §14 init | infeasible | **infeasible** | **infeasible** | **infeasible** | reproduces |
| §19 stride monotone climb | → 1.80e-2 @ r40 | **→ 1.66e-2** | **→ 1.95e-2** | — | reproduces |
| §20 ep50 seed | dies | **dies** | **dies** | — | reproduces |
| §20 ep100 seed | crosses ~r75 | **crosses r104** | **crosses r92** | — | reproduces, slower |
| §20 ep150 seed | crosses ~r25 | **crosses ~r36** | — | — | reproduces, slower |
| §21 matvec ≥ exact LP | 2.16e-2 vs 1.80e-2 (1.20×) | **2.11e-2 vs 1.66e-2 (1.27×)** | — | — | reproduces |
| §21 0.2% readout step | 1.66–1.76e-2 plateau | **1.70–1.76e-2 plateau** | — | — | reproduces |
| §21 5% readout step | destroyed in one round | **destroyed r1, still infeasible at r400** | — | — | reproduces, strengthened |
| §20 softness, GD side | 0.5% quantile 0.0011 | **0.0010** | — | — | reproduces |

The trained-gate ceiling is the tightest: **4.40 / 4.41 / 4.36e-2** across three
independent fact seeds — a spread under 1%. §19's stride endpoint brackets the
published value rather than merely approaching it (1.66e-2 and 1.95e-2 against
1.80e-2), and §21's matvec arm lands at 2.11e-2 against 2.16e-2. The digit ceiling is looser
(2.85 / 2.52 / 2.81e-3, ±12%) but never approaches the trained value, so the
**~15× trained-vs-constructed separation that §14 exists to report is a
property of the problem, not of seed 42.**

The feasibility split is the most robust result in the whole program: the
random and init gates are infeasible outright — LP γ\* = 0, the do-nothing
point of Prop 1(a) — in **every cell tested, including d=64**, while the
trained gate at the same load is feasible with a large margin. That is §14's
central claim and it survives a change of both seed and dimension.

### §17's dense curve reproduces almost exactly

| epoch | 140–180 | 200 | 220 | 240 | 260 | 280 |
|---|---|---|---|---|---|---|
| seed 42 (published) | infeasible | 1.39e-2 | 2.0e-2 | 2.8e-2 | 3.4e-2 | 3.7e-2 |
| **seed 43** | **infeasible** | **1.51e-2** | **2.08e-2** | **3.05e-2** | **3.54e-2** | **3.73e-2** |
| ratio | — | 1.09 | 1.04 | 1.09 | 1.04 | 1.01 |

Infeasible through epoch 180, feasible by 200, then a smooth climb — reaching
4.09e-2 by epoch 360 against the trained gate's own 4.41e-2. §16's "feasibility
is won in a ~50-epoch window at the end of fitting" holds at a second seed.

### What does *not* transfer: the epoch-180 seed

§19's headline is "the **infeasible** epoch-180 seed crosses in ~4 rounds". At
seeds 43 and 44 **the epoch-180 state is already feasible** (ceiling 7.6e-3),
so the crossing claim cannot be tested from it — what reproduces there is the
monotone climb, not the crossing.

This is not a contradiction; it sharpens §18. The edge's *location* drifts
between realizations, and we now have a cleaner measurement of how much: within
**one** fact seed (43), two independently trained micro-realizations disagree
about epoch 180 — `window_checkpoints` has it infeasible, `edge_state` has it
feasible at 7.6e-3. Torch CPU training is not bit-deterministic (§17), so
"epoch 180" does not name a fixed state even at fixed seed. **The epoch number
is not the object; the distance to the edge is.** Anyone re-running §19 at a
new cell must locate the edge first rather than reuse epoch 180 — at seed 43
the analogous infeasible seed is epoch 150, which does cross (round ~36).

Same warning at d=64, more strongly: epoch 180 there is train accuracy 0.698,
against 0.867 at d=32. The epoch ladder in §20 is defined by position relative
to the edge, and transplanting epoch numbers across d replicates a different
object.

**Run from seed 43's actual edge, the crossing reproduces.** Seeding the stride
process at epoch 150 — infeasible at seed 43 — gives §19's result directly:

| round | 0–32 | 36 | 40 | 44 | 52 | 56 | 60 |
|---|---|---|---|---|---|---|---|
| ceiling σ90 | **infeasible** | **4.40e-3** | 6.13e-3 | 6.47e-3 | 7.78e-3 | 9.76e-3 | **1.05e-2** |

Infeasible for 32 rounds, then a crossing and a monotone climb, with live train
accuracy rising 0.824 → 0.881 and net drift only 3.7%. Published ep150 crosses
~round 25 to 1.52e-2 at round 60; seed 43 crosses at round 36 to 1.05e-2. The
*phenomenon* — an exact-solve iteration taking an infeasible pattern across the
storability edge and then improving monotonically, which is the whole content
of §19 — reproduces; the rate does not transfer precisely. (Note the ascent
oscillates before crossing: accuracy at r20 and r28 reads 0.837/0.847 against
0.032 at r24/r32, all still γ\* = 0. Only after r36 is it stable at 1.000.)

### §20's bootstrap bracket holds at both new seeds — only the rate moves

§20 brackets the bootstrap threshold in GD-epochs at **(50, 100]**: below it the
stride flow dies on contentless fit pressure, above it the flow crosses. Both
edges reproduce at both new seeds.

| ep100 stride | r0–r88 | r92 | r100 | r104 | r112 | r120 |
|---|---|---|---|---|---|---|
| seed 42 (published) | crosses ~r75 | | | | | |
| **seed 43** | **infeasible** (through r100) | — | — | **2.27e-3** | 5.41e-3 | **6.84e-3** |
| **seed 44** | **infeasible** | **3.19e-3** | 5.20e-3 | 6.02e-3 | 7.90e-3 | **1.11e-2** |

Both cross and both climb monotonically afterwards, 1.2–1.4× later than the
published ~round 75. The bracket's *internal* ordering survives as well: within
seed 43 the state nearer the edge crosses sooner — ep150 at r36, ep100 at r104 —
which is the ordering §20's story predicts. So the transferring content is the
bracket and its structure; what does not transfer is the number of stride rounds
the bootstrap costs, the same rate-not-mechanism pattern as §19's crossing.

The lower edge is unambiguous: at ep50 live train accuracy collapses 0.33 → 0.02
by round 1 and every ascent is infeasible, at both seeds.

### A new result: near-tie-ness is *necessary*, not incidental

§17 measured that gradient descent flips only extreme near-ties (median
|pre-activation| 0.002–0.03 against 0.85–1.6 for the population) and concluded
"the flip policy is trivial — cross whatever is nearest zero along the motion."
The converse was never tested: is near-tie-ness *required*, or would any small
flip set do? Take the trained gate and flip 0.5% of its bits — the stride's own
per-round quota — two ways:

| perturbation of the trained gate (0.5% of bits, d=32) | ceiling σ90 | accuracy |
|---|---|---|
| none (baseline) | 4.3956e-2 | 1.000 |
| the 0.5% of bits **closest to zero** | **4.4008e-2** | **1.000** |
| 0.5% of bits **at random**, draw 1 | **infeasible** | 0.032 |
| 0.5% of bits **at random**, draw 2 | **infeasible** | 0.032 |
| 0.5% of bits **at random**, draw 3 | **infeasible** | 0.032 |

Repeated at fact seed 43: baseline 4.4097e-2, near-tie flips **4.3987e-2**
(intact), three random draws **all infeasible**. Six random draws across two
seeds, six destructions; two near-tie perturbations, two survivals.

At matched flip count, near-tie flips leave the gate *entirely intact* — the
ceiling moves by 0.1%, within ascent noise — while random flips of the same
size destroy feasibility outright, three draws for three. So the near-tie
restriction is not a description of what gradient descent happens to do; it is
a **necessary condition** for a flip set that size to preserve storability.
This tightens §17's "the flip policy is trivial": the policy is trivial to
*state*, but the set it selects is a vanishing and highly non-generic subset —
consistent with `docs/theory.md`'s counting corollary that realizable patterns
are exponentially rare. (Found by accident: this was the perturbation built to
test LP warm starts, which failed for unrelated reasons.)

### Cost, and why d=64 is only partly covered

LP solve time is ~99% of runtime. At d=32 one stride round is 31.4s and one
`ascend_best` ~0.9 min (high power mode; the ~45s/round in §21 only reproduces
under low power). At d=64, n=6336 the ascent LP is 481,536 rows and a single
stride emb-LP ran **>18 minutes without converging**, overrunning its 900s
`time_limit` — so the d=64 rows here are the ones reachable by ascent alone
(§14's split), and the stride program (§§19–21) is not attempted at d=64. A
full d=64 stride replication is >130 h under the current formulation.

Ten optimizations were measured and nine rejected; the details are in the
handoff. The load-bearing negative: HiGHS's IPM cannot use more than one core
here by any route (scipy silently ignores `threads`; a direct `highspy` build
refuses to run above one), so a single LP is unparallelizable, and concurrent
LP processes contend for memory bandwidth rather than scaling with cores.
Notably `--ipm-tol`'s claimed 2–4× does not reproduce (measured 1.02×), and
cutting planes on the pattern-consistency rows — the natural idea, since only
2–3% of them bind — buys nothing, because those rows carry 2 nonzeros each and
are 13% of the nnz despite being 73% of the rows.

### Caveats on this section

* n at d=64 was set to 6336, holding the load *fraction* at 38.7% of `4d²`.
  n=1584 at d=32 is the digit code's measured capacity point; the d=64
  equivalent was not re-measured, so the d=64 column is load-fraction-matched,
  not capacity-matched.
* The §21 `fw-full` arms use `--snap-every 20` rather than 4, cutting 101
  ascents to 21. This changes ceiling-curve *sampling density*, not any
  measured quantity, but it is a deviation from the published protocol.
* The 5% readout step was run twice at seed 43: 4 rounds
  (`flippolicy_s43_fwfull5pct.json`) and 400
  (`..._long.json`). The long arm is the informative one — the gate is
  destroyed in round 1 and is **still infeasible at round 400**, live accuracy
  flat at ~0.33 throughout. So the 5% step does not merely cost a slow
  recovery; within the horizon measured there is no recovery at all, which is a
  stronger statement than §21's published "destroyed in one round".
* The two ep50 arms rest on unequal evidence, and the gap cannot be closed under
  this formulation. Seed 43 ran 44 rounds with 12 ascents, all infeasible. Seed
  44 stops at **3 rounds and 1 ascent**: the spread LP's objective collapses to
  `mean_m = -0.0000` by round 2, and at round 4 the embedding LP exhausts its
  900s `time_limit` without converging, which breaks the stride loop. Attempted
  twice, same round, same message — so this is a property of the collapsed state
  and not an interrupted run. "Dies" at seed 44 therefore rests on the round-1
  accuracy collapse (0.33 → 0.027) plus one infeasible ascent, where seed 43 has
  twelve. Arguably the LP's own failure corroborates death — there is no
  feasible spread direction left to find — but that is an argument, not the
  matched measurement, and it is recorded here as the weaker cell it is.
* §§15, 16, 18 were not re-measured; only the rows named as headline claims
  were.

## 25. Rate–distortion: the ceiling's description is ten kilobits with no structure

`probe_ratedistortion.py`. §§14 and 23 asked whether gate quality is
*describable* and answered no, statistic by statistic. This section asks the
quantitative version instead: compress the trained solution's *description*
and measure what each compressed description still buys. The generator is the
pre-activation embedding pair (u, v) — 4,096 scores, 131,072 bits at float32
— and each variant's induced pattern is handed to §14's two-LP ascent
(3 rounds, started from the variant's own embeddings with the trained
readout), so the weights are re-solved from the pattern and the measurement
isolates what the *description* names. The baseline reproduces the protocol
gate exactly (ceiling 4.40e-2, first emb-LP γ = 11.593, §24's value).
Description lengths are entropy-coded level streams plus the codebook.

**Per-parameter quantization keeps almost everything:**

| levels k | bits/param (quantile) | flips | ceiling | bits/param (linear) | flips | ceiling |
|---|---|---|---|---|---|---|
| 2 | 1.01 | 23.9% | infeasible | 1.02 | 22.6% | infeasible |
| 3 | 1.59 | 15.4% | **4.28e-3** | 1.12 | 22.9% | infeasible |
| 4 | 2.01 | 10.3% | 2.51e-2 | 1.54 | 17.0% | 5.00e-3 |
| 6 | 2.58 | 7.2% | 3.87e-2 | 2.15 | 9.7% | 3.27e-2 |
| 8 | 3.04 | 4.7% | 4.23e-2 | 2.63 | 6.8% | 3.92e-2 |
| 16 | 4.10 | 3.5% | 4.35e-2 | 3.76 | 3.1% | 4.36e-2 |
| 32 | 5.23 | 2.1% | 4.37e-2 | 4.93 | 2.6% | 4.39e-2 |

The feasibility edge sits between 1.0 and 1.6 bits/param. A three-level
alphabet on the 4,096 scores — ~6.5 kilobits, a 20× compression — still names
a storable pattern **above the construction record** (4.28e-3 vs §14's
2.85e-3), and 3 bits/param recovers 96% of the trained ceiling. (These are
compressed *trained* artifacts, not constructions — the record stands; what
they measure is description length, not buildability.)

**Every structural compression fails, at matched or better distortion.** The
same trained state, compressed along every cross-parameter axis available:
SVD truncation of the token matrix is infeasible at every rank tried up to
r=24 of 32 (9.4% flips); k-means token codebooks are infeasible at every size
up to c=64 for 128 tokens — merely pairing tokens up (18.2% flips); zeroing
the smallest half of the weights is infeasible (14.1% flips). Flip fraction
does not predict survival — quantile k=4 survives 10.3% flips where r=24 dies
at 9.4%, and linear k=4 survives 17.0% where c=64 dies at 18.2%. What
predicts it is the perturbation's *direction*: per-parameter rounding bounds
each token's score error and concentrates flips on near-ties; every
structural projection moves scores incoherently. The flip-curve control makes
the geometry explicit — flipping the q fraction of pattern bits nearest zero
leaves the ceiling intact through q=4%, degrades it to 3.19e-2 at 8%, and
kills it by 16%, while random flips of matched count kill it at 0.5% and 2%
(§24's necessity result, now a full curve: the near-tie direction is ≥16×
more tolerant).

So the trained solution's ceiling-relevant content is compressible along
exactly one axis — per-scalar precision — and along none of the structural
axes tried: **4,096 scores × ~2.5 irreplaceable bits each ≈ 10 kilobits of
co-adaptation with no found redundancy across parameters.** That is
theory.md 6‴ as a measurement rather than a suspicion. One suggestive
alignment, offered with its caveat: the fact table itself carries 1584 ×
log₂32 ≈ 7.9 kilobits of label information, and the pattern's feasibility
edge lands at 6.5–8.2 kilobits — but the ascent receives the labels, so the
pattern need not encode them; the near-match invites a counting argument, it
does not yet make one.

Two side measurements. As a *deployed model* (no re-solve), the same weights
need ~5 bits/param — k=32 keeps σ90 within noise, k=16 is broken — matching
§1's activation-side "~32 levels" anchor on the weight side; and the readout
is full-rank-hungry as deployed (rank 16 of 32 drops accuracy to 0.45), the
as-built face of §7. Caveats: one cell (32, 1584, 42), one quantizer family
per axis, k-means as the only clustering tried; and a measurement trap worth
recording — an emb-LP that hits its `time_limit` on the *first* round records
as infeasible with nothing proven (linear k=6 was first mis-recorded this way
under three concurrent LP processes; re-run alone it is feasible at 3.27e-2
with margin 8.9). Negative verdicts here were accepted only when round-0
completed with γ* = 0.

## 26. Designed gates: realizability is free, storability is not

`probe_designedgates.py`. §§13–14 built patterns by iterated solves and §22
by flow; the untried third path is *forward design* — pick the pattern from a
combinatorial object with distance properties, then one LP pass for weights.
Theorem 2 (theory.md) makes the design space exact: every additive gate
column is a token ordering plus thresholds, i.e. any per-neuron score pair
(a, b) with firing rule 1[a(x₀) + b(x₁) > 0] is realizable by construction,
and *only* such columns are. So "realizable high-margin patterns are
exponentially rare" splits cleanly: realizability can be had for free by
designing in score space; the open question is purely whether any declarative
score design is storable, given that Gaussian random scores (init,
random_additive) are not.

Four families, measured under §14's protocol (ridge-seeded readout, 3-round
ascent):

| family | design | density | ceiling |
|---|---|---|---|
| hadamard | Hadamard-row token signatures (pairwise distance 16), AND/NAND gates | 0.499 | infeasible |
| modular ×2 seeds | per-neuron affine bijections mod 64, fire iff fractional parts sum past 1 | 0.475 | infeasible |
| thermo ×2 variants | tokens as base-8 digit pairs, thermometer comparisons at 8 thresholds | 0.496 | infeasible |
| shuffle ×2 seeds | the trained gate's own score columns, token-permuted per neuron | 0.536 | infeasible |

All seven at γ* = 0, the do-nothing point. The thermometer family is the
inequality-native digit code — distances between fact signatures are literal
L1 distances between digit vectors, no equality channel anywhere, so the rank
barrier (`what-gd-builds.md` item 2) does not apply — and it dies with the
rest. A design-level trap en route: pure-AND Hadamard gates leave 44 facts
with empty active sets (trivially unstorable before any LP), and only one of
four mixed variants avoids empty facts at every column roll tried — even
*nonemptiness* is a designed property in this family.

The control carries the section. The shuffle gate matches every first-order
per-neuron score statistic of the trained gate — same marginals, same
densities, same near-tie profile per column — and destroys only the token
co-adaptation. It is infeasible at both seeds. No per-neuron property,
however exactly matched, buys storability; what is missing is *which token
gets which score*, jointly across all 4,096 cells — precisely the ~10
kilobits §25 measures as structurally incompressible. With §14 (magnitude
statistics), §23 (order statistics), and §18's Remark 4.1, this is the fourth
independent leg of 6‴, and the first from the design side: the program
searched the description space where a construction would have to live, and
found the trained solution's content absent from every family tried.
First-pass scope, honestly stated: seven configs, one cell, densities near
0.5, and no family co-designed with the fact list — co-adapted *declarative*
design (scores chosen as a function of the specific fact table) is the one
door this section does not close.

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
