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
epoch 10 to ~0.5% by epoch 30 — while accuracy is still at 21%. The gate is found
first; the remaining ~1600 epochs grow margins on an almost-settled pattern.

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
