# What the trained model actually does

Notes toward the post's stated goal — *"a clearer understanding of how look-ups are
encoded"* — rather than toward the capacity benchmark. All measurements are on the
Figure-4 toy model, `handcode/` for the code.

Method throughout: take a model's hidden activations `H` on its fact set, coarsen the
**magnitudes** while keeping the **pattern** (`h → 1[h>0]`, or quantised to `L` levels),
and retrain a linear readout on each variant. Retraining matters — ridge under-reads
badly here (0.479 where the model itself gets 1.000) because ridge optimises L2 while the
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

**At low load the trained model is a pattern code — binarising its activations costs
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
| trained | 6566 | 0.53 | 1.000 | 0.103 | 0.10 | **magnitude** |
| their hand-coded | 194 | 0.83 | 1.000 | 1.000 | 1.00 | **pattern** |
| linsolve | 5414 | 0.11 | — | — | — | magnitude (by construction) |

Their silence code is a *pure* pattern code: binarising is free, because the code is
literally "are these neurons off". That is exactly the regime the trained model occupies
at ≤10% load and abandons thereafter.

`linsolve` needs no probe: its decoded sum *is* the label, so it is a magnitude code by
construction. (A probe would mislead — a randomly-initialised readout scores 0.035 on its
features, because the decode needs weights of ~2.5e6 that Adam will not find from a small
init. That is an optimisation failure, not an information one.)

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

## Caveats

* Probes are trained linear readouts; a probe failing bounds what a *linear* readout can
  extract, not what information is present. Where a construction has a known decode
  (linsolve), the analytic answer is used instead.
* Quantisation is uniform between the min and max of the positive activations. A
  quantiser matched to the activation distribution would likely need fewer levels, so
  "~32 levels" is an upper bound on what the trained model requires.
* One seed per cell. The trend is monotone and consistent across three model sizes, but
  the individual numbers are single measurements.
