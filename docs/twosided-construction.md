# Letting the ReLU be the gate

*A construction for [Linsefors & Bushnaq's sequence-memorisation
challenge](https://www.lesswrong.com/posts/KWtchKwwnJkd4bwCi/challenge-hand-coding-weights-for-efficient-sequence-1)
that stores more than the trained model at every size tested, and an account of why.*

Code: `handcode/twosided.py`. No gradient descent; every weight comes from a ridge
regression.

---

## The short version

A fact's label is stored as a **number**: the sum of the model's active hidden
activations equals `t0 + ℓ + 1`, and the post's own Appendix-A quadratic readout turns
"the sum is near `ℓ+1`" into "the argmax is `ℓ`".

The new part is what gates the neurons. Every value code in this family so far — the
authors' Appendix A, and my earlier `linsolve` — *builds* a gate, spending one whole
embedding matrix on mask entries. Doing so caps the fact-carrying parameters at `2d²`.
This construction builds no gate at all: the ReLU's own sign pattern is the gate. Both
embeddings then carry values, the budget doubles to `4d²`, and measured capacity follows.

The cost is that the sign pattern depends on the very weights being solved for. Freezing
it makes the fact equations exactly linear, so the algorithm is: freeze the pattern, solve,
re-read the pattern, repeat — with one non-obvious ingredient (a step-length cap) without
which it collapses to chance.

---

## 1. Why value codes have been stuck at `2d²`

The model is

```
h_i = relu(u_i[a] + v_i[b])          logits = W h
```

with `u, v` the two embeddings, each `n_input_vocab × d = 2d × d`.

A value code needs a decode, and the natural one is "sum the active neurons". Suppose the
active set is `S_a`, depending only on the first token — which is what you get when you
build the gate the obvious way, putting `0` on selected neurons and `-BIG` elsewhere in the
first embedding. Then

```
s = Σ_{i ∈ S_a} (u_i[a] + v_i[b])  =  [ Σ_{i ∈ S_a} u_i[a] ]  +  [ Σ_{i ∈ S_a} v_i[b] ]
                                       ^^^^^^^^^^^^^^^^^^^^
                                       one number per first token
```

The first bracket depends on `a` alone. **The entire first embedding therefore contributes
`2d` numbers to the fact equations, not `2d × d`.** It does not matter whether its entries
are `0`/`-BIG` or free reals; a mask that depends on one token collapses that matrix to one
scalar per token.

With one equation per fact, capacity is bounded by the count of genuinely free unknowns:

```
(d − 1) · n_input_vocab  ≈  2d²
```

This is not a loose bound. `linsolve` measures 83–91% of it at every `d` — 520, 2144, 8704,
34816 against `2d²` = 512, 2048, 8192, 32768. The bound binds, and it is the reason
`linsolve` loses to the trained model below `d ≈ 60`: it has half the parameters in play.

**The obstruction is the constructed gate, not two-sidedness.** To get the second embedding
into the equations, the active set must depend on both tokens. Building a two-token gate
explicitly — masks `m[a] ∧ n[b]`, values `p[a] + q[b]` — runs into a different wall: a
gated neuron has to output a *positive* number to be seen at all, since a ReLU cannot report
a negative one. That positivity constraint forces the gate density above `ρ = 0.866` and
costs rank; the budget caps at about `2.48d²`, which is roughly the 1.2× I measured. Details
in the README's negative-results section.

## 2. The escape: don't build a gate

Let the active set be whatever the ReLU makes it:

```
A(a,b) = { i : u_i[a] + v_i[b] > 0 }
s(a,b) = Σ_{i ∈ A(a,b)} (u_i[a] + v_i[b])
```

Two things change at once.

* `A(a,b)` depends on **both** tokens by construction, so neither bracket in §1 collapses.
  The unknowns are all `2 · (d−1) · 2d ≈ 4d²` of them.
* The positivity constraint disappears. In an explicit gate, "this neuron is on but wants a
  negative value" is a contradiction that has to be designed around. Here it is not a
  contradiction at all — the neuron simply switches off. **That is the ReLU doing its job,
  and it is precisely the freedom an explicitly built gate throws away.**

The requirement is still one equation per fact, `s(a,b) = t0 + ℓ + 1`, so the counting
ceiling doubles to `4d²`.

There is a coincidence worth noticing: the fact space has `n_input_vocab² = (2d)² = 4d²`
possible pairs. **This construction's parameter ceiling is exactly the size of the entire
fact space.** Whether that is deep or an artifact of the post's `n_input_vocab = 2d` tying,
I do not know — but it means the construction is measured against a dataset that runs out
at the same place its parameters do.

## 3. Solving a system whose own solution moves the equations

`s(a,b) = t` is nonlinear in `(u, v)`: change the weights and the active set moves. But it
is **piecewise** linear, and freezing the active set makes it exactly linear. So:

```
repeat R times:
    A ← { i : u_i[a] + v_i[b] > 0 }        read off the current weights
    solve  Σ_{i ∈ A} (u_i[a] + v_i[b]) = t   for (u, v),  A held fixed
```

The inner solve decomposes. With `v` held, each first token's row `u[a]` appears only in
that token's own facts, giving an independent least-squares problem with `n_a` equations in
`d−1` unknowns; symmetrically for `v`. Sweeping u-then-v is block Gauss-Seidel, and each
block is a ridge regression — which the challenge rules name explicitly as permitted. The
blocks are solved for the *correction* rather than for the row itself, which keeps whatever
the current iterate holds in the null space and is what makes a sweep a Gauss-Seidel step
rather than a restart.

## 4. The step-length cap, without which none of this works

Taking the full solve as the next iterate does not work. Measured at d=16: 43% of the sign
pattern flips in the first round, and accuracy sits at chance for as long as you let it run.

The reason is worth spelling out, because it is a **one-sided** error and therefore a
ratchet, not noise.

* If neuron `i` was in the frozen pattern but the solved weights make `z_i < 0`, the
  equation counted `z_i` (a negative number) while the ReLU contributes `0`. The true sum
  is **larger** than the equation's.
* If `i` was *not* in the pattern but ends up with `z_i > 0`, the ReLU contributes `z_i`
  where the equation counted nothing. The true sum is **larger** again.

Every pattern mismatch pushes the decode up, never down. The next round therefore has to
shrink the weights to compensate, which pushes more neurons below zero, which increases the
overshoot. Left alone this empties the network: measured densities fall from 0.53 to 0.05
while accuracy sits at chance.

The fix is to take the longest step that leaves the pattern nearly intact — at most 2% of
units flipping. This needs no search. Along the step, each unit is an affine function
`pre + α · step_pre` of the step length, so it crosses zero at exactly one `α`, and only
when `step_pre` opposes its current sign. Collect those crossing points; the number of flips
at step `α` is just how many lie below `α`. So **the longest admissible step is the 2% order
statistic of the crossing times** — one `kthvalue` over the array.

It is also self-scheduling, which no fixed damping factor reproduces: the step starts near
`0.01` while the pattern is still being found and rises to `1.0` once it settles. When it
reaches `1.0` with zero crossings, the solve has reproduced the pattern it was handed — a
genuine fixed point — and the iteration stops.

## 5. The offset, and why it is much smaller than `linsolve`'s

Both constructions add a constant `t0` to every target, for different reasons.

`linsolve` needs every *gated* value to stay positive against a label range of `d`, which
forces `t0 ~ d²` and drags its weights to `7e3` in the embedding and `2.5e6` in the
unembedding. It is the reason that construction is precision-limited and dies in float32
around d≈290.

Here `t0` has only to keep the sign pattern from churning: a neuron's pre-activation must be
large compared with how far the solve moves it, and that movement is set by the label range.
`t0 = 16d` suffices — `t0 = 0` fails outright at 0.07 accuracy. The upper limit is float32:
the readout resolves a half-unit gap out of logits of size `d·(t0 + d)` formed by subtracting
two nearly equal large numbers, so `t0 ~ d²` already costs 4% accuracy at d=128 while
float64 still reads 1.000. Scaling `t0` with `d` keeps that ratio flat as models grow.

One further free improvement: the bias neuron's height `β` is a pure gauge — it appears as
`β/2` in the embedding and divides the readout's bias column, so it cancels exactly from
every logit. Setting it to twice the largest solved value instead of to `d` drops
`max|W_U|` from `1056` to `64` at d=64 with the embedding and the accuracy untouched.

## 6. What the measurements say

Capacity, best over the hyperparameter sweep and 3 seeds, as in the post:

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

Three things in that table are worth more than the headline ratio.

**The counting argument reproduces itself.** Run the construction at exactly `n = 4d²` —
every pair that exists — and the best accuracy is 0.8516, 0.8518, 0.8520, 0.8508 at
d=16/32/64/128. That is the `keep = 0.85` greedy-drop schedule returning *precisely* what it
was asked to keep: drop 15% of the facts and the remaining `3.4d²` equations fit inside
`4d²` unknowns, so they are all satisfied and nothing else is. Four sizes, four decimal
places, one number.

**The exponent is not really super-quadratic — and the plain power law is clearer.**
Fitted in the post's `a·d^b/ln d` form, `twosided` gives `b ≈ 2.3`, which looks
super-quadratic. It is not: a pure `C·d²` law fitted in that form over d=16–128 *also* gives
`b ≈ 2.28`, because the `/ln d` has to be absorbed somewhere. Dropping the `ln d` and fitting
`C·d^p` directly is much more legible:

| condition | `p`, acc=1 | `p`, acc≥0.9 | capacity grows by |
|---|---|---|---|
| their hand-coded | 1.23 | 1.64 | ~2.4–3.1× per doubling of `d` |
| **trained** | **1.88** | **1.73** | **~3.3×** |
| linsolve | 2.01 | 2.02 | ~4.0× |
| **twosided** | **2.06** | **2.03** | **~4.1×** |

Both value codes are clean `d²` laws — capacity multiplies by 4.0–4.1 every time `d`
doubles, which is exactly what "one equation per fact against `C·d²` free parameters"
predicts. **The trained model is measurably sub-quadratic at `d^1.73`–`d^1.88`**, multiplying
by only 3.3×. That single fact explains the whole shape of the comparison: the constructions
do not have a better constant, they have a better *exponent*, so they start behind at small
`d` and pull away. It also predicts the gap keeps widening, which is worth stating as a
falsifiable claim rather than an extrapolation to be trusted.

**The first embedding really is carrying facts.** Freezing it at its random initialisation
— halving the free parameters back to `2d²` — collapses accuracy from 1.000 to 0.06 at
d=32, n=3072, at every load tested. This is checked in the test suite
(`test_first_embedding_carries_facts`). It is not a fair stand-in for `linsolve` at the same
budget, since there the first embedding is a *designed* mask rather than a random one, but
it does rule out the possibility that the second embedding is quietly doing all the work.

## 7. Does it look like a trained model?

This is the post's actual goal, and the more interesting axis. Measured by
`probe_coding.py` at ~90% of each construction's own acc=1 capacity, d=64:

| | trained | their hand-coded | linsolve | **twosided** |
|---|---|---|---|---|
| facts stored | 6566 | 194 | 5414 | **11520** |
| density | 0.53 | 0.82 | 0.08 | **0.44** |
| binarise activations → | 0.11× accuracy | 1.00× | *(n/a)* | *(n/a)* |
| coding scheme | magnitude, ~5 bits/neuron | pattern | magnitude (analytic) | **magnitude (analytic)** |
| max abs embedding weight | 6.4 | 1 | 4.7e2 | **5.4e2** |
| max abs unembedding weight | 6.3 | 2 | 4.2e4 | **6.4e1** |
| parameters carrying facts | all `5d²` | — | `2d²` | `4d²` |

`FINDINGS.md` establishes the first column: near capacity the trained model is **not** using
a pattern code — binarising its activations retains 11% of its accuracy, and ~32 magnitude
levels are needed to keep all of it. The authors' construction is a pure pattern code
(binarising is free), which is the regime the trained model occupies below ~10% load and
abandons thereafter. That measurement is what motivated a value code in the first place.

The *n/a* entries are a caveat about the instrument, not a result. The probe retrains a
linear readout on coarsened activations, and on both value codes it scores 0.07–0.10 where
the construction's own readout scores 1.000 on the identical activations — it fails to
re-derive a decode that demonstrably exists, because that decode is a narrow large-weight
solution the probe's objective does not lead to. No ratio computed from it means anything.
For these two the answer is analytic anyway: the decoded sum *is* the label, so they are
magnitude codes by construction and binarising destroys the label outright.

On the axes that *are* directly measurable, this construction lands much closer to a trained
model than `linsolve` did: density 0.44 against the trained 0.53 (where `linsolve` sits at
0.08), and an unembedding whose largest weight is 64 rather than 42000. It is still not a
trained model — the embedding's weights grow like `d` where trained weights stay `O(1)`, and
the unembedding's `d²` parameters carry no fact information at all, where gradient descent
uses all `5d²`. That last point is also the obvious place to look for the next factor of
`1.25×`.

## 8. Why doesn't gradient descent find it?

The weights exist inside the architecture and gradient descent does not reach them, so
either it *cannot get there*, or it *would not stay*. `probe_reachability.py` tests the
second directly, at d=32 and n=3168 — the construction's own acc=1 capacity, 1.52× the
trained model's.

**It would not stay.** Start Adam exactly at the construction and it walks off within 200
epochs and settles around 0.83:

| | accuracy | cross-entropy |
|---|---|---|
| the construction | **1.0000** | 0.898 |
| + 2000 Adam epochs from there | 0.834 | **0.737** |

Adam lowered its loss by 18% while giving up 17% of the facts. Nothing is going wrong: it is
minimising its objective correctly, and **the objective disagrees with the metric**. The
construction is not a stationary point of cross-entropy at all, so no amount of better
initialisation or longer training would settle on it.

The visible difference is decision margin. At 100% accuracy on their respective fact sets:

| | median best-minus-second logit | min |
|---|---|---|
| the construction | **0.42** | 0.02 |
| trained model | **7.77** | 5.93 |

The quadratic decode's gap between adjacent labels is exactly `0.5`, and the construction
spends all of it — every fact is correct by a hair, against logits spanning ±500.
Cross-entropy regards that as a poor solution and buys margin with facts.

**I could not confirm that this is the *cause* of the capacity gap, and the reason is worth
recording.** The obvious test is to retrain under an objective that stops caring once a small
margin is met. That test does not work in this architecture, because the loss is
scale-invariant: the unembedding is unconstrained, so the network meets any fixed margin
target by simply scaling itself up, and a margin cap is not actually a cap. Both attempts
behaved accordingly — a multi-class hinge at margin 0.5 and 1.0 failed to train at all (0.15
where cross-entropy reaches 1.000), and cross-entropy on `β·logits` for `β` up to 100 left
capacity unchanged or slightly worse. So the margin gap is a solid *description* of where
the two solutions sit, and an unproven hypothesis about why.

What is established is narrower but still useful to the post: **the hand-coded/trained gap is
not purely an optimisation-difficulty story.** Part of it is that the training objective does
not want this solution. That is a different kind of claim from "gradient descent is bad at
finding it", and it predicts that no improvement in initialisation or schedule will close it.

A second, independent piece of evidence points the same way. The retrained-readout probe in
§7 hands gradient descent the construction's *own activations* — the hard part solved — and
asks only for the readout. Starting from a closed-form ridge fit, it reaches 0.07–0.10 where
the construction's readout scores 1.000 on those identical features. Even the easy half of
the problem is not something gradient descent recovers.

## 9. Where this is vulnerable

Stated plainly, because a referee will ask.

**The iteration count.** The rules forbid "gradient descent, or any other generic black-box
optimizer", and permit "closed-form computations, greedy algorithms, and combinatorial
constructions", naming ridge regression explicitly. Every weight this algorithm produces
comes from a ridge regression; no gradient of any loss is ever computed. But it runs 150
freeze-and-solve rounds, and that is an iterative numerical procedure with a step-length
rule, which a reasonable referee could call a line search.

The case for it: the search direction is *determined*, not explored — freezing a piecewise
linear system's active set and re-solving is exactly a Newton step, and the step cap is an
order statistic, not a trial-and-error backtrack. What the iteration searches for is a
self-consistent **active set**, a combinatorial object, in the same spirit as the authors'
own greedy search over connection matrices. And the rule's stated spirit — "the algorithm
should embody an explanation of *how* the facts get stored" — is met about as directly as it
can be: you can read a fact's label off the activations by adding them up. The iteration
count does not change what the weights mean.

I would rather flag this than have it found.

**Capacity is a best-of-sweep order statistic**, over 4 initial densities × 3 ridge
strengths × 3 drop schedules × 3 seeds. That is the post's own protocol, so comparisons are
like-for-like, but the absolute numbers are optimistic for every condition alike.

**Convergence, not counting, binds at the top.** By parameter counting the `keep = 0.92`
schedule ought to reach 0.92 at `n = 4d²`, clearing the 0.9 threshold and saturating the
fact space. It does not, at a 150-round budget — it reaches 0.84. Raising the budget to 400
rounds gets 0.905 at d=32, i.e. the whole fact space. So the reported acc≥0.9 numbers at
large `d` are limited by the round budget, and the true capacity of the *construction* is
somewhat higher than the table says. I have reported the fixed-budget figure rather than
tuning the budget upward per point.

**Single fact-set seed.** All capacity numbers use fact seed 42, as in the reference
implementation. Sampling variation across fact sets is not measured.

**The trained baseline is partly budget-limited, which shrinks the acc=1 ratios.** The
comparison uses the post's own training recipe — Adam, lr 1e-2, full batch, up to 5000
epochs, early stop after 100 epochs without improvement — and my reproduction of it matches
their published curves to 0.95–1.05 per point, so the comparison is like-for-like against
the post. But that recipe is not converged. At d=32, single seed:

| n_facts | 5k epochs (the recipe) | 50k | 200k |
|---|---|---|---|
| 2080 (reported capacity) | 0.9995 | 1.0000 | 1.0000 |
| 2560 | 0.8148 | **0.9969** | 0.9980 |
| 2880 | 0.6347 | 0.8382 | 0.8278 |
| 3168 (the construction's) | 0.6133 | 0.7080 | 0.7465 |

A model trained 40× longer reaches roughly 2560 at acc=1 rather than 2080, so the honest
d=32 acc=1 ratio against *converged* gradient descent is about **1.24×, not 1.52×**. The
acc≥0.9 baseline barely moves (2528 → ~2700, since 2880 plateaus at 0.83).

**This turns out to be a small-`d` artifact, not a general one.** The same check at d=64:

| n_facts | 5k epochs | 60k |
|---|---|---|
| 7296 (reported capacity) | 1.0000 (@1613) | 1.0000 (@1429) |
| 9600 | 0.7168 | 0.7765 |
| 11200 | 0.5380 | 0.6300 |
| 12800 (the construction's) | 0.4237 | 0.4909 |

Here the recipe is already converged — 7296 is reached by epoch ~1600 whatever the budget,
and 9600 plateaus at 0.78 with twelve times the epochs. So the d=64 ratio of **1.75× stands
against converged gradient descent**, and only the d=16 and d=32 acc=1 points are inflated
by the training budget. Worth stating precisely rather than either ignoring it or letting it
discredit the larger sizes.
