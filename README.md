# Hand-coded weights for efficient sequence memorization

A reproduction of [Linsefors & Bushnaq, *Challenge: Hand coding weights for efficient
sequence memorisation*](https://www.lesswrong.com/posts/KWtchKwwnJkd4bwCi/challenge-hand-coding-weights-for-efficient-sequence-1)
(LessWrong, 2026-07-23), and an entry to the challenge it poses.

**Result: a hand-coded construction that stores more than the gradient-trained model at
every size tested — 1.2× at d=16 rising to 2.3× at d=128 — and 11–66× the authors' own
construction**, with no gradient descent anywhere. At acc≥0.9 it stores 90–97% of every
fact that exists.

The reason it wins is a single structural change, and it is the interesting part: every
value code in this family so far *builds* a gate, which spends one whole embedding matrix
and caps the fact-carrying parameters at `2d²`. Letting the ReLU's own sign pattern be the
gate costs nothing and doubles the budget to `4d²`. The full account is in
[docs/twosided-construction.md](docs/twosided-construction.md), written to be readable as a
contribution back to the post.

Measured directly, the constructions store `~C·d²` facts while the trained model stores
`~C·d^1.73`–`d^1.88` — so this is an exponent difference, not a constant, and the gap
widens with `d`.

**The honest counterweight: the capacity is bought with fragility the benchmark does not
score.** At matched load the construction tolerates ~1000× less weight noise than a
trained model — less than a single Adam step — while the authors' own construction sits in
the trained model's robustness class (`probe_robustness.py`, FINDINGS.md §5). Fixing the
readout and retraining shows the remaining capacity gap is the optimizer, not the
objective: Adam reaches the trained model's full capacity through a readout carrying zero
fact information, then stalls where converged training stalls, short of the construction
(FINDINGS.md §7). So the construction settles what the architecture *can hold*; why
gradient descent holds robustly less is the question the measurements in `FINDINGS.md`
now put numbers on — and a robustness-qualified metric would be the version of this
challenge that tracks trained storage. The emerging answer to what gradient descent
actually builds — and how much of it can be reproduced without gradient descent (about
two of the three-and-a-half orders of magnitude, so far) — is assembled in
[docs/what-gd-builds.md](docs/what-gd-builds.md).

## The task

Memorize `n_facts` structureless facts — a random pair of input tokens maps to a random
label — in the bias-free one-layer ReLU MLP of the post's Figure 4:

```
hidden = relu(E1[tok_a] + E2[tok_b])      up   = [E1 | E2] : (d_mlp, 2 * n_input_vocab)
logits = hidden @ W_U.T                   down = W_U       : (n_output_vocab, d_mlp)
```

No attention, no norms, no residuals, no biases; `W_in` is folded into the embeddings and
`W_out` into the unembedding, so the whole model is those two matrices. Dimensions are
tied to one scale parameter `d`: `n_input_vocab = 2d`, `d_mlp = d`, `n_output_vocab = d`.

Capacity is measured by binary-searching the largest `n_facts` storable at accuracy = 1
and at accuracy ≥ 0.9, then fitting `max_facts = a · d^b / ln(d)`.

## Reproduction

Their four conditions, fitted over d ∈ {16, 32, 64, 128}, 3 seeds per cell
(`hand-coded` also measured at d=256, where it hits 9728 against its published fit's
9290 — a 1.05× ratio at a size beyond anything in the post):

| condition | threshold | ours | published |
|---|---|---|---|
| trained | acc≥0.9 | **8.49·d^2.00** | **9.42·d^1.97** |
| hand-coded | acc≥0.9 | **1.17·d^1.91** | **1.16·d^1.93** |
| hand-coded | acc=1 | 2.23·d^1.50 | 2.17·d^1.55 |
| hybrid | acc≥0.9 | 1.67·d^2.09 | 2.20·d^2.02 |
| rand-emb | acc≥0.9 | 0.211·d^2.30 | 0.278·d^2.21 |

Per-point ratios to the published curves are 0.95–1.02 for `trained` and `hand-coded`.
`rand-emb` is the noisiest, as it is in their data too.

Faithfulness is checked directly against the authors' own code in `reference/`
(`tests/test_matches_reference.py`): fact generation is byte-identical, the construction's
weights are bit-identical wherever the frequency ranking is unambiguous, and where it ties
both implementations pick tokens of the same frequencies and produce accuracy
distributions whose means agree to ~1 SE.

## The construction: a value code solved as a linear system

`handcode/linsolve.py`. Two ideas from the post itself, combined:

* **Appendix A**: a ReLU neuron's output is a *number*, so one neuron can carry a whole
  label rather than one bit of a pattern. Appendix A gives each first token its own
  selector neuron and a quadratic readout that turns "activation ≈ ℓ+1" into "argmax = ℓ".
  It needs `d_MLP = n_input_vocab + 1` — twice the width the challenge allows.
* **Dugan et al.**, which the post names as a valid entry: fix the gating at random, and
  what is left is a linear system in the value weights, solvable exactly.

Giving each first token a random *set* `S_a` of `k` selectors, and asking only that the
**sum** come out right, removes Appendix A's width requirement:

```
u_i[a] = 0 if i in S_a else -BIG        v_i[b] = w_i[b]
  =>  h_i = 1[i in S_a] * w_i[b]
  =>  sum_i h_i = sum_{i in S_a} w_i[b]  =  T0 + l + 1     <- one equation per fact
```

The equations for different second tokens are disjoint, so this is `n_input_vocab`
independent small ridge solves. One neuron is held permanently on (`u = v = 1`, so `h = 2`)
to supply the decode's per-class thresholds:

```
logit_c = (c+1)(s - T0) - (c+1)^2/2 = -(c-l)^2/2 + const   ->  argmax at c = l
```

Both terms are linear in `h`, so this is an ordinary unembedding.

Three details carry most of the performance:

* **`T0`** is a constant offset on every target. A ReLU cannot report a negative value, and
  the raw targets `ℓ+1` span `[1,d]` with mean `(d+1)/2` — spread as large as the mean — so
  about half the solved values would be clamped to zero. `T0` lifts them clear, and costs
  nothing because it is identical for every fact and the bias neuron subtracts it back off.
  It is bounded only by float32 round-off, since the decode must still resolve ±1/2 against
  a sum of size `T0`. Tuning it was worth 5–11 accuracy points at the capacity edge.
* **Greedy drop.** Past the exact-solve threshold, least squares spreads a little error
  over *every* fact — the worst allocation when the metric counts facts. Sacrificing the
  hardest few and re-solving concentrates the damage where it is already paid for.
* **`pertoken` pre-drop.** A second token's system has `d-1` unknowns and can satisfy at
  most that many of its facts, so the excess is written off up front.

### Results

| max facts, acc≥0.9 | d=16 | d=32 | d=64 | d=128 | d=256 |
|---|---|---|---|---|---|
| their hand-coded | 86 | 244 | 768 | 2592 | 9728 |
| **linsolve** | **520** | **2144** | **8704** | **34816** | **126976**† |
| trained | 760 | 2528 | 8320 | 27648 | not measured |
| linsolve / their hand-coded | 6.0× | 8.8× | 11.3× | **13.4×** | **13.1×** |
| linsolve / trained | 0.68× | 0.85× | **1.05×** | **1.26×** | — |

† float64-verified only; marginal in float32 (see [Caveats](#caveats)).

`trained` at d=256 was not measured (it takes hours per point); their published fit
extrapolates to ~94k there, against linsolve's 127k, but that is an extrapolation and is
not claimed.

At acc=1: 384 / 1408 / 6016 / 24832, against trained's 496 / 2080 / 7296 / 25088 — i.e.
0.99× of the trained model at d=128, and 32× the authors' construction. The acc=1 metric
allows no error budget, so there the `2d²` bound still binds.

Fitted over d ∈ {16 … 256}, `linsolve` is `3.11·d^2.24/ln d` against trained's
`8.49·d^2.00/ln d`.
**The exponent exceeds the trained model's** while the prefactor is lower, so the curves
cross — at d≈60, down from d≈112 before tuning. At d=128 the construction reaches 0.43
facts per parameter, i.e. **3.0 bits per parameter**, at or slightly above the usual
empirical ceiling for memorization in transformer weights.

Note that 34816 exceeds the `2d² = 32768` exact-solve bound: the greedy drop buys that by
spending part of the 10% error budget, which is precisely what it is for.

### Why it falls short at small d — and why that is structural

All fact-specific information lives in the second embedding, so the facts that can be
pinned exactly are bounded by that matrix's size, `(d-1)·n_input_vocab ≈ 2d²`. Measured
capacities sit at 83–91% of that bound at every `d`. Matching a trained model storing
`9.42·d²/ln d` therefore needs `2 ln d > 9.42`, i.e. **d > 111** for an exact solve — the
greedy drop pushes the crossover down to d≈60 by trading exactness for the error budget.
Below that, no construction in this family matches it, whatever the tuning.

## The construction that fixes it: let the ReLU be the gate

`handcode/twosided.py`. Full write-up in
[docs/twosided-construction.md](docs/twosided-construction.md); the short version:

The `2d²` bound above is not about value codes, it is about *built* gates. If the active set
`S_a` depends only on the first token, then

```
s = Σ_{i ∈ S_a} (u_i[a] + v_i[b])  =  [Σ_{i ∈ S_a} u_i[a]]  +  [Σ_{i ∈ S_a} v_i[b]]
                                       one number per first token
```

so the entire first embedding contributes `2d` numbers to the fact equations rather than
`2d·d`, whatever its entries are. Building a two-token gate explicitly does not help either:
a gated neuron must output a *positive* value to be seen at all, and that constraint costs
enough rank to cap the budget near `2.48d²`.

So build no gate. Take the active set to be `A(a,b) = {i : u_i[a] + v_i[b] > 0}` — the
ReLU's own sign pattern. It depends on both tokens by construction, and the positivity
constraint evaporates, because a neuron that would go negative simply switches off. Both
embeddings now carry facts: `4d²` unknowns against one equation per fact.

The equations are nonlinear, since the active set moves with the weights, but only
*piecewise* — freezing the pattern makes them exactly linear, and the frozen system
decomposes into one independent ridge regression per token. So: freeze, solve, re-read the
pattern, repeat. The one non-obvious ingredient is the step rule; taking the full solve
flips 43% of the pattern and collapses the whole thing to chance, via a one-sided ratchet
described in the write-up. Capping the step so that at most 2% of the pattern flips fixes
it, and the cap is an order statistic of per-neuron zero-crossing times, so it needs no
search.

### Results

| max facts | d=16 | d=32 | d=64 | d=128 |
|---|---|---|---|---|
| their hand-coded, acc=1 | 53 | 130 | 216 | 776 |
| trained, acc=1 | 496 | 2080 | 7296 | 25088 |
| **twosided, acc=1** | **696** | **3168** | **12800** | **51200** |
| ratio to trained | 1.40× | 1.52× | 1.75× | **2.04×** |
| trained, acc≥0.9 | 760 | 2528 | 8320 | 27648 |
| **twosided, acc≥0.9** | **928** | **3904** | **15872** | **63488** |
| ratio to trained | 1.22× | 1.54× | 1.91× | **2.30×** |
| — as % of the whole fact space | 90.6% | 95.3% | 96.9% | 96.9% |
| whole fact space, `4d²` | 1024 | 4096 | 16384 | 65536 |

Two results in that table say more than the ratios.

**The counting argument reproduces itself to four decimal places.** Run the construction at
exactly `n = 4d²` — every pair that exists — and the best accuracy is 0.8516, 0.8518,
0.8520, 0.8508 at d=16/32/64/128. That is the `keep=0.85` greedy-drop schedule returning
precisely what it was asked to keep: drop 15% and the remaining `3.4d²` equations fit inside
`4d²` unknowns, so all of them are satisfied and nothing else is.

**Fitted as a plain power law, the constructions and the trained model differ in exponent,
not constant:** `twosided` and `linsolve` both scale as `d^2.0`–`d^2.1` (capacity ×4.1 per
doubling of `d`), the trained model as `d^1.73`–`d^1.88` (×3.3), the authors' construction as
`d^1.35`–`d^1.71`. The post's `a·d^b/ln d` form obscures this — a pure `C·d²` law fits it
with `b ≈ 2.28`, so `twosided`'s `b = 2.3` is not super-quadratic.

### How it compares to a trained model

`probe_coding.py`, at ~90% of each construction's own acc=1 capacity, d=64:

| | trained | their hand-coded | linsolve | **twosided** |
|---|---|---|---|---|
| density | 0.53 | 0.82 | 0.08 | **0.44** |
| binarize activations → | 0.11× accuracy | 1.00× | n/a | n/a |
| max abs embedding weight | 6.4 | 1 | 4.7e2 | **5.4e2** |
| max abs unembedding weight | 6.3 | 2 | 4.2e4 | **6.4e1** |
| parameters carrying facts | all `5d²` | — | `2d²` | `4d²` |

Density and unembedding scale both move a long way toward the trained model relative to
`linsolve`. The `n/a`s are a limitation of the probe, not a result: it retrains a linear
readout, and on a value code it scores 0.07–0.10 where the construction's own readout scores
1.000 on the same activations, so it has failed to re-derive a decode that demonstrably
exists. For those two the coding scheme is known analytically — the decoded sum *is* the
label — so they are magnitude codes and binarizing destroys the label outright.

## Caveats

* **This is a capacity result, not a mechanism result.** `linsolve` uses weights spanning
  ±7e3 in the embedding and reaching -2.5e6 in the unembedding, where trained models have
  small, smeared weights. Nothing in the rules forbids it — the post says the same of its
  own Appendix A, whose overall readout scale is likewise free — but the post's actual goal
  is understanding *how trained models store facts*, and on that axis this construction is
  about as unlike a trained model as Appendix A is. It says what the architecture can do,
  not what gradient descent does.
* **The construction spends numerical precision to buy capacity.** The decode resolves
  ±1/2 against sums of size `T0` (~17 bits of dynamic range at d=128), so accuracy depends
  on the arithmetic precision in a way the ternary constructions do not. Individual
  large-`T0` cells disagree between float32 and float64 by up to 0.19 accuracy; the `T0`
  sweep routes around this, selecting a smaller `T0` when scoring in float32. Up to d=128
  the reported capacities hold under both conventions (at d=128, n=34816: 0.9007 float32,
  0.9214 float64; n=37000 fails under both).

  **At d=256 this stops being free.** The decode accumulates ~`d` terms against a sum of
  size `T0`, so the demand grows with `d`: at `T0 = 100d = 25600`, float32's ~1e-7 relative
  error over 255 terms is ~0.65, exceeding the +-1/2 the decode needs. The d=256 acc>=0.9
  figure of 126976 verifies in float64 (0.9206) but came in at 0.8979 in float32 under a
  reduced sweep -- marginal, and not independently confirmed in float32. The acc=1 figure
  (110592) is confirmed under both. Treat the d=256 acc>=0.9 point as float64-only, and
  expect float32 to fail outright somewhere beyond it: **this construction's capacity is
  ultimately precision-limited, which is itself a finding about the value code** -- it
  stores facts in the low-order bits of large numbers, and eventually runs out of them.
* **Capacity is a best-of-sweep order statistic**, as it is in the post: the maximum over
  the hyperparameter grid and 3 seeds. That is their protocol, so the comparison is
  like-for-like, but the absolute numbers are optimiztic for every condition alike.
* **`trained` uses 3 seeds, not their 11.** This costs the baseline ~20% at d=16 (496 vs
  600); from d=32 up the two agree exactly, and the fitted curve matches their published
  one to 0.95–1.02 per point.
* **The trained baseline is partly budget-limited, which shrinks the acc=1 ratios.** The
  comparison uses the post's recipe — Adam, lr 1e-2, full batch, ≤5000 epochs, early stop
  after 100 epochs without improvement — and reproduces their published curves to 0.95–1.05
  per point, so it is like-for-like *against the post*. But at small `d` that recipe is not
  converged: at d=32 a model trained 40× longer reaches ~2560 facts at acc=1 rather than
  2080, making the honest d=32 acc=1 ratio **~1.24×, not 1.52×** (acc≥0.9 barely moves,
  2528 → ~2700). **This is a small-`d` artifact.** At d=64 the recipe is already converged —
  7296 is reached by epoch ~1600 at any budget, and 9600 plateaus at 0.78 with 12× the
  epochs — so the d=64 ratio of 1.75× stands against converged gradient descent. Only the
  d=16 and d=32 acc=1 points are inflated by the training budget.
* **`twosided`'s iteration count is the part a referee should push on.** Every weight comes
  from a ridge regression and no gradient of any loss is computed, but it runs 150
  freeze-and-solve rounds with a step-length rule. The argument that this is a construction
  rather than an optimizer — the search direction is determined, not explored, and what it
  searches for is a self-consistent *active set* — is set out in
  [docs/twosided-construction.md](docs/twosided-construction.md) §9, along with the reasons
  it might not persuade.

## Two smaller constructions

Both keep the authors' architecture and the no-gradient-descent rule.

`hc-tiebreak` — their embedding, plus a readout fix. Measured first: at their acc=1
capacity points, **100% of their errors are ties at logit 0**. The correct label always
scores exactly 0 by construction, so an error is a wrong label also scoring 0, resolved by
argmax's index order — the construction never misranks, it *abstains*. Adding a ridge
correction scaled small enough to adjudicate only ties (the bound is provable from
`h_i ∈ {0,1,2}` making the silence logits even integers) recovers 1.23–1.80× more facts.

`coin-tiebreak` — an active-pattern embedding, motivated by the post's Figure 8 note that
trained models use patterns of *active* neurons where their construction uses inactive
ones. With `u_i[a] = ±1`, `v_i[b] = ±1` a neuron fires **iff** `a ∈ A_i ∧ b ∈ B_i`: an
exact AND over a rectangle, with none of the collateral silencing a `-1` causes in their
scheme. Worth 1.10–1.75×.

## Negative results (kept deliberately)

* **A Hebbian/prototype tie-breaker resolves ties at chance.** On a tied fact `h` is
  already zero on every tied label's neurons, so `⟨h, μ_c⟩` collapses to the same value for
  all of them — the same degeneracy as the authors' own "uniform positive weights change
  nothing" observation, one order up. Ridge works only because of the `(HᵀH)⁻¹` whitening;
  the Hebbian rule is literally ridge's λ→∞ limit.
* **Pure ridge is worse than their construction** at d=128 (0.90×). Their exact-zero
  silence trick carries real information; the win comes from adding to it, not replacing it.
* **Graded values inside a rectangle hurt monotonically.** Within a rectangle a real fact
  and a cross-pair are not additively separable, so no `x_i[a] + y_i[b]` can separate them —
  the `x,z→l; p,q→l; x,q→not l` obstruction from the post. This is why the magnitude
  channel is unusable per-neuron, and why Appendix A needs `d_MLP = n_input_vocab + 1`.
* **Two-sided value codes with an *explicitly built* gate cap at ~1.24×, provably.** To
  beat the `2d²` bound both embeddings must carry values, which needs a gate depending on
  both tokens. A *constant* gate size would remove the fluctuation problem, but constant
  intersection means `M Nᵀ = r·J`, which is rank 1 — valid masks solve `N·1_{m[a]} = r·1`
  and so live in a space of dimension `nv − rank(N)`. Mask diversity trades against
  `rank(N)` one-for-one and the total caps back at `V·nv = 2d²`. With fluctuation allowed,
  positivity forces `ρ > 0.866` and the rank is `2Vρ·nv − nv² ≈ 2.48d²` — matching the
  ~1.2× measured.

  **The bound is real but its scope is narrower than it first looked.** Every step above
  assumes the gate is a pair of mask matrices `m[a] ∧ n[b]`, so that a gated neuron must
  output a *positive* `p[a] + q[b]` to be seen at all. Drop the explicit gate and let the
  ReLU's own sign pattern do the gating and the positivity constraint disappears with it,
  because a neuron that would go negative simply switches off. That is `twosided` below,
  which reaches the full `4d²` budget. The lesson kept here is that the obstruction was
  the *constructed* gate, not two-sidedness.
* **Projection methods lose to the greedy drop.** Kaczmarz projecting to exact targets
  restores an equality system, which past the DOF limit is inconsistent and cycles. POCS to
  interval edges fixes that and is still worse, because least squares optimizes L2 while the
  metric is L0 — projection perturbs already-correct facts to help violated ones. This is
  *why* the greedy drop is the right tool.
* **Per-neuron readout gains don't help**; conditioning was not the binding constraint.
* **Splitting neurons between two codes** is worse than either alone.

## Performance

`handcode/fastsolve.py` is a batched rewrite of the solver, ~9× end-to-end, with identical
results (d=16 acc≥0.9: 66s → 4s; d=64: 1354s → 183s). Three things make it work:

* **The solve does not depend on `T0`.** Every design row has exactly `k` ones, so
  `wanted − baseline·k` collapses to `ℓ+1` and `T0` never reaches the linear system — it is
  only a shift applied afterwards. The whole `T0` sweep costs one solve.
* **Facts are bucketed by second token once**; dropping facts only flips a validity mask,
  so the grouping is never rebuilt and the O(N) rescan per token disappears.
* **Ridge's primal and dual forms** are chosen *per token*. They are equal in exact
  arithmetic but not in floating point: with tiny `mu` the primal squares the condition
  number of an under-determined system and the dual does the same for an over-determined
  one. Choosing globally is silently wrong by ~1e2 against a decode tolerance of 1/2.

The runner also splits BLAS threads across however many jobs will actually run
concurrently, so the tail of a search does not leave cores idle.

## Running it

```bash
uv sync
uv run pytest                                          # 53 checks incl. differential vs reference/
uv run python precompute_conn.py --ds 16 32 64 128     # ~1 min, cached to results/conn_cache
uv run python run_scaling.py --ds 16 32 64 128         # the experiment (resumable)
uv run python run_scaling.py --ds 16 32 64 128 --report-only
uv run python plot.py                                  # results/scaling.png
uv run python probe_coding.py --d 64                   # density, weight scale, pattern-vs-value
uv run python probe_reachability.py --d 32             # why gradient descent does not find it
uv run python probe_robustness.py --d 32               # accuracy under weight/activation noise
uv run python probe_walkaway.py --d 32                 # which facts Adam gives up, by margin
uv run python probe_fixed_readout.py --d 32            # train embeddings under the fixed decode
uv run python plot_robustness.py                       # results/robustness.png
uv run python probe_structure.py                       # weight-level shape of the trained solution
uv run python probe_digitcode.py --d 32                # capacity-robustness frontier of digit codes
uv run python probe_pedestal.py                        # where the digit code's fragility lives
uv run python probe_maxmargin.py --d 32                # is trained = max-margin of its own geometry?
uv run python probe_optimizers.py --d 32               # does any optimizer beat converged Adam?
uv run python probe_badcombo.py                        # the post's unexplained bad architecture
uv run python probe_patterns.py                        # gate churn + pattern/decision co-sizing
uv run python probe_gatequality.py --phase metrics     # what makes a gate good? (then: predict, drift [--pressure spread], curve)
uv run python plot_frontier.py                         # results/frontier.png, the whole plane
uv run python plot_gatecurve.py                        # results/gatecurve.png, when the gate is built
uv run python probe_flippolicy.py --phase dense        # which flips build it? (then: stats, interp, direction, stride, softseed)
uv run python probe_ordering.py                        # order-structure gate statistics (theory.md 6'')
```

## Layout

```
handcode/data.py        fact generation (verbatim from the post)
handcode/model.py       the Figure-4 architecture + the trained baselines
handcode/connection.py  assigning S neurons to each label
handcode/handcoded.py   the authors' silence construction
handcode/readouts.py    closed-form unembeddings (ridge, tie-break correction)
handcode/coincidence.py the active-pattern rectangle-detector embedding
handcode/linsolve.py    the value code (reference implementation)
handcode/fastsolve.py   batched solver for the value code
handcode/twosided.py    the two-sided value code -- the ReLU is the gate
handcode/digitcode.py   the redundant value code -- label digits on neuron groups
handcode/softseed.py    ridge-built soft seeds for the stride flow (fit + near-ties, no GD)
handcode/capacity.py    binary search over n_facts, sweeps, scaling fit
probe_coding.py         density, weight scale, pattern-vs-magnitude probes
probe_reachability.py   does gradient descent stay at the construction?
probe_robustness.py     noise tolerance, load-matched vs trained (FINDINGS.md §5)
probe_walkaway.py       which facts Adam sacrifices when it walks off (§6)
probe_fixed_readout.py  Adam under the fixed quadratic decode (§7)
probe_structure.py      radii, readout rank: the trained solution's shape (§8)
probe_digitcode.py      the digit-code capacity-robustness frontier (§9)
probe_pedestal.py       fragility attribution and the t0 sweep (§9)
probe_maxmargin.py      max-min-margin LP on frozen geometries (what-gd-builds)
probe_optimizers.py     Adam vs SGD vs L-BFGS capacity ladder (§10)
probe_badcombo.py       the post's unexplained bad architecture combo (§11)
probe_patterns.py       gate churn during training, pattern/decision co-sizing (§12)
probe_gatequality.py    gate-quality metrics, LP ceilings, drift, quality curve (§14-16)
probe_flippolicy.py     flip stats, edge interp, direction test, stride flows, soft seeds (§17-22)
probe_ordering.py       rank-space gate statistics -- the ordering-invariant probe (§23)
plot_robustness.py      results/robustness.png from the probe output
plot_frontier.py        results/frontier.png, the capacity-robustness plane
plot_gatecurve.py       results/gatecurve.png, gate quality along the training run
docs/twosided-construction.md   why the two-sided construction works
docs/what-gd-builds.md          toward the constructive account of trained storage
docs/theory.md                  formal notes: realizability, free flips, edge geometry
reference/              the authors' own source, fetched for cross-checking
reference_post.md       the full post text
```

`reference/` and `reference_post.md` are **not my work** — they are Linsefors & Bushnaq's,
included so the differential test can check against their code rather than against my
reading of their post. See [reference/README.md](reference/README.md) for provenance.

## Deviations from the original

* **3 attempts per cell, not 11.** Fewer attempts can only lower measured capacity. At
  d=16 this costs the `trained` baseline about 20% (496 vs 600 at 11 seeds); from d=32 up,
  3 and 11 seeds give identical answers.
* **Simulated annealing** builds the connection matrix (inherited from the authors'
  `hc2.py`). A strict reading of "no generic black-box optimizer" would catch it, though its
  objective never sees the facts or the task loss. `linsolve` does not use it at all.
* **Gather instead of one-hot matmul** — exactly equivalent (asserted in the tests), but
  drops a factor of `n_vocab` from every training step.
