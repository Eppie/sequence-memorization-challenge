# What gradient descent actually builds

*Toward the constructive account the challenge is really asking for. Everything here is
measured on the Figure-4 toy model; sections cite the probe that produced each number.
Status: the negative half of the account is established; the constructive half has hit
a measured wall — gate quality is real, decisive, and (per `FINDINGS.md` §§14–15)
neither statistically characterizable nor reachable by exact-solve co-adaptation under
either pressure structure. §§15–16 relocate where gradient descent builds it: in a
narrow window at the end of the fitting phase, not the late consolidation.*

The challenge asks for an algorithm that stores facts the way a trained model does. The
capacity benchmark turned out to be answerable without answering that question — the
two-sided value code beats trained capacity while being three orders of magnitude more
fragile than anything gradient descent produces (`FINDINGS.md` §5). So the real question
splits in two: **what solution does gradient descent build**, and **can it be built
without gradient descent?**

## 1. The three facts any account must fit

From `FINDINGS.md` §5–8, at d=32 unless noted:

1. **Robustness class.** Trained solutions tolerate relative weight noise σ90 ≈ 1–60×
   10⁻² depending on load — always a few multiples of the optimizer's own step size.
   Every exact-solve value code sits at σ90 ≈ 10⁻⁵, below a single Adam step. The
   authors' own hand-coded silence construction is *inside* the trained class.
2. **Margin-floor equalization.** Per-fact weight-space radii (margin/grad-norm): the
   trained model's are broad (CV ≈ 0.7–0.85) with a high floor that *continued training
   raises 4.4× faster than the median*. The construction's are uniformly thin (CV 0.13).
   Gradient descent is doing max-margin work on the worst facts — the implicit-bias
   story, visible in the weights.
3. **A high-rank, non-graded readout.** The trained unembedding has effective rank
   ~12–18 of 32 and essentially zero linear-in-class ladder structure (graded fraction
   0.03), while every value code reads out through a rank-1-or-2 ladder. Trained
   activations are magnitude-graded (`FINDINGS.md` §1–3), but the *readout geometry* is
   a distributed codebook, not a number line.

## 2. Equality codes cannot get there at any redundancy: the digit-code frontier

The value codes' fragility has an information-theoretic reading: they pack all log₂(d)
bits of a label into one coordinate read at precision 1/d. The natural repair is
redundancy — spread the label's digits over `m` coordinates at precision `d^(1/m)` each,
paying `m` equations per fact (capacity ceiling `4d²/m`) for per-coordinate margins.
`handcode/digitcode.py` makes that constructive: write the label in base
`p = ⌈d^(1/m)⌉`, partition the neurons into `m` groups, and require each group's
active-sum to carry one digit — `m` parallel copies of the two-sided mechanism, still
solved entirely by ridge regressions. At `m = 1` it *is* the two-sided code.

`probe_digitcode.py` traces the family's (capacity, σ90) frontier and trains a model at
each frontier point for a load-matched comparison; `probe_pedestal.py` then localizes
the remaining fragility and removes what can be removed. Result (d=32; capacity is the
largest grid point solved at acc=1 under a fixed 1000-round budget):

| m | levels per coordinate | capacity (ceiling `4d²/m`) | σ90 at capacity | trained σ90, same n |
|---|---|---|---|---|
| 1 (= twosided) | 32 | 3168 (3968) | 1.6e-5 | — (above trained capacity) |
| 2 | 6 | 1584 (1984) | 1.1e-4 | **4.3e-2** |
| 3 | 4 | 1056 (1322) | 1.4e-4 | 1.0e-1 |
| 5 | 2 | 392–512 (793) | 3.5e-4 | 2.1e-1 |

**Redundancy alone buys a factor of ~20 across the whole family while capacity falls
8×** — the frontier stays 400–700× below the trained point everywhere. So "spread the
label over more coordinates" is *not*, by itself, the missing ingredient.

Where the fragility actually lives (`probe_pedestal.py`): perturbing only the readout
reproduces the full σ90, while perturbing only the embeddings is ~9× more tolerable.
The culprit is the **pedestal** — the stabilizer `t0` that every frozen-pattern equality
solve in this family needs to keep the ReLU signs still. It inflates every activation
to O(t0), and readout noise is amplified by ‖h‖. It is also partly removable: sweeping
`t0` down from `16d` to `d/2`, σ90 rises roughly as `1/t0` to **1.3–1.6e-3**, the
embedding RMS falls to trained scale (~3, from ~70), the solve still reaches 100% at
1584 facts — and below `t0 ≈ d/4` the pattern-stability trick stops working and the
solve breaks. That is the equality-code family fully optimized: **n = 1584 (76% of
trained capacity) at σ90 = 1.3e-3, a residual 30× short of the trained model at the
same load.**

The decomposition of the original ~1300× robustness gap at matched load, then:

* ~10× — digit redundancy (m = 2–3 instead of everything in one coordinate);
* ~10–20× — shrinking the pedestal to the edge of solvability;
* **~30× residual** — what no equality-constrained solve in this family reaches.

The residual has a name. An equality solve pins the decode to a point and spends no
slack anywhere (§1's uniformly-thin radii); gradient descent satisfies argmax
*inequalities* and pushes every fact's margin up until the floor equalizes. Inequality
cells are regions, not points — they pack more facts per parameter at the same
tolerance, and their slack *is* the robustness. That is also why the trained readout is
high-rank and unstructured (§1): margin-maximizing directions have no reason to lie on
a ladder.

## 3. The account, as it stands

Putting §1 and §2 together with `FINDINGS.md` §7 (the readout is an optimization aid,
not storage; the capacity gap is the optimizer, not the objective):

**Gradient descent builds a margin-floor-equalized solution of the storage
*inequalities*, on a magnitude-graded substrate, read out through a high-rank
(effective rank ≈ d/2, ladder-free) codebook, pedestal-free because its patterns are
stabilized dynamically rather than by offset, with the margin floor set by what
survives the optimizer's own churn.**

Each clause is now a measurement: the floor-equalization is §1.2 (continued training
lifts the minimum radius 4.4×), the codebook shape is §1.3 (rank90 = 12 at d=32, 24 at
d=64 — it scales with d, and the graded-ladder content is 0.03), the pedestal-free
clause is §2 (the equality codes' dominant, and only partly removable, fragility is the
offset they need where gradient descent needs none), and the churn-sized floor is
`FINDINGS.md` §5 (tolerance = 5–20 optimizer steps at every load).

**The keystone test has now run** (`probe_maxmargin.py`; HiGHS interior-point with
cutting planes, scipy added to the project for it). Freeze a trained model's own ReLU
pattern and readout; the margin and pattern-consistency conditions are linear in the
embeddings, so the max-min-margin point is a linear program. The LP solution
reproduces the trained model's robustness — σ90 4.6e-2 vs 4.8e-2 at d=16, 1.6e-2 vs
1.6e-2 at d=32 — and its radii distribution. The dynamic-stabilization clause also has
direct evidence now (`probe_patterns.py`, `FINDINGS.md` §12): the sign pattern's
per-epoch churn collapses 200× within 30 epochs while accuracy is still at 21%, yet
the finished gate differs from its init in 41% of bits — gradient descent rebuilds the
gate gradually and substantially while never churning fast. At the end, gate stability
and decision margin sit at the same noise scale (accuracy fails at 3.2e-2, 1% of gate
bits flip at 1.0e-2), where the exact solve's accuracy dies 316× before its gate
moves. Gradient descent co-sizes every slack; the equality codes spend all of theirs
in one place. To first order, **the trained model is
the max-margin point of its own active-set geometry**: the implicit-bias story stated
as a checkable identity rather than an asymptotic theorem. One refinement the LP adds:
the trained model holds only ~60% of its geometry's optimal *minimum* margin (5.8 vs
9.4 at d=32) at essentially optimal σ90 — robustness saturates before the margin
optimum, and gradient descent stops pushing where pushing stops paying. Equally sharp are the
failures: every mixed condition is *infeasible* — a random codebook cannot decode the
trained pattern, the trained readout cannot separate a random pattern, and a random
pattern with a random codebook cannot hold even 992 facts (48% of trained capacity) at
any positive margin. **The capacity does not live in the margin solve; it lives in the
joint adaptation of gate and codebook that gradient descent performs on the way.**

Quantitatively: of the ~1300× robustness gap between the best capacity construction
and the trained model at matched load, ~100× is now constructively recovered by a
ridge-only algorithm (the pedestal-optimized digit code: 76% of trained capacity at
30× less robustness), a further ~2.5× by exact margin ascent plus one flip-capped
drift step on its gate (σ90 3.18e-3, under spread pressure; `FINDINGS.md` §§14–15),
and the remaining ~14× is the gate-quality deficit that no exact solve in this family
crosses — the part of gradient descent's implicit bias that lives in its coupled
dynamics rather than in its optimum. §15 adds where those dynamics do the work: the
gate at the *first* 100%-accuracy epoch already ascends to σ90 4.07e-2 against the
full-budget gate's 4.40e-2, so the quality is built during error-driven fitting, and
the long margin phase that follows adds only ~8%. Its two known inefficiencies — first-order convergence (`FINDINGS.md` §7)
and spending parameters on margin the metric never scores (§5) — are the price and the
point, respectively.

## 4. What would finish the constructive half

1. **Gate discovery.** The max-margin reconstruction (§3) settled the margin half, and
   two construction attempts (`FINDINGS.md` §13) then narrowed the rest by one more
   level. A ridge-bootstrap pipeline dies immediately (a ridge readout supports zero
   margin even on the trained gate — co-adaptation cannot be started from an
   infeasible point), and a two-LP max-margin coordinate ascent that stays feasible
   raises the digit gate's minimum margin 1000× while improving σ90 only ~2× — still
   ~15× short of trained at the same load, on the same machinery that reproduces
   trained σ90 exactly when given the trained gate. Values, readout, and margins are
   all an LP away; **the one unconstructed object is the gate**.

   The characterization program has now run (`FINDINGS.md` §14), and its answer is a
   sharp negative that changes what "construct the gate" can mean. The candidate
   statistics (per-token design conditioning, same-token active-set decorrelation)
   do not separate the trained gate from a *random* additive gate — or from the
   trained run's own init — yet the LP machinery shows those gates cannot store the
   fact set at all (accuracy 0.03 with embeddings *and* readout exactly optimized),
   while the trained gate supports full trained robustness. Gate quality is
   therefore not an intrinsic property of the binary pattern; it is the relational
   property that the cone of embeddings realizing the pattern's signs contains
   good value embeddings. And the constructive imitation of gradient descent's
   mechanism — pattern drift under exact margin pressure with twosided's flip cap —
   buys exactly one step before iterating destroys the gate entirely, under *both*
   pressure structures: worst-fact max-min (σ90 2.96e-3) and per-fact capped-sum,
   the exact-solve analog of softmax's saturating every-fact pull (σ90 3.18e-3, the
   best constructed gate to date; `FINDINGS.md` §15). So "pressure from every fact
   at once" is not the missing ingredient either. The remaining ~14× is the
   measured value of gradient descent's coupled small-step dynamics, and §15
   locates where they pay: the first-acc=1 gate already ascends to 4.07e-2 of the
   full-budget 4.40e-2, so the quality is built during error-driven *fitting* —
   before the margin phase every imitation so far has targeted. The first step of
   the trajectory-measurement program has now run (`FINDINGS.md` §16): the
   gate-quality curve over training checkpoints shows the gate is *infeasible*
   through train accuracy 0.77, crosses to 4.5× the best construction's ceiling in
   a ~50-epoch window (train accuracy 0.77 → 0.90, only 4.6% of pattern bits), and
   reaches 92% of its final ceiling at first-acc=1 — the gate becomes good before
   the model does, and the long §12 consolidation is polish worth ~8%. The flip
   policy inside that window has now been measured (`FINDINGS.md` §17), and it is
   trivial: gradient descent flips only extreme near-ties (bits 50–500× closer to
   zero than typical), error-agnostically, with no bit-level signature at the
   feasibility edge — while an embedding interpolation along the training
   direction crosses from infeasible to 3× the constructed record within 0.35% of
   bits. Same capped-step mechanics as the failed drifts, opposite outcome: all of
   gate quality lives in the *direction field* of the embedding dynamics. What
   would finish this item: characterize that direction (it is the aggregated
   fitting gradient, not any margin-LP vertex) and build an exact solve that
   points along it from an infeasible ~90%-fit start; the ~180 decisive bits at
   the edge are an enumerable object for that study.
2. **Codebook targets instead of digit targets.** §1's readout measurement says trained
   labels live on ~12–18 random-ish directions, not on `m` ladders. The digit solver
   generalizes: replace the per-group digit targets with margin *inequality* targets
   toward a random high-rank codebook and solve by iterated ridge with per-fact
   reweighting (greedy, rules-legal). That is the construction the measurements point
   at.
3. **The robustness-qualified benchmark.** Score max facts under weight noise of one
   optimizer step (or report σ90 alongside capacity). Under that metric the account
   above is exactly what a winning entry must reproduce.
