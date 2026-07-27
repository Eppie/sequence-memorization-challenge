# Formal notes on the gate: realizability, free flips, and what the edge measurements mean

*Companion to `FINDINGS.md` §§13–18. Everything labeled **Lemma/Theorem/Proposition**
is proved here; **Corollary 2.1**'s counting is exact combinatorics; **Proposition 5**
is a first-order argument and labeled as such; **Conjecture 6** is open and under
test. Numerical checks cited inline were run against the repo's own gate data
(epoch-180/200 edge states, d=32, n=1584).*

## 0. Setup

The model is the post's Figure-4 MLP: tokens a, b ∈ [V] with V = 2d, embeddings
u, v ∈ R^{V×d}, readout W ∈ R^{L×d} with L = d. For fact f = (a_f, b_f, ℓ_f):

    p_{f,j} = u_{a_f, j} + v_{b_f, j}          (pre-activation)
    h_f     = relu(p_f)                        (hidden)
    logits  = h_f W^T

The **gate** (pattern) is P_{f,j} = 1[p_{f,j} > 0] ∈ {0,1}^{n×d}. The
**masked-linear model** for a frozen P replaces relu(p_f) by P_f ∘ p_f; when (u,v)
realizes P (sign-consistency), the two coincide. The **margin** of fact f is
gap_f = logit_{ℓ_f} − max_{c≠ℓ_f} logit_c.

Two properties of a pattern matter throughout:

* **REAL(P)** — P is *additively realizable*: ∃ (u,v) with sign(u_{a_f,j} +
  v_{b_f,j}) matching P on every observed (fact, neuron) cell.
* **STOR(P)** — the fact set is *storable on P*: ∃ (u, v, W) (in the weight box)
  with every masked-linear margin > 0.

The two-LP ascent (`probe_geometry_ascent.py`) measures a lower bound on the best
robustness available inside STOR ∩ REAL; §§14–18 call its σ90 the pattern's
**ceiling**.

## 1. The measurement apparatus is sound

**Proposition 1.** Fix P and the weight box.
(a) The max-min-margin LPs are never infeasible in the LP sense: u = v = 0 (or
W = 0) makes every logit 0 and every margin 0, so γ* ≥ 0 always. "Infeasible" in
§§14–18 means exactly γ* = 0: *no positive margin exists*, i.e. ¬STOR(P).
(b) From a feasible start, the two-LP alternation is monotone: each half-step
maximizes the true min margin over its block with the other block fixed, and the
current point is feasible for that block problem, so the min margin never
decreases.
(c) If additionally the embeddings satisfy the sign-consistency rows, the
masked-linear model *is* the ReLU model, so a positive masked margin certifies
true storage at accuracy 1.

*Proof.* (a) is the exhibited point. (b): the block LP's optimum is ≥ the value at
the current point by feasibility. (c): where P_{f,j}=1 the constraint p ≥ 0 gives
relu(p) = p = masked value; where P_{f,j}=0, p ≤ 0 gives relu(p) = 0 = masked
value. ∎

## 2. Which patterns are realizable at all

Fix a column j and drop the subscript. Write x_a = u_a and y_b = −v_b; the column's
constraints are x_a > y_b on active observed cells A and x_a ≤ y_b on inactive
observed cells I.

**Theorem 2 (realizability = no strict alternating cycle).** Build the digraph D on
nodes {x_a} ∪ {y_b} with a *strict* edge y_b → x_a for each (a,b) ∈ A and a *weak*
edge x_a → y_b for each (a,b) ∈ I. The column is realizable iff D contains no
directed cycle with at least one strict edge. Columns are independent, so a full
pattern is realizable iff every column is.

*Proof.* Necessity: along any directed edge the head is ≥ the tail (strictly for
strict edges), so a directed cycle containing a strict edge forces z > z.
Sufficiency: suppose no such cycle. Define φ(node) = the maximum number of strict
edges on any directed path ending at that node. φ is finite: a path longer than the
node count revisits a node, and the enclosed cycle contains no strict edge, so
pruning it does not reduce the strict count. For a weak edge x → y every path
ending at x extends to y, so φ(y) ≥ φ(x); for a strict edge y → x likewise
φ(x) ≥ φ(y) + 1. Assigning each node the real value φ(node) satisfies every
constraint (unobserved cells are unconstrained). ∎

**Total-grid case.** If every (a,b) cell is observed, realizability is equivalent
to the absence of a single alternating 2×2 — cells (a,b) ∈ A, (a,b′) ∈ I,
(a′,b′) ∈ A, (a′,b) ∈ I — which is the length-4 cycle of Theorem 2; and a
realizable total column is a *staircase*: sort tokens by x and y values and the
active region is a Ferrers diagram. (Each row's active set is {b : y_b < x_a}, a
lower set of one fixed order on b, so the row sets form a nested chain.) For
*partial* observation the 2×2 test is necessary but not sufficient — longer
alternating cycles bind. Measured on the repo's data: a 50/50 per-bit hybrid of the
epoch-180 and epoch-200 gates (which differ on only 2.3% of bits) has only a
handful of 2×2 obstructions yet **19 of 32 columns are LP-unrealizable** — the long
cycles do the work. The gates themselves: 32/32 columns realizable, zero
obstructions; a density-matched random pattern: 0/32 columns, thousands of 2×2s.

**Corollary 2.1 (scarcity).** A realizable total column is determined by the nested
chain structure: at most V! orderings of the y's times (V+1)^V row thresholds, so
at most 2^{log₂ V! + V log₂(V+1)} distinct realizable columns. At d = 32 (V = 64):
≤ 2^{296+385} = 2^{681} realizable column patterns, against 2^{1584} arbitrary
patterns on the n = 1584 observed cells — a fraction below 2^{−900} per column,
below 2^{−28000} for the full 32-column gate. Restriction to observed cells only
shrinks the realizable side. **Almost no pattern is the pattern of any additive
model.**

**Corollary 2.2 (per-bit hybrids are not experiments).** Mixing two realizable
patterns bit-wise leaves realizability with no protection — the mixture generically
creates alternating cycles (measured above: 19/32 columns dead at 2.3% mixing). Any
intervention on patterns must therefore move in *embedding space* (as §§17–18's
interpolations and direction steps do), where every point realizes its own pattern
by construction.

## 3. Flips are free along continuous paths

**Lemma 3 (free crossings).** Along any continuous path in embedding space, every
logit, every margin, and the loss are continuous, and the pattern changes only at
points where the flipping coordinate's pre-activation is exactly 0 — where its
contribution to every logit is relu(0)·W = 0. A sign flip therefore carries no jump
in any margin.

*Proof.* p is linear in (u,v) and relu is continuous, so h and the logits are
continuous piecewise-linear in the path parameter; the pattern is constant on the
open cells of the induced hyperplane arrangement and changes only on cell
boundaries, where p_{f,j} = 0 and the flipped coordinate contributes 0. ∎

This is §17's measurement — flipping bits sit at |p| ~ 0.002 against a population
at ~1 — stated exactly: gradient descent never pays for a flip because *no*
continuous path pays for a flip. The failed drifts of §§14–15 also paid nothing per
flip (`capped_step` stops at crossings); what they got wrong was not the cost of
flips but their *choice*.

## 4. What the edge distance means, and what §18 does and does not say

**Lemma 4 (segments flip Hamming-minimally).** Along a straight segment in
embedding space each p_{f,j} is affine in t, so each bit flips at most once. The
segment from x₀ to any x* strictly realizing P′ flips exactly the cells where
sign(x₀'s pattern) and P′ disagree. Consequently, for any realizable target set:

    min flips to reach it from x₀ = min over realizable P′ in the set of
    Hamming(P₀, P′),

and a single straight segment achieves the minimum. (Genericity: no cell identically
zero along the segment.) Verified exactly on the edge data: the segment from the
epoch-180 to the epoch-200 state crosses 1166 cells = their Hamming distance.

**Remark 4.1 (the sharp reading of §18).** By Lemma 4, single crossing *directions*
exist in abundance — the line to any storage-feasible realizer is one, and the
realized-training-delta row of §18 is exactly such a line. What §18 establishes is
that no *locally computable* direction among the natural candidates — the loss
gradient, the optimizer's own next step, fit-pressure and margin LP optima — is
one, at matched flip budget. Finding a crossing direction is, by Lemma 4,
*equivalent to finding a storage-feasible realizable pattern near P₀* — which is
the construction problem itself. Gradient descent solves it only as a path
integral; §18's gradient and Adam rows say its own instantaneous data does not
contain the answer either. That equivalence is why the direction experiment could
not have been a shortcut: any oracle that names a crossing direction has already
solved the problem the challenge poses.

## 5. Why margins and robustness decouple (first-order)

**Proposition 5 (radii govern σ90; margins do not).** Perturb each weight matrix M
by σ·rms(M)·ξ_M with ξ i.i.d. standard Gaussian (the repo's noise model). To first
order the change in fact f's margin gap_f is Gaussian with standard deviation
σ · s(∇gap_f), where s(·) aggregates the per-matrix rms-scaled gradient norms. Fact
f survives while

    σ ≲ radius_f := gap_f / s(∇gap_f),

so σ90 tracks a lower quantile of the radius distribution — the quantity
`probe_structure.per_fact_radii` measures — not the margin distribution. This is
first-order in σ and ignores cross-fact correlation; the repo's measured radii
reproduce measured σ90 across models (§8), which is the empirical license for the
approximation.

**Corollary 5.1 (the §13 decoupling).** Box-constrained max-margin ascent can grow
gap_f by scaling weights toward the box corners, which grows s(∇gap_f)
proportionally: margins rise, radii — and σ90 — do not. Measured: the two-LP ascent
raises the digit gate's min margin 1000× and σ90 only 1.3× (§13). Margin per unit
weight, not margin, is the robustness currency.

## 6. The stride conjecture — now CONFIRMED

**Conjecture 6 (stride conjecture; confirmed, `FINDINGS.md` §19).** Let x₀ be an
infeasible fitting-phase state (train accuracy ≈ 0.87, ¬STOR(P₀)). Consider the
iterated process: at each round, solve an exact-solve direction oracle at the
*current* state (fit-pressure spread LP in *both* blocks — the max-min readout
refit is degenerate off STOR by Prop 1(a), and observably fatal), step along it
with the order-statistic rule capped at gradient descent's own per-step gross flip
rate (~0.5% of bits), re-read the pattern, refit; no memory. Measured: the
pattern enters STOR within **four** rounds (0.87% of bits) at a ceiling matching
gradient descent's own crossing gate, and improves monotonically thereafter
(1.80e-2 at forty rounds — 5.7× the best one-shot-era construction, ~2.4× from
the trained ceiling).

The confirmed statement replaces the "gradient field" reading of §18: one-shot
fails, iteration succeeds, and the ingredient is the re-solve loop, not the
specific field being integrated. Two follow-ups sharpened it further
(`FINDINGS.md` §§20–21): the oracle can be as cheap as a subgradient matvec
(exactness is not the ingredient; step granularity is), and the flow's seed
requirements were measured — it bootstraps from any GD state past ~half-fit
(threshold in GD-epochs (50, 100]) but not from scratch, and cheap oracles fail
outright on stiff constructed geometry, where the near-tie reservoir that
GD-built states maintain (0.5% of bits within ~10⁻³ of zero, ~12× denser than
the pedestal construction's) is absent.

**Resolution of the seed problem (6′; `FINDINGS.md` §22).** The seed is
constructible: a ridge-built state at train accuracy 0.63 (0.52 also works,
crossing later) with its readout rescaled to GD's rms bootstraps the fw flow
past the GD-seeded runs — crossing at §19's own 1.45e-2, reaching 3.0–3.2e-2
by round 400 with no plateau, ≤1.4× from the trained full-budget ceiling —
with no gradient descent anywhere in the ancestry. Two attributions of §20 are
thereby revised: the property that gated the cheap flows was the readout's
τ-saturation regime (a 19×-undersized readout leaves every fit fact below the
cap, so fit facts never stop pulling — §14's churn), not embedding stiffness;
and the near-tie reservoir is a second-order stabilizer (longer feasible
windows under the cheapest readout), not the gatekeeper. What gradient descent
retains is a residual ceiling edge of ≤1.4× that shrinks with the flow's round
count — possibly nothing but iteration budget.

**Open problem 6″ (the ordering invariant — first pass refuted).** By
Theorem 2 a gate *is* d token-orderings plus thresholds. §14 refuted
magnitude/correlation statistics as quality predictors; `FINDINGS.md` §23 now
refutes the first pass over *order-structure* statistics at the same depth: ten
rank-space statistics (softness, boundary attention, ordering diversity,
interleaving, threshold dispersion, fact degree, label structure) fail to
separate good gates from bad within matched provenance — the ridge seed and the
fw product grown from it (ceiling 0 vs 2.31e-2, 7.6% of bits apart) agree on
every one, as do ep180 and the trained gate. The statistics that do separate
detect provenance, not quality, and the seed row kills those as predictors.
Still open in principle: joint/higher-order statistics of the permutation
ensemble relative to the fact hypergraph. The candidate softness invariant is
resolved: a stabilizer, not a quality predictor (§22).

**Open problem 6‴ (incompressibility — now three-legged).** The standing
alternative: no state-local description predicts gate quality, and the process
is the shortest description of its own fixed point. Evidence: state statistics
are blind in both magnitude (§14) and order (§23) coordinates; the building
delta's marginals are structureless (§23 — near-boundary, uniform over columns,
labels, and tokens, error-agnostic, for GD's window flips and the GD-free
flow's alike); and naming a crossing direction is equivalent to solving the
construction problem (Remark 4.1). The missing theorem is a lower bound over a
natural oracle class. The companion quantitative question has nearly dissolved:
the ceiling advantage of gradient descent's full trajectory over hand-specified
flows, ~1.8× at §21's horizons, is ≤1.4× at round 400 and still shrinking
(`FINDINGS.md` §22) — it may be iteration budget, not integrand.

## 7. What is proved, what is measured, what is open

Proved here: the apparatus (Prop 1), the complete characterization and scarcity of
realizable gates (Theorem 2, Corollaries 2.1–2.2), the zero cost of flips along
continuous paths (Lemma 3), and the identification of flip-budget geometry with
Hamming distance to the feasible realizable set (Lemma 4) — with Remark 4.1
reducing "find a crossing direction" to the construction problem itself. Measured
(FINDINGS): where in training the gate becomes good (§16), the triviality of the
flip policy (§17), the failure of every one-shot direction (§18), the success of
the iterated fit-pressure process (§19) — Conjecture 6 confirmed — and the
resolution of the seed problem (§22): a ridge-built half-fit seed with a GD-scale
readout serves, so no stage of the pipeline needs gradient descent. Open: the
declarative question, now sharpened to its hard core — first-order state and
delta statistics are refuted in both coordinates (§§14, 23), so either a
higher-order invariant exists, or the incompressibility bound (6‴) is a theorem.
Beneath it, the last quantitative residue: the ~1.8× ceiling edge of gradient
descent's full trajectory over every hand-specified flow. §18 plus §19 locate
the mystery in the *loop*; §§22–23 add that the loop needs no privileged seed
and leaves no statistical fingerprint — direction recomputation after each
step's flips remains the one ingredient every failure lacked and every success
has, and nothing measurable about its products yet explains why.
