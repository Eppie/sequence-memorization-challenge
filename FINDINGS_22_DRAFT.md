## 22. Replication guard: §§14–21 across three fact seeds and two dimensions

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
