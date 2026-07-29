# Draft reply to the post

*A draft comment for [Linsefors & Bushnaq's challenge
post](https://www.lesswrong.com/posts/KWtchKwwnJkd4bwCi/challenge-hand-coding-weights-for-efficient-sequence-1),
summarizing this repo's results in the order the authors are likely to care about
them. Numbers at d=32 unless noted; everything is reproducible from
<https://github.com/Eppie/sequence-memorization-challenge>.*

*STATUS: DRAFT — posting is the maintainer's call. The replication guard
(FINDINGS §24) has landed: §§14–21's headline rows reproduce at fact seeds 43
and 44, and the feasibility split holds at d=64; what does not transfer is
epoch locations, not mechanisms. Still single-seed: the §§22–23 seed/flow
results and the §§25–26 description-length results.*

---

I took up the challenge, and I want to report both halves honestly: a construction
that beats the trained model on your benchmark, and measurements showing that this
says less than it seems to — plus what those measurements reveal about the thing you
actually asked for, "a weight-level account of how the storage in real models works."

**The entry.** Store each fact's label as a number — the sum of the active hidden
units — but build no gate: let the ReLU's own sign pattern be the gate, freeze it,
solve the resulting linear system by per-token ridge regressions, re-read the pattern,
repeat (with a step-length cap; details in the repo). Every explicit gate spends one
embedding matrix and caps fact-carrying parameters at `2d²`; the ReLU-as-gate frees
both matrices, and measured capacity follows the `4d²` count: 3168 facts at acc=1 at
d=32 vs. 2080 trained, growing to 2.0–2.3× trained at d=128. Your published trained
and hand-coded curves reproduce to within a few percent throughout, so the comparison
is like-for-like. (One caveat on your baselines: the ≤5000-epoch recipe under-trains
at d=16/32 — converged Adam reaches ~2560 at d=32, not 2080 — but is already converged
at d=64, so the construction's lead survives against converged training from d=64 up.)

**Why this is not the understanding you asked for.** Your instinct about Dugan-style
exact solves ("very unlike something models trained with gradient descent would or
could learn") is quantitatively right, and the axis it is right on is noise tolerance.
Perturb each weight matrix with Gaussian noise scaled to its own RMS and find the
level where accuracy drops below 0.9: at matched load, a trained model tolerates
~1300× more relative weight noise than my construction (2.0e-2 vs 1.5e-5 at n=2080;
replicated at d=64). In absolute units the construction is broken by about **one ninth
of a single Adam step**, while trained solutions carry 5–20 steps of cushion — a
solution can only be found and held by an optimizer whose own churn it survives.
Meanwhile *your* hand-coded construction sits within ~2× of trained robustness. You
were playing the right game; the exact solves win the benchmark by leaving that game.
**Suggested amendment: report σ90 (noise tolerance) alongside max facts, or score
capacity under weight noise of about one optimizer step.** Under either, exact-solve
entries lose their advantage and the benchmark tracks your stated goal.

**What gradient descent actually builds** (each clause is a measurement in the repo):
a margin-floor-equalized solution of the storage *inequalities* — continued training
raises the worst per-fact weight-space radius 4.4× faster than the median, the classic
implicit-bias signature — on magnitude-graded activations, read out through a
high-rank, ladder-free codebook, with no stabilizing pedestal because patterns are
stabilized dynamically. The load-bearing object is the ReLU sign pattern (the "gate"):
freeze a trained model's gate and the max-margin linear program reproduces its full
robustness, while every mixed condition (random gate, trained readout; trained gate,
random readout; random/random) is infeasible outright. So the question "how does
storage work" reduces to "what makes a gate good, and how is one built" — and I
chased that to the bottom. The short version:

1. **Gate quality is invisible to every state-level description I could test.**
   Not magnitude statistics (density, correlations, conditioning — a random gate
   matches a trained one on all of them and cannot store half the load at any
   margin), and not *order-structure* statistics either: an additive gate column is
   exactly a token ordering plus thresholds, and ten rank-space statistics, a joint
   fragile-bit-placement statistic, and the flip-set's marginals all fail to
   separate storage-capable gates from worthless ones of matched provenance. The
   information is there — the gate decides everything — but no first-pass invariant
   names it.

2. **No one-shot solve builds it, but iterated fit pressure does — and nothing
   about gradient descent's specifics is needed.** Every single-step direction at a
   matched flip budget fails (loss gradient, Adam's own next step, fit-pressure and
   max-margin LP optima). But the same pressure *re-solved every ~0.5%-of-bits
   step* crosses into storage-feasibility and keeps building. The oracle can be as
   cheap as a sign-snapped subgradient; backprop-exact gradients, Adam,
   cross-entropy, and LP-exact directions are all measured as non-load-bearing.
   What is load-bearing: step granularity, fit pressure in both weight blocks, and
   a saturation cap so already-fit facts stop pulling.

3. **The seed is constructible too.** A ridge-only recipe (freeze the ReLU pattern,
   ridge the readout to one-hot targets, per-token ridge sweeps, flip-capped steps
   — the same moves your construction is allowed) reaches ~60% train accuracy, and
   from that seed the cheap flow climbs monotonically to σ90 = 3.46e-2 — within
   ~1.3× of the fully trained model's 4.4e-2, with no gradient descent anywhere in
   the pipeline's ancestry. From *scratch*, the flow fails: roughly half a fit is a
   real prerequisite, but ridge regression suffices to build it.

4. **What remains of "only gradient descent can do this" is one bounded number:**
   at matched iteration count, GD's gate ceiling is ~1.24× higher than the
   hand-specified flow's, shrinking slowly with flow rounds. Everything else —
   optimizer specifics, the seed, the claimed special geometry of GD states — is
   eliminated by construction.

The uncomfortable conclusion I keep failing to escape: the robust solution seems to
be **process-shaped, not description-shaped**. Four independent results point the
same way — state statistics are blind in both magnitude and order coordinates, the
gate-building flip set has no imitable structure (near-boundary cells, uniform over
columns/labels/tokens, error-agnostic — the *choice* carries the quality), naming
a single crossing direction is provably equivalent to solving the construction
problem itself, and forward-designed gates fail wholesale: Hadamard token
signatures, hash-balanced staircases, thermometer digit codes, and even the
trained gate's own per-neuron score distributions with the token assignment
shuffled are all infeasible at one LP pass. The content also now has a measured
size: the trained pattern's description compresses to ~2.5 bits per embedding
score — a three-level alphabet already names a pattern above my own best
construction — yet resists every *structural* compression tried (rank, token
clustering, sparsity) at matched distortion. Ten kilobits of pure co-adaptation,
with nothing anyone has found to say about them shorter than the process that
builds them. I would genuinely love a counterexample: a
declaratively chosen gate that stores at trained robustness would refute this and
win your challenge in the strong sense. I could not find one, and I now believe
the interesting theorem is the lower bound. One figure with everything on it:
`results/frontier.png`, the capacity-robustness plane.

A note on rules: I do *not* count the iterated-fit-pressure pipeline as a
hand-coding entry, even though it is gradient-free — iterating solves against the
task's own margins is training under a different name. The construction ledger I
defend is: capacity benchmark, the two-sided code (3168 vs 2080 trained at d=32);
best *declarative* gate under the robustness bar, 2.85e-3 (ridge-built digit code +
max-margin ascent, ~15× short of trained); best gradient-free *process* artifact,
3.46e-2 (~1.3× short). The gap between the last two is, as far as we can measure,
the entire content of what training contributes.

**Your appendix-B outlier.** MLP+Norms+NoRes+NoBias+ReLU: rerunning your model with
each setting flipped one at a time, the deficit is real (0.68 vs 0.86–1.00 for every
one-flip control at n=768), is *not* cured by 12× budget and 100× patience, and is not
dead units at scoring time — though extended training progressively kills whole facts'
hidden layers (1.8% → 12%), a ratchet. Reading of the controls: that combination is
the unique one with **no slack mechanism** — GELU leaks gradient, a bias shifts
thresholds, a residual bypasses, removing the norm frees the scale — and each single
valve restores most capacity. So: training dynamics, but of a structural kind that
more training does not fix.

Code, probes, and full writeups: the repo above — `FINDINGS.md` for the measurements,
`docs/twosided-construction.md` for the entry, `docs/what-gd-builds.md` for the
assembled account and what would finish it.
