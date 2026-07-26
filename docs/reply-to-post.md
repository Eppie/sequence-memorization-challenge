# Draft reply to the post

*A draft comment for [Linsefors & Bushnaq's challenge
post](https://www.lesswrong.com/posts/KWtchKwwnJkd4bwCi/challenge-hand-coding-weights-for-efficient-sequence-1),
summarizing this repo's results in the order the authors are likely to care about
them. Numbers at d=32 unless noted; everything is reproducible from
<https://github.com/Eppie/sequence-memorization-challenge>.*

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
implicit-bias signature — on magnitude-graded activations (at capacity, binarizing
trained activations keeps 9% of accuracy; ~5 bits/neuron are needed), read out through
a high-rank, ladder-free codebook (effective rank ~d/2, essentially zero
linear-in-class structure), with no stabilizing pedestal because patterns are
stabilized dynamically. Supporting experiments: (1) fixing the readout to my
construction's decode and training only the embeddings still reaches the trained
model's full capacity — the unembedding's `d²` parameters are an optimization aid, not
storage — and still stalls short of `4d²`, so the gap to the construction is the
optimizer, not the objective's taste for margin. (2) No standard optimizer beats
converged Adam (SGD+momentum and L-BFGS both do worse), so the missing capacity is not
reachable by a better generic trainer. (3) A redundant "digit code" family
(label digits carried by neuron groups, still ridge-only) recovers about two of the
three-and-a-half orders of magnitude of the robustness gap at 76% of trained capacity;
the residual ~30× is precisely the inequality-packing/margin work. (4) The max-margin
linear program on a trained model's own frozen pattern and readout reproduces the
trained model's robustness (σ90 4.6e-2 vs 4.8e-2 at d=16; 1.5e-2 vs 1.6e-2 at d=32) —
to first order, the trained model *is* the max-margin point of its own active-set
geometry — while every mixed condition is *infeasible*: a random codebook cannot
decode the trained pattern, the trained readout cannot separate a random pattern, and
a random pattern with a random codebook cannot hold even half of trained capacity at
any positive margin. The capacity lives in the joint adaptation of gate and codebook;
the margins come free once the geometry is right.

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
