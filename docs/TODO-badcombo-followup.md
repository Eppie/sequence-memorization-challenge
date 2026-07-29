# TODO — replicating and extending the appendix-B outlier comment

**Status:** not started. Written 2026-07-28.
**Owner:** unassigned.
**Prompted by:** a comment on the LessWrong reply from GitHub user `Ameya-bit`,
with their working repo at <https://github.com/Ameya-bit/mem-toy-scratch>.
**Touches:** `FINDINGS.md` §11, `docs/reply-to-post.md` (the appendix-B
paragraph), `probe_badcombo.py`, `results/badcombo.json`,
`results/badcombo_high.json`.

---

## Why this is worth doing

§11 answers Linsefors & Bushnaq's "training dynamics or architectural
capacity?" with *dynamics, of a structural kind*, and explains the rescue as
each single flip adding a **slack mechanism**. The comment does two things to
that account: it makes the dynamics half provable rather than inferred, and it
contradicts one of the named mechanisms.

**Their reported numbers** (d=16, CPU, no-attention block, argmax/CE, "max
facts"; seed count and accuracy criterion both unstated — see T1):

| condition | max facts |
|---|---|
| outlier (no bias) | 504 |
| in-bias only | 488 |
| out-bias only | 1024 |
| head-bias only | 616 |
| norms off | 832 |
| GELU | 888 |

### The load-bearing disagreement

§11's third bullet reads "a bias shifts thresholds without rotating
hyperplanes" — that is the **in**-bias mechanism. The vendored model drives all
three Linear layers from one `bias` flag (`reference/models.py:102,104,109`),
so our "flip bias → on" row turned on in-, out- and head-bias together and
attributed the rescue to the only one of the three that (per their split) does
nothing. **If T3 replicates, §11's explanation is misattributed and must be
rewritten.** This is the main reason the file exists.

### The claim that strengthens us

With no bias and no residual, RMSNorm contributes a per-sample *positive
scalar*, which commutes through ReLU (positively homogeneous) and through
bias-free linear maps; an elementwise learnable gain folds into the next weight
matrix. So **norms-on and norms-off are the same function class here, with
identical achievable argmax patterns.**

Two consequences, the second stronger than what they claimed:

1. Their transplant (norms-off weights scoring 100% when swapped into the
   outlier) is then almost a corollary rather than an experiment — but it is a
   real check that the homogeneity holds in the implementation, epsilon and all.
2. Capacity of the outlier config *equals* capacity of norms-off. Their own
   table therefore already contains a **1.65× pure-optimization gap (504 vs
   832)** with no transplant needed — and our §11 "flip norms → off" row (0.90
   vs 0.68 at n=768) becomes a proof rather than a suggestive control, since the
   two rows are the same function class by construction.

### The constraint they are missing

§11 measured all-dead-hidden-layer facts at **0–2% at scoring time** in short
runs, rising to 12% only under extended training. A 2% dead rate cannot produce
a 30-point accuracy deficit, so death itself is not the mechanism — the damage
happens in the *approach* to death, where `rms(ff_out)` is small but nonzero and
the RMSNorm Jacobian `~1/rms` amplifies. They independently guessed the
amplification (`probe_badcombo.py`'s docstring already hypothesized it), but
they still frame death as the mechanism. Send them the 0–2% number.

---

## Tasks

### T1 — Reconcile protocols before comparing any number *(do first, cheap)*

Their statistic is "max facts"; ours is accuracy-at-fixed-n
(`N_VALUES`, best-of-2 seeds, short budget 5k/pat 100). These are not
comparable as printed.

- [ ] Establish their accuracy criterion for "max facts" (100%? ≥0.9?). Their
      transplant sentence says "100% on 672 facts", which suggests 100%.
- [ ] Note the n ranges differ across our own two result files:
      `probe_badcombo.py` currently has `N_VALUES = (256, 384, 512, 640)` while
      §11's table shows 512–1000 (`results/badcombo_high.json`). Say which file
      backs which row.
- [ ] Decide one statistic for the follow-up. Recommend **max facts at
      acc = 1.0, bisected**, since it is theirs and it is what a capacity claim
      should report — but see the T2 caveat about threshold statistics.

### T2 — Replicate their table with variance *(the gate for everything else)*

Their table looks like single runs. **504 vs 488 is not a difference**, and the
in-bias null — which is the whole basis for the disagreement in T3 — rests on
exactly that gap.

- [ ] ≥5 seeds per condition; report median and full range, not the max.
- [ ] Check whether 1024 is **right-censored**. It is exactly 2¹⁰ and the
      largest entry in the table; if their search grid topped out there, the
      out-bias number is a lower bound and their case is stronger than stated.
- [ ] Treat the result as a threshold statistic and expect it to move. §22's
      replication guard found the trained *ceiling* stable to 1% across fact
      seeds while an epoch-indexed *threshold* moved 1.2–1.4×. Max-facts is the
      second kind.

### T3 — The three-way bias split *(the actual new science)*

**Do not edit `reference/models.py`.** It is third-party verbatim source whose
whole purpose is `tests/test_matches_reference.py`; editing it voids the
"checked against the original" guarantee the reproduction claim rests on
(`reference/README.md`).

Instead build with `bias=True` and zero-and-freeze the ones that should be off:

```python
model = MemoryToyModel(ModelSettings(..., bias=True))
for layer in (model.ff[0], model.ff[2], model.head):   # in, out, head
    if layer not in keep:
        layer.bias.data.zero_()
        layer.bias.requires_grad_(False)               # Adam then skips it
```

(`model.ff[1]` is the activation — see `diagnostics()` in `probe_badcombo.py`.)

- [ ] Add the four conditions (none / in / out / head) to `CONDITIONS`.
- [ ] **RNG caveat:** `nn.Linear` draws its bias init from the same stream after
      the weight, so `bias=True`-then-zeroed does *not* reproduce `bias=False`'s
      weight init at a given seed — later layers see a shifted stream. Either
      seed per layer or accept it and let T2's seed count absorb it. Do not
      compare a single run of each and call a 3% gap real.
- [ ] Confirm or refute in-bias ≈ outlier and out-bias ≫ both.

### T4 — Mechanism: the escape valve must be downstream of the dead gate

Proposed resolution of their open puzzle ("only in-bias can move a ReLU
threshold, but out-bias is what rescues"): once a fact's pre-activations are all
negative, `ReLU' = 0` severs the gradient to *everything upstream, including
`b_in` itself* — so the one bias that could keep units alive receives no
gradient from precisely the facts that need it. `b_out` and `b_head` are
downstream and always get gradient. Of those two, `b_out` sits *before* the norm
so it floors `rms(ff_out)` and kills both the exact-zero state and the `1/rms`
blowup; `b_head` sits *after* it, so a dead fact still normalizes to 0 and it
buys only a constant logit prior.

Predicted ordering **out ≫ head > in ≈ none**; their table reads 1024, 616, 488,
504. Three tests that separate this from the alternatives:

- [ ] **Hand `b_in` the answer.** Initialize `b_in` large and positive so no
      fact starts dead. If the story is "cannot be *learned* because dead facts
      give it no gradient", this rescues the in-bias config. If it does not, the
      story is wrong.
- [ ] **Freeze `b_out` at a random nonzero constant, never trained.** The
      mechanism is flooring `rms`, not learning, so the rescue should survive.
- [ ] **Track `all_dead_fact_fraction` over training per condition** —
      `diagnostics()` already computes it. Expect ~0 for out-bias and a rising
      ratchet for in-bias/none. This also re-measures the 12%-under-long-budget
      ratchet from §11 per bias site.

### T5 — Verify the homogeneity claim in the implementation

- [ ] Numerically: train norms-off to some n, transplant into the outlier
      config, assert identical argmax on every fact (not just equal accuracy).
- [ ] Check the epsilon in RMSNorm does not break it at small `rms` — this is
      the one place the algebra can fail in practice, and it is the same region
      T4 says the damage happens in.
- [ ] If it holds, state the strong version: outlier capacity **equals**
      norms-off capacity, so §11's norms row is a pure-optimization measurement.

### T6 — Fold back

- [ ] Rewrite §11's third bullet if T3 confirms (the bias credit is misassigned).
- [ ] Upgrade §11's "dynamics not capacity" from the budget argument to the
      homogeneity proof, which is strictly stronger and cheaper to state.
- [ ] Update the appendix-B paragraph in `docs/reply-to-post.md` to match.
- [ ] Credit `Ameya-bit` for the bias split and the transplant.

---

## Notes and traps

* `probe_badcombo.py` imports the vendored model by `sys.path` insertion and
  stubs `wandb`. Keep that; it is what makes this "their model, our harness".
* Their run is CPU and ours has been too for this probe; they report reproducing
  the GPU numbers. Torch CPU training is not bit-deterministic (§17), so do not
  expect run-to-run identity even at fixed seed.
* Everything here is d=16, V=32, `output_vocab_size=16`, no attention — the
  post's own appendix-B scale. Do not mix with the d=32 gate-quality cells.
* Machine sharing: LP probes elsewhere in this repo are memory-bandwidth-bound
  and contend badly. `pgrep` before launching a sweep alongside one.

## Open questions

* Why does GELU (888) beat norms-off (832) if the amplification is the dominant
  term? GELU cures the dead gate but leaves the `1/rms` Jacobian intact, so on
  the T4 account it should be the weaker fix. Either the dead gate matters more
  than the 0–2% figure suggests, or the two are not separable this way.
* Does out-bias remain the top condition at larger d, or is it an artifact of
  `d_ff = 16` being small enough that all-negative pre-activation rows are
  common?
* Their closing line — "I also believe this dead-ReLU isn't the full story" — is
  probably right, and the 0–2% number is the reason. What is the rest?
