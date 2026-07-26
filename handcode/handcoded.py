"""The hand-coded construction (post: "Our attempt").

The idea, in one line: a fact with label l is encoded by *all of label l's
neurons being silent*. ReLU gives exact zeros for free, so "all these neurons
are off" is a condition the readout can check exactly -- give label l's neurons
weight -2 into logit l and the correct label is the only one scoring exactly 0.

Building the up matrix, per neuron:
  0. start every weight at +1, so relu(1 + 1) = 2 fires on everything;
  1. list the facts whose label is assigned to this neuron -- it must be silent
     on all of them ("guarded" facts below);
  2. count token frequencies at position 1 among those facts, set the
     `top_fraction` most frequent to -1  (relu(-1 + 1) = 0);
  3. same for position 2;
  4. any guarded fact still uncovered gets both its tokens zeroed
     (relu(0 + 0) = 0);
  5. everything else stays at +1, so the neuron fires on as much else as possible.

Down matrix: -2 from each neuron to the labels it is assigned to, 0 elsewhere.

`build_up_matrix` is the vectorized implementation used everywhere; the
per-neuron loop it replaces is kept as `build_up_matrix_loop`, which reads like
the description above and is checked against the fast path in the tests.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .connection import get_connection_matrix
from .model import ModelShape


@dataclass(frozen=True)
class HandCodedParams:
    """The construction's two hyperparameters (both swept in the post)."""

    S: int  # neurons assigned to each label
    top_fraction: float  # fraction of the most frequent guarded tokens set to -1

    def __str__(self) -> str:
        return f"S={self.S}, top_fraction={self.top_fraction:g}"


def build_up_matrix(
    shape: ModelShape,
    facts: dict,
    conn: np.ndarray,
    top_fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Embedding weights: neuron i is silenced on every fact whose label is
    assigned to i, while firing on as many other inputs as possible.

    Vectorized over neurons. `generator` breaks ties between equally frequent
    tokens at random (as the authors' code does by shuffling before ranking);
    pass None for deterministic lowest-token-id-first tie-breaking.
    """
    inputs, labels = facts["inputs"], facts["targets"]
    n_vocab, d_mlp = shape.input_vocab_size, shape.d_mlp
    conn_t = torch.tensor(conn, dtype=torch.float32)  # (d_mlp, n_labels)

    # guarded[i, f] -- neuron i must stay silent on fact f.
    guarded = conn_t[:, labels].bool()  # (d_mlp, n_facts)

    up = torch.ones(d_mlp, 2 * n_vocab)
    selected = []

    for pos in (0, 1):
        # counts[i, t] = how often token t appears at this position among the
        # facts neuron i guards. One matmul over the per-label token counts.
        per_label = torch.zeros(shape.output_vocab_size, n_vocab)
        per_label.index_put_(
            (labels, inputs[:, pos]), torch.ones(len(labels)), accumulate=True
        )
        counts = conn_t @ per_label  # (d_mlp, n_vocab)

        # Rank present tokens by count, breaking ties randomly; absent tokens
        # (count 0) must never be picked, so push them to the bottom.
        present = counts > 0
        score = counts.clone()
        if generator is not None:
            score = score + torch.rand(counts.shape, generator=generator) * 0.5
        score = torch.where(present, score, torch.full_like(score, -1.0))

        # k = max(1, int(n_present * top_fraction)), per neuron.
        n_present = present.sum(1)
        k = torch.clamp((n_present * top_fraction).long(), min=1)

        order = score.argsort(dim=1, descending=True, stable=True)
        rank = order.argsort(dim=1)
        top = (rank < k.unsqueeze(1)) & present  # (d_mlp, n_vocab)
        selected.append(top)
        up[:, pos * n_vocab : (pos + 1) * n_vocab][top] = -1

    # A guarded fact is covered if either of its tokens was set to -1.
    covered = selected[0][:, inputs[:, 0]] | selected[1][:, inputs[:, 1]]
    remaining = guarded & ~covered  # (d_mlp, n_facts) -- still fires, must not

    # Zero both tokens of every remaining fact: relu(0 + 0) = 0.
    for pos in (0, 1):
        zero_mask = torch.zeros(d_mlp, n_vocab)
        zero_mask.scatter_reduce_(
            1,
            inputs[:, pos].unsqueeze(0).expand(d_mlp, -1),
            remaining.float(),
            reduce="amax",
        )
        up[:, pos * n_vocab : (pos + 1) * n_vocab][zero_mask.bool()] = 0

    return up


def build_up_matrix_loop(
    shape: ModelShape,
    facts: dict,
    conn: np.ndarray,
    top_fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Readable per-neuron reference for `build_up_matrix` (see module docstring)."""
    inputs, labels = facts["inputs"], facts["targets"]
    n_vocab = shape.input_vocab_size
    conn_t = torch.tensor(conn, dtype=torch.bool)

    up = torch.ones(shape.d_mlp, 2 * n_vocab)

    for neuron in range(shape.d_mlp):
        guarded = inputs[conn_t[neuron][labels]]  # facts this neuron must not fire on
        if guarded.shape[0] == 0:
            continue

        tops = []
        for pos in (0, 1):
            uniq, counts = torch.unique(guarded[:, pos], return_counts=True)
            score = counts.float()
            if generator is not None:
                score = score + torch.rand(score.shape, generator=generator) * 0.5
            k = max(1, int(len(uniq) * top_fraction))
            tops.append(uniq[torch.argsort(score, descending=True, stable=True)[:k]])

        up[neuron, tops[0]] = -1
        up[neuron, tops[1] + n_vocab] = -1

        covered = (guarded[:, :1] == tops[0]).any(1) | (guarded[:, 1:] == tops[1]).any(1)
        remaining = guarded[~covered]
        if remaining.shape[0] > 0:
            up[neuron, remaining[:, 0]] = 0
            up[neuron, remaining[:, 1] + n_vocab] = 0

    return up


def build_down_matrix(conn: np.ndarray) -> torch.Tensor:
    """Unembedding: -2 from each of label l's neurons into logit l, 0 elsewhere.

    logit[l] = -2 * (# firing neurons assigned to l), so logit[l] = 0 exactly
    when none of them fire -- which the up matrix arranges precisely for the
    facts labeled l. Every other label scores <= -2, so the argmax is l.
    """
    return -2.0 * torch.tensor(conn, dtype=torch.float32).T


def hand_coded_weights(
    shape: ModelShape,
    facts: dict,
    params: HandCodedParams,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    conn = get_connection_matrix(
        D=shape.d_mlp, T=shape.output_vocab_size, S=params.S, seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    up = build_up_matrix(shape, facts, conn, params.top_fraction, generator)
    return up, build_down_matrix(conn)
