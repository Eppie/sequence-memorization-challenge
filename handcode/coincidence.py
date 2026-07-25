"""An active-pattern construction: coincidence detectors instead of suppressors.

Motivation is the authors' own observation (Figure 8): "our hand-coded algorithm
stores facts as patterns of inactive neurons, trained models seem to use patterns
of active neurons instead." This is the active-pattern dual of their construction.

The mechanism. Take binary embedding weights

    u_i[a] = +1 if a in A_i else -1        v_i[b] = +1 if b in B_i else -1

then the pre-activation u_i[a] + v_i[b] is +2 when both tokens are in their sets
and 0 or -2 otherwise, so

    h_i = relu(u_i[a] + v_i[b]) = 2 * 1[(a,b) in A_i x B_i]

-- an exact AND over a combinatorial rectangle. Two properties matter:

  * No collateral damage. In the authors' construction a -1 on one token
    silences that neuron across an entire row (or column) of the V x V pair
    grid, whether or not those cells had anything to do with the fact being
    encoded. Here a token's weight only matters in conjunction with the other
    position's, so a neuron's response is a *pair* feature rather than a
    thresholded sum of two independent ones.
  * The feature is genuinely non-additive. Pre-activations are always additive
    in the two tokens; the ReLU mask is the only place pair-specific information
    can enter, and a rectangle is the sharpest mask this architecture admits.

Choosing the rectangles. Neuron i serves a set of labels L_i (the same
connection-matrix design the authors use). Setting A_i to the first tokens and
B_i to the second tokens of every fact labelled in L_i makes neuron i fire on
*all* of those facts by construction -- no false negatives, a Bloom-filter-style
code where "all of label l's neurons fired" is evidence for l.

The cost is false positives: the rectangle has |A_i| * |B_i| cells but only
about |L_i| * N / n_labels of them are real facts, so it also fires on
cross-pairs (the first token of one fact with the second token of another). The
`shrink` hyperparameter trades this off -- keeping a random subset of each token
set shrinks the rectangle quadratically while losing coverage only linearly.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .connection import get_connection_matrix
from .model import ModelShape


@dataclass(frozen=True)
class CoincidenceParams:
    """Hyperparameters, mirroring the authors' (S, top_fraction) pair."""

    S: int  # neurons assigned to each label
    shrink: float = 1.0  # fraction of each token set kept (1.0 = full coverage)
    grade: float = 0.0  # amplitude of the within-rectangle graded channel

    def __str__(self) -> str:
        return f"S={self.S}, shrink={self.shrink:g}, grade={self.grade:g}"


def _token_sets(
    facts: dict,
    conn_t: torch.Tensor,
    n_labels: int,
    n_vocab: int,
    pos: int,
) -> torch.Tensor:
    """(d_mlp, n_vocab) boolean: does neuron i's label set use token t at `pos`?"""
    inputs, labels = facts["inputs"], facts["targets"]
    per_label = torch.zeros(n_labels, n_vocab)
    per_label[labels, inputs[:, pos]] = 1.0  # label l uses token t at this position
    return (conn_t @ per_label) > 0  # union over the labels each neuron serves


def _shrink_mask(
    mask: torch.Tensor, shrink: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Keep a random ceil(shrink * |row|) subset of each row's True entries.

    Shrinking a rectangle's sides by `shrink` cuts its area by shrink^2 but its
    fact coverage only by about shrink, which is the whole point of the knob.
    """
    if shrink >= 1.0:
        return mask
    scores = torch.rand(mask.shape, generator=generator) if generator is not None else (
        torch.arange(mask.shape[1]).float().expand(mask.shape[0], -1) / mask.shape[1]
    )
    scores = torch.where(mask, scores, torch.full_like(scores, 2.0))
    keep = torch.ceil(mask.sum(1) * shrink).long().clamp(min=1)
    rank = scores.argsort(dim=1, stable=True).argsort(dim=1)
    return mask & (rank < keep.unsqueeze(1))


def coincidence_up(
    shape: ModelShape,
    facts: dict,
    conn: np.ndarray,
    shrink: float = 1.0,
    grade: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Binary embedding making each neuron an AND over a rectangle of tokens.

    With `grade` = 0 the neuron is a pure indicator: h_i is 2 on the rectangle
    and 0 off it, i.e. exactly one bit per neuron per fact. The magnitude
    channel ReLU leaves available is then entirely unused.

    With `grade` = g in (0, 1) the in-set weights become 1 +- g and the out-set
    weights -(1 + g), which leaves the mask *exactly* where it was:

        in / in     (1+x) + (1+y) >= 2 - 2g > 0    -> fires, value 2 + x + y
        in / out    (1+x) - (1+g) =  x - g <= 0    -> silent
        out / out   -2(1+g)                 <  0   -> silent

    so on its rectangle the neuron now reports 2 + x_i[a] + y_i[b] rather than a
    constant. That graded part is additive in the two tokens, but it is *gated*
    by the rectangle, so it carries pair information the readout can use to tell
    apart facts that share a rectangle signature -- which a pure indicator code
    cannot do at all.
    """
    n_vocab = shape.input_vocab_size
    conn_t = torch.tensor(conn, dtype=torch.float32)  # (d_mlp, n_labels)

    up = torch.empty(shape.d_mlp, 2 * n_vocab)
    for pos in (0, 1):
        mask = _token_sets(facts, conn_t, shape.output_vocab_size, n_vocab, pos)
        mask = _shrink_mask(mask, shrink, generator)
        inside = torch.ones_like(up[:, :n_vocab])
        if grade > 0.0:
            jitter = torch.rand(mask.shape, generator=generator) * 2 - 1
            inside = inside + grade * jitter
        up[:, pos * n_vocab : (pos + 1) * n_vocab] = torch.where(
            mask, inside, torch.full_like(inside, -(1.0 + grade))
        )
    return up


def coincidence_down(conn: np.ndarray) -> torch.Tensor:
    """Hard readout: +1 from each of label l's neurons into logit l.

    logit_l = 2 * (number of label l's neurons that fired). At shrink=1 the
    correct label scores the maximum 2S by construction; a wrong label only ties
    it if every one of its own neurons fired by coincidence.
    """
    return torch.tensor(conn, dtype=torch.float32).T.clone()


def coincidence_weights(
    shape: ModelShape,
    facts: dict,
    params: CoincidenceParams,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    conn = get_connection_matrix(
        D=shape.d_mlp, T=shape.output_vocab_size, S=params.S, seed=seed
    )
    generator = torch.Generator().manual_seed(seed)
    up = coincidence_up(shape, facts, conn, params.shrink, params.grade, generator)
    return up, coincidence_down(conn)
