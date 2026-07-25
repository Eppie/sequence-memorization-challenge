"""Checks for the two closed-form unembeddings in handcode/readouts.py."""

import torch

from handcode.capacity import _score_once
from handcode.connection import get_connection_matrix
from handcode.data import generate_facts
from handcode.handcoded import HandCodedParams, build_down_matrix, hand_coded_weights
from handcode.model import ModelShape, accuracy, hidden_activations
from handcode.readouts import (
    RIDGE_ALPHAS,
    label_means,
    ridge_down,
    silence_base,
    tiebreak_down,
)

CASES = [(16, 90, 4, 0.1), (32, 260, 5, 0.15), (64, 800, 8, 0.15)]

# Fact counts near each d's acc>=0.9 capacity, where the silence code leaves the
# most facts undecided and a tie-breaker has the most to recover.
NEAR_CAPACITY = [(32, 244, 5, 0.1), (64, 768, 8, 0.1), (128, 2592, 11, 0.1)]


def _setup(d, n_facts, S, top_fraction, seed=1000):  # noqa: PLR0913
    shape = ModelShape.from_d(d)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    params = HandCodedParams(S=S, top_fraction=top_fraction)
    up, down = hand_coded_weights(shape, facts, params, seed)
    conn = get_connection_matrix(D=shape.d_mlp, T=shape.output_vocab_size, S=S, seed=seed)
    return shape, facts, up, down, conn


def test_label_means_are_unit_norm():
    shape, facts, up, _down, _conn = _setup(*CASES[0])
    hidden = hidden_activations(up, facts["inputs"])
    mu = label_means(hidden, facts["targets"], shape.output_vocab_size)
    assert mu.shape == (shape.output_vocab_size, shape.d_mlp)
    assert torch.allclose(mu.norm(dim=1), torch.ones(shape.output_vocab_size), atol=1e-5)


def test_hidden_activations_are_ternary_sums():
    """The tie-breaker's eps bound relies on h_i in {0,1,2} and hence on the
    silence code's logits being even integers <= 0."""
    shape, facts, up, down, _conn = _setup(*CASES[1])
    hidden = hidden_activations(up, facts["inputs"])
    assert set(hidden.unique().tolist()) <= {0.0, 1.0, 2.0}
    logits = hidden @ down.T
    assert torch.all(logits <= 0)
    assert torch.all(logits % 2 == 0)


def test_tiebreak_never_overturns_a_decided_case():
    """The correction may only adjudicate ties: wherever the silence code
    strictly preferred one label, the perturbed readout must agree."""
    for case in CASES + NEAR_CAPACITY:
        shape, facts, up, down, conn = _setup(*case)
        hidden = hidden_activations(up, facts["inputs"])
        base_logits = hidden @ down.T
        correction = ridge_down(hidden, facts["targets"], shape.output_vocab_size, 1e-2)
        new_logits = hidden @ tiebreak_down(silence_base(conn), correction, hidden).T

        best = base_logits.max(dim=1, keepdim=True).values
        decided = (base_logits == best).sum(dim=1) == 1  # a unique argmax
        assert torch.equal(
            base_logits[decided].argmax(-1), new_logits[decided].argmax(-1)
        ), f"tie-breaker changed a decided prediction at d={case[0]}"


def test_hebbian_correction_would_carry_no_signal():
    """Why the correction has to be ridge and not a prototype rule.

    On a tied fact, h is already zero on every tied label's neurons, so
    <h, mu_c> loses the term that could distinguish them and the prototype score
    is the same for all tied labels up to normalisation noise. The residual
    spread is not zero, but it is uninformative: the correct label wins such
    ties at chance rate. That is the property worth pinning, since it is the
    reason the correction has to be whitened.
    """
    wins, expected, total = 0, 0.0, 0
    for case in NEAR_CAPACITY:
        for seed in (1000, 1001, 1002):
            shape, facts, up, down, _conn = _setup(*case, seed=seed)
            hidden = hidden_activations(up, facts["inputs"])
            mu = label_means(hidden, facts["targets"], shape.output_vocab_size)
            logits = hidden @ down.T
            scores = hidden @ mu.T

            for fact in ((logits == 0).sum(1) > 1).nonzero().flatten().tolist():
                tied = (logits[fact] == 0).nonzero().flatten()
                pick = tied[scores[fact, tied].argmax()]
                wins += int(pick.item() == facts["targets"][fact].item())
                expected += 1.0 / len(tied)  # chance rate for this tie
                total += 1

    assert total >= 30, f"only {total} ties sampled; need more to judge"
    assert wins <= 1.5 * expected, (
        f"prototype rule resolved {wins}/{total} ties, chance is {expected:.1f} -- "
        "it carries more signal than the design assumes"
    )


def test_tiebreak_recovers_facts_near_capacity():
    """Where the silence code leaves facts undecided, the tie-breaker should
    convert a clear fraction of them."""
    gains = []
    for d, n_facts, S, top_fraction in NEAR_CAPACITY:
        shape, facts, up, down, _conn = _setup(d, n_facts, S, top_fraction)
        params = HandCodedParams(S=S, top_fraction=top_fraction)
        base = accuracy(up, down, facts)
        got = _score_once("hc-tiebreak", shape, facts, 1000, params)
        gains.append(got - base)
        assert got > base, f"hc-tiebreak at d={d}: {got:.4f} <= hand-coded {base:.4f}"
    assert sum(gains) / len(gains) > 0.02


def test_ridge_solution_satisfies_its_normal_equations():
    shape, facts, up, _down, _conn = _setup(*CASES[0])
    hidden = hidden_activations(up, facts["inputs"]).double()
    targets = facts["targets"]
    n_labels = shape.output_vocab_size
    alpha = RIDGE_ALPHAS[2]

    w = ridge_down(hidden.float(), targets, n_labels, alpha).double()
    onehot = torch.zeros(len(targets), n_labels, dtype=torch.float64)
    onehot[torch.arange(len(targets)), targets] = 1.0

    gram = hidden.T @ hidden
    lam = alpha * gram.diagonal().mean()
    residual = (gram + lam * torch.eye(shape.d_mlp, dtype=torch.float64)) @ w.T - hidden.T @ onehot
    assert residual.abs().max() < 1e-3 * gram.abs().max()
