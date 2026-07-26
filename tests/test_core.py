"""Correctness checks for the pieces the reproduction depends on."""

import math

import torch

from handcode.connection import evaluate_connection_matrix, make_connection_matrix
from handcode.data import generate_facts, one_hot_pair
from handcode.handcoded import (
    HandCodedParams,
    build_up_matrix,
    build_up_matrix_loop,
    hand_coded_weights,
)
from handcode.model import ModelShape, accuracy, forward, hidden_activations


def test_facts_are_unique_and_balanced():
    facts = generate_facts(n_facts=500, input_vocab_size=32, output_vocab_size=16)
    inputs, targets = facts["inputs"], facts["targets"]
    assert inputs.shape == (500, 2)
    pairs = {tuple(p.tolist()) for p in inputs}
    assert len(pairs) == 500, "input pairs must be distinct"
    counts = torch.bincount(targets, minlength=16)
    assert counts.max() - counts.min() <= 1, "labels dealt as evenly as possible"


def test_facts_are_seed_reproducible():
    a = generate_facts(100, 32, 16, seed=42)
    b = generate_facts(100, 32, 16, seed=42)
    assert torch.equal(a["inputs"], b["inputs"])
    assert not torch.equal(a["inputs"], generate_facts(100, 32, 16, seed=43)["inputs"])


def test_gather_forward_matches_one_hot_matmul():
    """The fast forward must be exactly the architecture in Figure 4."""
    shape = ModelShape.from_d(16)
    facts = generate_facts(200, shape.input_vocab_size, shape.output_vocab_size)
    up = torch.randn(shape.d_mlp, 2 * shape.input_vocab_size)
    x_enc = one_hot_pair(facts["inputs"], shape.input_vocab_size)
    assert torch.allclose(hidden_activations(up, facts["inputs"]), torch.relu(x_enc @ up.T))


def test_vectorized_up_matrix_matches_loop():
    """The vectorized construction is the per-neuron reference, batched."""
    shape = ModelShape.from_d(16)
    facts = generate_facts(150, shape.input_vocab_size, shape.output_vocab_size)
    conn = make_connection_matrix(D=shape.d_mlp, T=shape.output_vocab_size, S=4, seed=0)
    for top_fraction in (0.0, 0.1, 0.3):
        fast = build_up_matrix(shape, facts, conn, top_fraction, generator=None)
        slow = build_up_matrix_loop(shape, facts, conn, top_fraction, generator=None)
        assert torch.equal(fast, slow), f"mismatch at top_fraction={top_fraction}"


def test_up_matrix_values_are_ternary():
    """The post notes the construction only ever emits -1, 0, +1."""
    shape = ModelShape.from_d(16)
    facts = generate_facts(150, shape.input_vocab_size, shape.output_vocab_size)
    up, _ = hand_coded_weights(shape, facts, HandCodedParams(S=4, top_fraction=0.1), seed=0)
    assert set(up.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_guarded_neurons_are_silent():
    """The construction's core invariant: for every fact with label l, all of
    label l's neurons have pre-activation <= 0, so logit[l] == 0 exactly."""
    shape = ModelShape.from_d(16)
    facts = generate_facts(150, shape.input_vocab_size, shape.output_vocab_size)
    params = HandCodedParams(S=4, top_fraction=0.1)
    up, down = hand_coded_weights(shape, facts, params, seed=0)

    hidden = hidden_activations(up, facts["inputs"])
    logits = forward(up, down, facts["inputs"])
    correct_logit = logits.gather(1, facts["targets"].unsqueeze(1)).squeeze(1)

    assert torch.all(correct_logit == 0.0), "correct label must score exactly zero"
    assert torch.all(logits <= 0.0), "every other label scores <= -2 or 0"
    assert hidden.min() >= 0.0


def test_hand_coded_model_memorizes_small_fact_sets():
    shape = ModelShape.from_d(16)
    facts = generate_facts(60, shape.input_vocab_size, shape.output_vocab_size)
    up, down = hand_coded_weights(shape, facts, HandCodedParams(S=4, top_fraction=0.1), seed=0)
    assert accuracy(up, down, facts) >= 0.9


def test_connection_matrix_is_well_formed():
    D, T, S = 32, 32, 4
    m = make_connection_matrix(D=D, T=T, S=S, seed=0, n_restarts=2, sa_steps=5000)
    stats = evaluate_connection_matrix(m)
    assert stats["col_sum_min"] == stats["col_sum_max"] == S, "S neurons per label"
    assert stats["row_sum_max"] - stats["row_sum_min"] <= 2, "balanced neuron usage"
    # T <= D(D-1)/(S(S-1)) here, so overlap <= 1 is achievable
    assert T <= D * (D - 1) // (S * (S - 1))
    assert stats["max_pairwise_overlap"] <= 1


def test_scaling_fit_recovers_known_law():
    from handcode.capacity import fit_scaling

    ds = [16, 32, 64, 128]
    truth_a, truth_b = 5.66, 2.06
    facts = [round(truth_a * d**truth_b / math.log(d)) for d in ds]
    a, b = fit_scaling(ds, facts)
    assert abs(a - truth_a) < 0.05 and abs(b - truth_b) < 0.01
