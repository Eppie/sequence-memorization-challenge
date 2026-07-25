"""Checks for the value-code construction in handcode/linsolve.py."""

import torch

from handcode.data import generate_facts
from handcode.handcoded import HandCodedParams, hand_coded_weights
from handcode.linsolve import (
    LinsolveParams,
    best_linsolve_accuracy,
    linsolve_weights,
    selector_masks,
    solve_values,
)
from handcode.model import ModelShape, accuracy, hidden_activations

CASES = [(16, 200), (32, 1000), (64, 4000)]
PARAMS = LinsolveParams(k=4, mu=1e-6, t0_scale=20.0)


def _setup(d, n_facts, params=PARAMS, seed=1000):
    shape = ModelShape.from_d(d)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    up, down = linsolve_weights(shape, facts, params, seed)
    return shape, facts, up, down


def test_selector_masks_have_exactly_k_ones_per_token():
    mask = selector_masks(64, 31, 4, torch.Generator().manual_seed(0))
    assert mask.shape == (64, 31)
    assert torch.all(mask.sum(1) == 4)
    assert set(mask.unique().tolist()) == {0.0, 1.0}


def test_bias_neuron_is_always_on():
    """The quadratic decode needs a constant channel; u = v = 1 gives h = 2."""
    shape, facts, up, _down = _setup(*CASES[1])
    hidden = hidden_activations(up, facts["inputs"])
    assert torch.all(hidden[:, shape.d_mlp - 1] == 2.0)


def test_unselected_neurons_are_silent():
    """-BIG on unselected tokens must dominate any value the second token adds."""
    shape, facts, up, _down = _setup(*CASES[1])
    n_vocab, n_value = shape.input_vocab_size, shape.d_mlp - 1
    hidden = hidden_activations(up, facts["inputs"])
    selected = up[:n_value, :n_vocab].T[facts["inputs"][:, 0]] == 0.0  # (n_facts, n_value)
    assert torch.all(hidden[:, :n_value][~selected] == 0.0)


def test_values_are_non_negative():
    """A ReLU cannot report a negative value; the T0 offset exists to ensure this."""
    shape, facts, up, _down = _setup(*CASES[1])
    assert torch.all(up[: shape.d_mlp - 1, shape.input_vocab_size :] >= 0.0)


def test_decoded_sum_lands_on_the_label():
    """The construction's core invariant: the value neurons sum to T0 + l + 1."""
    shape, facts, up, _down = _setup(32, 1000)
    t0 = PARAMS.t0_scale * shape.output_vocab_size
    hidden = hidden_activations(up, facts["inputs"])
    decoded = hidden[:, : shape.d_mlp - 1].sum(1) - t0
    wanted = (facts["targets"] + 1).float()
    # Correct argmax needs the decode within +-1/2 of the target.
    assert (decoded - wanted).abs().max() < 0.5


def test_readout_implements_the_quadratic_decode():
    """logit_c should equal -(c - l)^2 / 2 up to a per-fact constant."""
    shape, facts, up, down = _setup(32, 1000)
    logits = hidden_activations(up, facts["inputs"]) @ down.T
    labels = torch.arange(shape.output_vocab_size).float()
    wanted = -0.5 * (labels.unsqueeze(0) - facts["targets"].unsqueeze(1).float()) ** 2
    residual = logits - wanted
    # Each row should differ from the ideal by a single constant offset.
    assert (residual - residual[:, :1]).abs().max() < 0.05 * shape.output_vocab_size


def test_beats_the_hand_coded_construction_by_a_wide_margin():
    for d, n_facts in CASES:
        shape = ModelShape.from_d(d)
        facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
        theirs = max(
            accuracy(*hand_coded_weights(shape, facts, HandCodedParams(S=s, top_fraction=0.1), 1000), facts)
            for s in (2, 4, 8)
        )
        ours = max(best_linsolve_accuracy(shape, facts, k, 1000) for k in (4, 8, 16))
        assert ours > theirs + 0.3, f"d={d}: ours {ours:.3f} vs theirs {theirs:.3f}"


def test_solve_is_exact_while_under_determined():
    """Each second token's system has ~N/n_vocab equations in d-1 unknowns."""
    shape = ModelShape.from_d(64)
    n_facts = 2000  # ~15 equations per second token, 63 unknowns
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    n_value, k, t0 = shape.d_mlp - 1, 4, 20.0 * shape.output_vocab_size
    mask = selector_masks(shape.input_vocab_size, n_value, k, torch.Generator().manual_seed(0))
    values = solve_values(facts, mask, shape.input_vocab_size, n_value, k, 1e-9, t0)

    design = mask[facts["inputs"][:, 0]]
    got = (design * values.T[facts["inputs"][:, 1]]).sum(1)
    wanted = (facts["targets"] + 1).double() + t0
    assert (got - wanted).abs().max() < 1e-3
