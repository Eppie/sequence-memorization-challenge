"""Checks for the grouped digit code (handcode/digitcode.py)."""

import torch

from handcode.data import generate_facts
from handcode.digitcode import (
    DigitCodeParams,
    assemble,
    digit_codebook,
    make_groups,
    solve_digit_code,
)
from handcode.model import ModelShape, accuracy


def test_digit_codebook_is_a_base_p_expansion():
    digits, p = digit_codebook(32, 2)
    assert p == 6  # smallest p with p^2 >= 32
    labels = (digits[:, 0] + digits[:, 1] * p).long()
    assert torch.equal(labels, torch.arange(32))
    # m = 1 keeps one level per label.
    digits1, p1 = digit_codebook(32, 1)
    assert p1 == 32
    assert torch.equal(digits1[:, 0].long(), torch.arange(32))


def test_groups_partition_the_neurons():
    masks = make_groups(31, 5, torch.Generator().manual_seed(0))
    assert masks.shape == (5, 31)
    assert torch.equal(masks.sum(0), torch.ones(31))  # every neuron in one group
    sizes = masks.sum(1)
    assert float(sizes.max() - sizes.min()) <= 1.0  # near-equal split


def test_solves_and_assembles_consistently():
    shape = ModelShape.from_d(16)
    facts = generate_facts(350, shape.input_vocab_size, shape.output_vocab_size, 42)
    solution = solve_digit_code(
        shape, facts, DigitCodeParams(m=2, rounds=400), seed=1000
    )
    up, down = assemble(shape, solution)
    assert abs(accuracy(up, down, facts) - solution.accuracy) < 1e-9
    assert solution.accuracy > 0.9


def test_m1_matches_twosided_scale_capacity():
    """m = 1 is the twosided design; it should solve well past half the
    family ceiling at d=16 (twosided's own measured capacity is 696)."""
    shape = ModelShape.from_d(16)
    facts = generate_facts(560, shape.input_vocab_size, shape.output_vocab_size, 42)
    solution = solve_digit_code(
        shape, facts, DigitCodeParams(m=1, rounds=400), seed=1000
    )
    assert solution.accuracy == 1.0
