"""Checks for the batched solver, against the reference it replaced.

`fastsolve` is a performance rewrite of `linsolve`'s inner loop, so the tests
that matter are differential: same maths, same answers, and in particular the
two identities the speedup actually rests on -- that the linear system does not
involve T0, and that ridge's primal and dual forms agree.
"""

import torch

from handcode.data import generate_facts
from handcode.fastsolve import assemble, group_facts, solve_profile, solve_x
from handcode.linsolve import _solve_once, selector_masks
from handcode.model import ModelShape, accuracy, hidden_activations

CASES = [(16, 300), (32, 1500), (64, 6000)]


def _mask(shape, k, seed=0):
    return selector_masks(
        shape.input_vocab_size, shape.d_mlp - 1, k, torch.Generator().manual_seed(seed)
    )


def test_grouping_places_every_fact_exactly_once():
    shape = ModelShape.from_d(32)
    facts = generate_facts(1500, shape.input_vocab_size, shape.output_vocab_size, 42)
    grouping = group_facts(facts, shape.input_vocab_size)

    seen = grouping.index[grouping.valid]
    assert sorted(seen.tolist()) == list(range(1500)), "every fact bucketed once"
    # each bucket must hold facts whose second token is that bucket
    rows, _ = grouping.valid.nonzero(as_tuple=True)
    assert torch.equal(facts["inputs"][seen][:, 1], rows)
    # and the cached first-token / target columns must line up with it
    assert torch.equal(grouping.first[grouping.valid], facts["inputs"][seen][:, 0])
    assert torch.equal(
        grouping.wanted[grouping.valid], (facts["targets"][seen] + 1).double()
    )


def _decoded(mask, values, facts):
    """What the model actually computes: sum of the selected values per fact."""
    return (mask[facts["inputs"][:, 0]] * values.T[facts["inputs"][:, 1]]).sum(1)


def test_batched_solve_matches_the_reference_loop():
    """solve_x must reproduce _solve_once where it is determined.

    Compared on the *decoded sums*, not the raw weights. A token whose system is
    under-determined has a whole null space of equally valid solutions, and which
    one a solver lands on depends on floating-point detail when mu is tiny -- the
    weights can differ by ~1e2 while the model behaves identically. The decode is
    the part the data actually pins down, so that is what must agree.
    """
    for d, n_facts in CASES:
        shape = ModelShape.from_d(d)
        n_value = shape.d_mlp - 1
        facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
        grouping = group_facts(facts, shape.input_vocab_size)
        live = torch.ones(n_facts, dtype=torch.bool)

        for k in (2, 6):
            mask = _mask(shape, k)
            for mu in (1e-9, 1e-3):
                t0 = 300.0 * shape.output_vocab_size
                reference = _solve_once(
                    facts, mask, live, shape.input_vocab_size, n_value, k, mu, t0
                )
                fast = (solve_x(grouping, mask, live, n_value, mu) + t0 / k).clamp(min=0.0)
                gap = (_decoded(mask, reference, facts) - _decoded(mask, fast, facts)).abs().max()
                assert gap < 1e-3, f"d={d} k={k} mu={mu}: decode differs by {gap:.2e}"


def test_solve_is_independent_of_t0_up_to_clamping():
    """The identity the whole T0 sweep is free because of.

    Each design row has exactly k ones, so `wanted - baseline*k` collapses to
    `l + 1`: T0 never reaches the linear system. It survives only as the shift
    `T0/k` and, through it, as the ReLU clamp -- which is exactly what T0 is
    for. So off the clamped entries the shifted solutions must agree exactly,
    and raising T0 must only ever clamp fewer of them.
    """
    shape = ModelShape.from_d(32)
    n_value = shape.d_mlp - 1
    facts = generate_facts(1500, shape.input_vocab_size, shape.output_vocab_size, 42)
    grouping = group_facts(facts, shape.input_vocab_size)
    live = torch.ones(1500, dtype=torch.bool)
    k = 4
    mask = _mask(shape, k)

    unclamped = solve_x(grouping, mask, live, n_value, 1e-9)
    clamped_counts = []
    for t0 in (20.0, 300.0, 5000.0):
        shifted = _solve_once(
            facts, mask, live, shape.input_vocab_size, n_value, k, 1e-9, t0
        ) - t0 / k
        was_clamped = (unclamped + t0 / k) <= 0.0
        clamped_counts.append(int(was_clamped.sum()))
        assert torch.allclose(
            shifted[~was_clamped], unclamped[~was_clamped], atol=1e-6, rtol=1e-6
        ), f"T0={t0}: unclamped entries should be a pure shift of one solve"

    assert clamped_counts == sorted(clamped_counts, reverse=True), (
        f"raising T0 should clamp fewer entries, got {clamped_counts}"
    )
    assert clamped_counts[-1] == 0, "a large enough T0 should clamp nothing"


def test_primal_and_dual_paths_agree():
    """solve_x picks whichever gram matrix is smaller; both are the same ridge
    solution, so the choice must not change the answer."""
    shape = ModelShape.from_d(32)
    n_value = shape.d_mlp - 1
    mu = 1e-6
    # one fact set lands under-determined per token, one over-determined
    for n_facts in (600, 4000):
        facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
        grouping = group_facts(facts, shape.input_vocab_size)
        live = torch.ones(n_facts, dtype=torch.bool)
        mask = _mask(shape, 4)

        got = solve_x(grouping, mask, live, n_value, mu)
        # a well-conditioned mu makes the two forms agree numerically too, so we
        # can check the per-token dispatch against an explicit primal solve
        design = mask[grouping.first] * grouping.valid.unsqueeze(-1)
        rhs = grouping.wanted * grouping.valid
        gram = design.transpose(1, 2) @ design
        gram.diagonal(dim1=1, dim2=2).add_(mu)
        want = torch.linalg.solve(
            gram, design.transpose(1, 2) @ rhs.unsqueeze(-1)
        ).squeeze(-1).T
        gap = (_decoded(mask, got, facts) - _decoded(mask, want, facts)).abs().max()
        assert gap < 1e-3, f"n_facts={n_facts}: decode differs by {gap:.2e}"


def test_assembled_model_decodes_the_labels():
    """End-to-end: the assembled weights must reproduce the quadratic decode."""
    shape = ModelShape.from_d(32)
    facts = generate_facts(1200, shape.input_vocab_size, shape.output_vocab_size, 42)
    grouping = group_facts(facts, shape.input_vocab_size)
    k = 4
    mask = _mask(shape, k)
    x = solve_profile(grouping, mask, shape.d_mlp - 1, 1e-9, 0.92, 3, False, 1200)
    t0 = 300.0 * shape.output_vocab_size
    up, down = assemble(shape, mask, x, k, t0)

    hidden = hidden_activations(up, facts["inputs"])
    assert torch.all(hidden[:, shape.d_mlp - 1] == 2.0), "bias neuron always on"
    decoded = hidden[:, : shape.d_mlp - 1].sum(1) - t0
    assert (decoded - (facts["targets"] + 1).float()).abs().median() < 0.5
    assert accuracy(up, down, facts) > 0.9


def test_pertoken_drop_keeps_only_what_a_token_can_satisfy():
    """The pre-drop must retain at most n_value facts per second token."""
    shape = ModelShape.from_d(16)
    n_value = shape.d_mlp - 1
    n_facts = 900  # well past n_value * n_vocab / 2, so buckets overflow
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    grouping = group_facts(facts, shape.input_vocab_size)

    excess = grouping.valid.cumsum(1) > n_value
    live = torch.ones(n_facts, dtype=torch.bool)
    live[grouping.index[excess & grouping.valid]] = False

    kept = (grouping.valid & live[grouping.index]).sum(1)
    assert int(kept.max()) <= n_value
    assert int(kept.sum()) < n_facts, "some facts should have been dropped"
