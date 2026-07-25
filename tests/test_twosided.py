"""Checks for the two-sided value code in handcode/twosided.py."""

import torch

from handcode.data import generate_facts
from handcode.linsolve import best_linsolve_accuracy
from handcode.model import ModelShape, accuracy, hidden_activations
from handcode.twosided import (
    TokenBlocks,
    TwoSidedParams,
    _capped_step,
    _decode_accuracy,
    assemble,
    init_embeddings,
    pad_group,
    solve_two_sided,
    two_sided_weights,
)

PARAMS = TwoSidedParams(rho=0.53, mu=1e-2, rounds=150, sweeps=4)


def _setup(d, n_facts, params=PARAMS, seed=1000):
    shape = ModelShape.from_d(d)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    solution = solve_two_sided(shape, facts, params, seed)
    up, down = assemble(shape, solution, params.delta, float(d))
    return shape, facts, solution, up, down


def test_pad_group_buckets_every_fact_exactly_once():
    keys = torch.tensor([3, 1, 3, 0, 1, 3])
    group = pad_group(keys, 5)
    assert group.index.shape == (5, 3)  # key 3 appears three times
    for key in range(5):
        got = sorted(group.index[key][group.valid[key]].tolist())
        assert got == sorted((keys == key).nonzero().flatten().tolist())


def test_init_embeddings_hit_the_requested_density():
    for rho in (0.2, 0.53, 0.8):
        u, v = init_embeddings(256, 64, rho, torch.Generator().manual_seed(0))
        # every (a, b) pair, not just the fact set, so this is the population value
        density = float(((u.unsqueeze(1) + v.unsqueeze(0)) > 0).double().mean())
        assert abs(density - rho) < 0.02, f"rho={rho} got {density:.3f}"


def test_capped_step_respects_the_flip_cap():
    generator = torch.Generator().manual_seed(0)
    pre = torch.randn(500, 40, generator=generator, dtype=torch.float64)
    step_pre = torch.randn(500, 40, generator=generator, dtype=torch.float64) * 3
    for cap in (0.01, 0.05, 0.2):
        step, crossings = _capped_step(pre, step_pre, cap)
        flipped = ((pre + step * step_pre) > 0) != (pre > 0)
        assert float(flipped.double().mean()) <= cap + 1e-9
        assert 0.0 < step <= 1.0
        assert crossings >= int(flipped.sum())


def test_capped_step_is_full_when_nothing_crosses():
    """No sign change anywhere means the frozen pattern survives the whole step.

    Reporting zero crossings is what tells the solve it has reached its fixed
    point, so it has to be exact, not merely below the cap.
    """
    pre = torch.full((100, 10), 5.0, dtype=torch.float64)
    step_pre = torch.full((100, 10), 0.5, dtype=torch.float64)
    assert _capped_step(pre, step_pre, 0.02) == (1.0, 0)


def test_token_blocks_solve_matches_a_direct_least_squares():
    """Each block is a ridge regression; check one token against lstsq."""
    generator = torch.Generator().manual_seed(0)
    n_facts, n_keys, n_value = 300, 8, 12
    keys = torch.randint(0, n_keys, (n_facts,), generator=generator)
    pattern = (torch.rand(n_facts, n_value, generator=generator) > 0.5).double()
    residual = torch.randn(n_facts, generator=generator, dtype=torch.float64)
    group = pad_group(keys, n_keys)
    live = torch.ones(n_facts, dtype=torch.bool)

    got = TokenBlocks(group, pattern, live, 1e-6, n_keys, n_value).solve(residual)
    for key in range(n_keys):
        rows = (keys == key).nonzero().flatten()
        design, rhs = pattern[rows], residual[rows]
        gram = design.T @ design + 1e-6 * torch.eye(n_value, dtype=torch.float64)
        wanted = torch.linalg.solve(gram, design.T @ rhs)
        assert torch.allclose(got[key], wanted, atol=1e-6)


def test_relu_is_the_only_gate():
    """No mask matrices: every weight is a value, so nothing is a sentinel.

    `linsolve` fills its first embedding with `-BIG` to force neurons silent.
    This construction has no such entries -- the sign of `u[a] + v[b]` is the
    gate -- which is what frees the first embedding to carry facts.
    """
    shape, _facts, _sol, up, _down = _setup(16, 400)
    values = up[: shape.d_mlp - 1]
    assert values.min() > -20 * values.abs().mean()


def test_decoded_sum_lands_on_the_label():
    """The core invariant: on a fact it gets right, the active neurons sum to
    shift + l + 1, and every such fact is one the model gets right."""
    shape, facts, solution, up, down = _setup(32, 2000)
    hidden = hidden_activations(up, facts["inputs"])
    decoded = hidden[:, : shape.d_mlp - 1].sum(1) - solution.shift
    wanted = (facts["targets"] + 1).float()
    within = (decoded - wanted).abs() < 0.5
    assert within.any()
    correct = (hidden @ down.T).argmax(-1) == facts["targets"]
    assert bool((within <= correct).all())  # landing within 1/2 implies correct


def test_analytic_accuracy_matches_the_assembled_model():
    """`_decode_accuracy` must agree with an argmax over real logits."""
    for d, n_facts in ((16, 400), (32, 2000)):
        shape, facts, solution, up, down = _setup(d, n_facts)
        assert abs(accuracy(up, down, facts) - solution.accuracy) < 1e-6


def test_readout_implements_the_quadratic_decode():
    """logit_c = -(c - l)^2 / 2 up to a per-fact constant, plus c times the miss.

    Writing `e` for how far the decoded sum lands from its target, the readout
    gives `logit_c - (-(c-l)^2/2) = (c+1) e + const`, so the departure from the
    ideal quadratic is exactly `c * e` -- zero on a fact the construction
    solves, and growing linearly in the class index otherwise. Checking the
    identity rather than a bound tests the readout independently of how well
    the solve went.
    """
    shape, facts, solution, up, down = _setup(32, 2000)
    hidden = hidden_activations(up, facts["inputs"])
    decoded = hidden[:, : shape.d_mlp - 1].sum(1) - solution.shift
    error = decoded - (facts["targets"] + 1).float()

    logits = hidden @ down.T
    labels = torch.arange(shape.output_vocab_size).float()
    ideal = -0.5 * (labels.unsqueeze(0) - facts["targets"].unsqueeze(1).float()) ** 2
    residual = logits - ideal
    departure = residual - residual[:, :1]
    assert torch.allclose(
        departure, error.unsqueeze(1) * labels.unsqueeze(0), atol=0.02, rtol=1e-3
    )


def test_bias_neuron_is_always_on():
    shape, facts, _sol, up, _down = _setup(16, 400)
    hidden = hidden_activations(up, facts["inputs"])
    assert torch.all(hidden[:, shape.d_mlp - 1] == float(shape.d_mlp))


def test_is_deterministic():
    first = two_sided_weights(*_seeded(16, 400))
    second = two_sided_weights(*_seeded(16, 400))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def _seeded(d, n_facts):
    shape = ModelShape.from_d(d)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
    return shape, facts, PARAMS, 1000


def test_stores_more_than_linsolve_at_the_same_fact_count():
    """The point of the construction: 4d^2 free parameters against linsolve's 2d^2."""
    for d, n_facts in ((16, 640), (32, 3072)):
        shape = ModelShape.from_d(d)
        facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)
        theirs = max(best_linsolve_accuracy(shape, facts, k, 1000) for k in (4, 6, 8))
        _shape, _facts, solution, up, down = _setup(d, n_facts, PARAMS)
        ours = accuracy(up, down, facts)
        assert ours > theirs + 0.2, f"d={d}: ours {ours:.3f} vs linsolve {theirs:.3f}"


def test_first_embedding_carries_facts():
    """The claim the whole construction rests on, as an ablation.

    Freezing the first embedding halves the free parameters to `2 d^2` and the
    solve collapses, at every load -- so the first embedding is carrying facts,
    not decorating. This is *not* a fair stand-in for `linsolve` at the same
    budget: there the first embedding is a deliberately designed mask, here it
    is left at its random initialisation. The budget comparison against
    `linsolve` proper is `test_stores_more_than_linsolve_at_the_same_fact_count`.
    """
    d, n_facts = 32, 3072  # 3 d^2: above linsolve's ceiling, below 4 d^2
    shape = ModelShape.from_d(d)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)

    full = solve_two_sided(shape, facts, PARAMS, 1000).accuracy
    ablated = max(
        solve_two_sided(
            shape, facts,
            TwoSidedParams(rho=rho, mu=PARAMS.mu, rounds=PARAMS.rounds,
                           sweeps=PARAMS.sweeps, freeze_first=True),
            1000,
        ).accuracy
        for rho in (0.35, 0.53, 0.70)
    )
    assert full > 0.95, full
    assert ablated < 0.5 * full, f"ablated {ablated:.3f} vs full {full:.3f}"


def test_density_stays_in_the_trained_models_range():
    """The construction is meant to look like a trained model, not just score."""
    for d, n_facts in ((32, 2000), (64, 8000)):
        _shape, _facts, solution, _up, _down = _setup(d, n_facts)
        assert 0.25 < solution.density < 0.75, f"d={d} density {solution.density:.2f}"


def test_decode_accuracy_scores_every_candidate_shift():
    sums = torch.tensor([1.4, 2.6, 3.5], dtype=torch.float64)
    targets = torch.tensor([0, 1, 2])
    shifts = torch.tensor([0.0, 0.5], dtype=torch.float64)
    got = _decode_accuracy(sums, targets, 1.0, shifts, 4)
    # shift 0: round(1.4)-1=0 ok, round(2.6)-1=2 wrong, round(3.5)-1=3 wrong
    assert torch.allclose(got, torch.tensor([1 / 3, 1.0], dtype=torch.float64))
