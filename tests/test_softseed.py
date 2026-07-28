"""Checks for the soft ridge seed builder (handcode/softseed.py)."""

import numpy as np

from handcode.data import generate_facts
from handcode.model import ModelShape
from handcode.softseed import SoftSeedParams, softness_report, solve_soft_seed

SHAPE = ModelShape.from_d(8)
FACTS = generate_facts(40, SHAPE.input_vocab_size, SHAPE.output_vocab_size, 42)


def test_softness_report_is_sane():
    rng = np.random.default_rng(0)
    rep = softness_report(rng.standard_normal((100, 8)))
    assert rep["q005"] <= rep["median"]
    assert 0.0 < rep["density"] < 1.0
    assert np.isclose(rep["ratio"], rep["q005"] / rep["median"])


def test_seed_is_gauge_normalized_and_shaped():
    res = solve_soft_seed(SHAPE, FACTS, SoftSeedParams(rounds=10), seed=0)
    assert res.u.shape == (SHAPE.input_vocab_size, SHAPE.d_mlp)
    assert res.v.shape == (SHAPE.input_vocab_size, SHAPE.d_mlp)
    assert res.W.shape == (SHAPE.output_vocab_size, SHAPE.d_mlp)
    inputs = FACTS["inputs"].numpy()
    pre = res.u[inputs[:, 0]] + res.v[inputs[:, 1]]
    assert abs(np.median(np.abs(pre)) - 1.0) < 1e-9
    assert abs(res.softness["median"] - 1.0) < 1e-9


def test_fit_climbs_above_chance():
    res = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=30), seed=0
    )
    # 8 classes -> chance is 0.125; the ridge rounds must do real work.
    assert res.accuracy > 0.5
    # The reported accuracy is the actual relu model's, self-consistently.
    inputs, targets = FACTS["inputs"].numpy(), FACTS["targets"].numpy()
    h = np.maximum(res.u[inputs[:, 0]] + res.v[inputs[:, 1]], 0.0)
    assert res.accuracy == float(((h @ res.W.T).argmax(1) == targets).mean())


def test_heavy_step_ridge_freezes_the_geometry():
    light = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=15), seed=0
    )
    heavy = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e6, rounds=15), seed=0
    )
    # mu is the fit-vs-softness knob: heavy ridge must move the pattern less.
    assert heavy.history[-1]["net_drift"] < light.history[-1]["net_drift"]
    assert heavy.history[-1]["net_drift"] < 0.02


def test_acc_stop_halts_the_build():
    res = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=30, acc_stop=0.01),
        seed=0,
    )
    assert len(res.history) == 1


def test_soften_builds_the_reservoir_without_flips():
    base = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=15), seed=0
    )
    target = base.softness["ratio"] / 5.0
    soft = solve_soft_seed(
        SHAPE, FACTS,
        SoftSeedParams(mu=1e-2, rounds=15, soften_ratio=target),
        seed=0,
    )
    # The reservoir is constructed: the ratio lands near the request...
    assert soft.softness["ratio"] < base.softness["ratio"] / 3.0
    assert np.isclose(soft.softness["ratio"], target, rtol=0.5)
    # ...the compression never crosses zero (identical patterns)...
    inputs = FACTS["inputs"].numpy()
    pre_b = base.u[inputs[:, 0]] + base.v[inputs[:, 1]]
    pre_s = soft.u[inputs[:, 0]] + soft.v[inputs[:, 1]]
    assert ((pre_b > 0) == (pre_s > 0)).all()
    # ...and the fit is untouched to first order.
    assert abs(soft.accuracy - base.accuracy) <= 0.1


def test_soften_is_a_noop_when_already_soft_enough():
    base = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=15), seed=0
    )
    lax = solve_soft_seed(
        SHAPE, FACTS,
        SoftSeedParams(mu=1e-2, rounds=15,
                       soften_ratio=base.softness["ratio"] * 10.0),
        seed=0,
    )
    assert np.allclose(lax.u, base.u)
    assert np.allclose(lax.v, base.v)


def test_w_rms_rescale_is_argmax_invariant():
    base = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=15), seed=0
    )
    scaled = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-2, rounds=15, w_rms=2.0), seed=0
    )
    assert np.isclose(np.sqrt((scaled.W**2).mean()), 2.0)
    assert scaled.accuracy == base.accuracy
    assert np.allclose(scaled.u, base.u)


def test_flip_cap_is_respected_per_round():
    res = solve_soft_seed(
        SHAPE, FACTS, SoftSeedParams(mu=1e-4, rounds=8, flip_cap=0.02),
        seed=0,
    )
    n_bits = len(FACTS["targets"]) * SHAPE.d_mlp
    drifts = [row["net_drift"] for row in res.history]
    steps = np.diff([0.0] + drifts)
    # Net drift can grow by at most the cap each round (plus one bit of
    # order-statistic slack); flips that revert can make it smaller.
    assert (steps <= 0.02 + 1.5 / n_bits).all()
