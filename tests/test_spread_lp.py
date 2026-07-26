"""Checks for the capped-sum (spread-pressure) mode of solve_max_margin."""

import numpy as np

from probe_maxmargin import solve_max_margin

D = 6
N_VOCAB = 12
N_LABELS = 6
N_FACTS = 20
BOX = 6.0
TAU = 0.4


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    inputs = np.stack(
        [rng.integers(0, N_VOCAB, N_FACTS), rng.integers(0, N_VOCAB, N_FACTS)],
        axis=1,
    )
    targets = rng.integers(0, N_LABELS, N_FACTS)
    pattern = rng.random((N_FACTS, D)) < 0.5
    pattern[:, 0] = True  # no empty active sets
    readout = rng.standard_normal((N_LABELS, D))
    return pattern, inputs, targets, readout


def _true_margins(pattern, inputs, targets, readout, u, v):
    h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
    logits = h @ readout.T
    n = len(targets)
    correct = logits[np.arange(n), targets]
    logits[np.arange(n), targets] = -np.inf
    return correct - logits.max(1)


def test_spread_respects_cap_and_underestimates_true_margins():
    pattern, inputs, targets, readout = _setup()
    u, v, obj, info = solve_max_margin(
        pattern, inputs, targets, readout, N_VOCAB, BOX,
        pattern_rows=False, spread_tau=TAU,
    )
    assert u is not None
    m = info["m"]
    assert m.shape == (N_FACTS,)
    assert (m <= TAU + 1e-6).all()
    # All wrong classes are constrained (wrong_sets=None), so each fact's
    # true masked-linear margin is at least its solved m_f.
    margins = _true_margins(pattern, inputs, targets, readout, u, v)
    assert (margins >= m - 1e-5).all()


def test_spread_objective_dominates_shared_gamma():
    pattern, inputs, targets, readout = _setup()
    _, _, gamma, _ = solve_max_margin(
        pattern, inputs, targets, readout, N_VOCAB, BOX, pattern_rows=False,
    )
    _, _, obj, _ = solve_max_margin(
        pattern, inputs, targets, readout, N_VOCAB, BOX,
        pattern_rows=False, spread_tau=TAU,
    )
    # The shared-gamma solution with every m_f = min(gamma, tau) is feasible
    # for the spread LP, so the spread optimum can only be larger.
    assert obj >= N_FACTS * min(gamma, TAU) - 1e-5


def test_default_path_reports_no_per_fact_margins():
    pattern, inputs, targets, readout = _setup()
    _, _, gamma, info = solve_max_margin(
        pattern, inputs, targets, readout, N_VOCAB, BOX, pattern_rows=False,
    )
    assert np.isfinite(gamma)
    assert "m" not in info
