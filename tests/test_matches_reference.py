"""Differential test against the authors' own code in `reference/`.

Everything here compares this repo's implementation with `reference/hc2.py`
(fetched from LindaLinsefors/Memory-Toy-Models, the repo the post links to).
Skipped automatically if `reference/` is not present.
"""

import os
import statistics
import sys
import types

import pytest
import torch

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(REFERENCE_DIR, "hc2.py")),
    reason="reference/ not present",
)


@pytest.fixture(scope="module")
def reference():
    """Import the authors' hc2 module.

    Their `models.py` imports wandb (unused by the pieces we compare against)
    and calls `torch.set_default_device` on import, so we stub the former and
    restore the device afterwards.
    """
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    sys.path.insert(0, REFERENCE_DIR)
    try:
        import hc2  # noqa: PLC0415
        import models  # noqa: PLC0415

        yield hc2, models
    finally:
        torch.set_default_device("cpu")
        sys.path.remove(REFERENCE_DIR)


def test_fact_generation_is_identical(reference):
    """Our generate_facts must produce byte-identical facts to theirs."""
    from handcode.data import generate_facts

    _hc2, models = reference
    for n_facts, v_in, v_out, seed in [(100, 32, 16, 42), (777, 64, 32, 42), (300, 32, 16, 7)]:
        theirs = models.generate_facts(
            n_facts=n_facts, input_len=2, input_vocab_size=v_in,
            output_vocab_size=v_out, seed=seed,
        )
        ours = generate_facts(n_facts, v_in, v_out, seed)
        assert torch.equal(theirs["inputs"].cpu(), ours["inputs"])
        assert torch.equal(theirs["targets"].cpu(), ours["targets"])


def _build_both(reference, monkeypatch, d, n_facts, S, top_fraction):
    """Build their up matrix and ours from the same facts and connection matrix,
    with their tie-breaking shuffle pinned to the identity and our jitter off."""
    from handcode.connection import get_connection_matrix
    from handcode.data import generate_facts
    from handcode.handcoded import build_up_matrix
    from handcode.model import ModelShape

    hc2, _models = reference
    shape = ModelShape.from_d(d)
    conn = get_connection_matrix(D=shape.d_mlp, T=shape.output_vocab_size, S=S, seed=0)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)

    # Their constructor calls generate_facts, which uses randperm too, so hand
    # it the fact set directly rather than let the patch reshuffle the data.
    monkeypatch.setattr(hc2, "generate_facts", lambda **kwargs: facts)
    monkeypatch.setattr(
        torch, "randperm", lambda n, **kwargs: torch.arange(n, device=kwargs.get("device"))
    )
    settings = hc2.HandCodedModel2Settings(
        input_vocab_size=shape.input_vocab_size,
        output_vocab_size=shape.output_vocab_size,
        n_facts=n_facts,
        seed=42,
        d_ff=shape.d_mlp,
        n_neurons_per_label=S,
        use_top_n_or_top_fraction="top_fraction",
        top_fraction=top_fraction,
    )
    their_model = hc2.HandCodedModel2(settings, precomputed_conn=conn)
    monkeypatch.undo()

    ours = build_up_matrix(shape, facts, conn, top_fraction, generator=None)
    return shape, facts, conn, their_model, ours


@pytest.mark.parametrize("d,n_facts,S,top_fraction", [
    (16, 80, 3, 0.1),
    (16, 120, 4, 0.2),
])
def test_weights_are_identical_when_tie_breaking_cannot_matter(
    reference, monkeypatch, d, n_facts, S, top_fraction
):
    """Same connection matrix + same tie-breaking => same weights, exactly.

    Only holds where the frequency ranking is unambiguous. Where it is not,
    both implementations make an arbitrary choice among equally frequent
    tokens; see the count-multiset test below for the invariant that survives.
    """
    from handcode.handcoded import build_down_matrix

    _shape, _facts, conn, their_model, our_up = _build_both(
        reference, monkeypatch, d, n_facts, S, top_fraction
    )
    assert torch.equal(their_model.up_matrix.cpu(), our_up)
    assert torch.equal(their_model.down_matrix.cpu(), build_down_matrix(conn))


@pytest.mark.parametrize("d,n_facts,S,top_fraction", [
    (32, 250, 5, 0.1),
    (32, 400, 6, 0.25),
    (64, 800, 8, 0.15),
])
def test_suppressed_tokens_have_the_same_frequencies(
    reference, monkeypatch, d, n_facts, S, top_fraction
):
    """Where the ranking ties, both implementations must still suppress tokens
    of the same frequencies -- i.e. they differ only in which of several equally
    good tokens they picked, not in what the rule selects for.
    """
    shape, facts, conn, their_model, our_up = _build_both(
        reference, monkeypatch, d, n_facts, S, top_fraction
    )
    theirs = their_model.up_matrix.cpu()
    inputs, labels = facts["inputs"], facts["targets"]
    conn_t = torch.tensor(conn, dtype=torch.bool)
    n_vocab = shape.input_vocab_size

    for neuron in range(shape.d_mlp):
        guarded = inputs[conn_t[neuron][labels]]
        if guarded.shape[0] == 0:
            continue
        for pos in (0, 1):
            uniq, counts = torch.unique(guarded[:, pos], return_counts=True)
            freq = dict(zip(uniq.tolist(), counts.tolist()))
            window = slice(pos * n_vocab, (pos + 1) * n_vocab)
            their_picks = (theirs[neuron, window] == -1).nonzero().flatten().tolist()
            our_picks = (our_up[neuron, window] == -1).nonzero().flatten().tolist()
            assert sorted(freq[t] for t in their_picks) == sorted(freq[t] for t in our_picks), (
                f"neuron {neuron} pos {pos}: suppressed different frequencies"
            )


@pytest.mark.parametrize("d,n_facts,S,top_fraction", [
    (16, 90, 4, 0.1),
    (32, 260, 5, 0.15),
    (64, 800, 8, 0.15),
])
def test_accuracy_matches_under_random_tie_breaking(reference, d, n_facts, S, top_fraction):
    """With tie-breaking left random (as in the real runs), the two
    implementations must draw accuracies from the same distribution."""
    from handcode.connection import get_connection_matrix
    from handcode.data import generate_facts
    from handcode.handcoded import build_down_matrix, build_up_matrix
    from handcode.model import ModelShape, accuracy

    hc2, _models = reference
    shape = ModelShape.from_d(d)
    conn = get_connection_matrix(D=shape.d_mlp, T=shape.output_vocab_size, S=S, seed=0)
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, 42)

    settings = hc2.HandCodedModel2Settings(
        input_vocab_size=shape.input_vocab_size,
        output_vocab_size=shape.output_vocab_size,
        n_facts=n_facts,
        seed=42,
        d_ff=shape.d_mlp,
        n_neurons_per_label=S,
        use_top_n_or_top_fraction="top_fraction",
        top_fraction=top_fraction,
    )

    n_seeds = 40
    down = build_down_matrix(conn)
    theirs, ours = [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        theirs.append(hc2.HandCodedModel2(settings, precomputed_conn=conn).evaluate()[0])
        up = build_up_matrix(
            shape, facts, conn, top_fraction, torch.Generator().manual_seed(seed)
        )
        ours.append(accuracy(up, down, facts))

    # Compare the means, which concentrate; the max over seeds is far too
    # high-variance a statistic to assert equality on.
    mean_gap = abs(statistics.mean(theirs) - statistics.mean(ours))
    pooled_se = (
        (statistics.variance(theirs) + statistics.variance(ours)) / n_seeds
    ) ** 0.5
    assert mean_gap <= 3 * pooled_se, (
        f"mean accuracy differs by {mean_gap:.4f} ({mean_gap / pooled_se:.1f} SE): "
        f"theirs {statistics.mean(theirs):.4f}, ours {statistics.mean(ours):.4f}"
    )
