"""Binary search for the maximum number of facts a condition can store.

Four conditions, matching Figure 5 of the post:

    trained     both matrices learned by Adam from a random init
    hybrid      hand-coded embedding (frozen) + learned unembedding
    hand-coded  both matrices from the construction, no gradient descent
    rand-emb    frozen random embedding + learned unembedding (the control)

For each n_facts we ask a yes/no question -- "can this condition reach the
accuracy threshold?" -- and binary-search the largest yes. Following the post
we use the "any" aggregation: n_attempts models are built with different seeds
(and, for the constructed conditions, every (S, top_fraction) cell of the
hyperparameter sweep is tried), and the answer is yes if any of them passes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .data import generate_facts
from .handcoded import HandCodedParams, hand_coded_weights
from .model import ModelShape, accuracy, hidden_activations, random_init, train
from .readouts import RIDGE_ALPHAS, ridge_down, silence_base, tiebreak_down

# The four conditions of the post's Figure 5, plus two of our own that keep the
# authors' embedding and only replace the hand-coded unembedding.
CONDITIONS = (
    "trained", "hybrid", "hand-coded", "rand-emb",
    "hc-tiebreak", "hc-ridge", "coin-tiebreak", "coin-ridge", "linsolve",
    "twosided",
)
OUR_CONDITIONS = (
    "hc-tiebreak", "hc-ridge", "coin-tiebreak", "coin-ridge", "linsolve", "twosided",
)

# Conditions whose weights come from the construction, so the (S, top_fraction)
# sweep applies to them.
CONSTRUCTED = ("hand-coded", "hybrid", *OUR_CONDITIONS)

# Binary search stops when the bracket is within this fraction of its top end,
# i.e. ~2% relative resolution on max_facts (same as the authors' setting).
PRECISION_FRACTION = 0.02


@dataclass
class SweepGrid:
    """The (S, top_fraction) cells tried for the constructed conditions."""

    S_values: list[int]
    top_fractions: list[float]

    @classmethod
    def for_d(cls, d: int, condition: str) -> "SweepGrid":
        # The post finds best_S ~ sqrt(d) for hand-coded and a bit less for
        # hybrid; we sweep generously around that.
        s_max = max(4, int(2.5 * math.sqrt(d)))
        if condition == "twosided":
            # S indexes the initial activation density; every other knob (ridge
            # strength, drop schedule) is swept inside the scorer.
            from .twosided import RHO_VALUES

            return cls(S_values=list(range(1, len(RHO_VALUES) + 1)), top_fractions=[0.0])
        if condition == "linsolve":
            # S is reused as k, the number of selector neurons per first token;
            # the ridge strength and target offset are swept inside the scorer.
            ks = sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, d // 2, d - 1})
            return cls(S_values=[k for k in ks if 1 <= k <= d - 1], top_fractions=[0.0])
        if condition.startswith("coin"):
            # The coincidence construction has no top_fraction; its rectangles
            # come straight from the fact set, so S is the only knob.
            return cls(S_values=list(range(1, s_max + 1)), top_fractions=[0.0])
        top = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        if condition != "hand-coded":
            # The post reports the best top_fraction running larger once the
            # unembedding is free rather than tied to the silence code.
            top = top + [0.4, 0.5]
        return cls(S_values=list(range(1, s_max + 1)), top_fractions=top)

    def cells(self, prior: HandCodedParams | None = None) -> list[HandCodedParams]:
        """All cells, ordered so that cells near `prior` (the last winner) come
        first -- with "any" aggregation a single pass is enough, so trying the
        likely winners first makes the successful searches much cheaper."""
        cells = [
            HandCodedParams(S=s, top_fraction=tf)
            for s in self.S_values
            for tf in self.top_fractions
        ]
        if prior is not None:
            cells.sort(
                key=lambda c: (abs(c.S - prior.S), abs(c.top_fraction - prior.top_fraction))
            )
        return cells


@dataclass
class Attempt:
    """One (condition, n_facts) evaluation: did anything reach the threshold?"""

    passed: bool
    best_accuracy: float
    best_params: HandCodedParams | None = None


def _score_once(
    condition: str, shape: ModelShape, facts: dict, seed: int,
    params: HandCodedParams | None, threshold: float = 1.0,
) -> float:
    if condition == "hand-coded":
        up, down = hand_coded_weights(shape, facts, params, seed)
        return accuracy(up, down, facts)
    if condition in OUR_CONDITIONS:
        return _score_our_readout(condition, shape, facts, seed, params, threshold)
    if condition == "hybrid":
        up, _ = hand_coded_weights(shape, facts, params, seed)
        _, down = random_init(shape, seed)
        return train(up, down, facts, train_up=False)
    if condition == "rand-emb":
        up, down = random_init(shape, seed)
        return train(up, down, facts, train_up=False)
    if condition == "trained":
        up, down = random_init(shape, seed)
        return train(up, down, facts, train_up=True)
    raise ValueError(f"unknown condition {condition!r}")


def _score_our_readout(
    condition: str, shape: ModelShape, facts: dict, seed: int,
    params: HandCodedParams, threshold: float = 1.0,
) -> float:
    """Our constructions: an embedding plus a closed-form unembedding.

    `hc-*` keep the authors' silence embedding; `coin-*` use the active-pattern
    coincidence embedding. `*-tiebreak` keeps the construction's own discrete
    readout as primary and lets ridge adjudicate only where it ties.
    `twosided` replaces both matrices (see `twosided.py`).
    """
    from .coincidence import CoincidenceParams, coincidence_weights
    from .connection import get_connection_matrix
    from .linsolve import best_linsolve_accuracy
    from .twosided import best_two_sided_accuracy

    if condition == "linsolve":
        return best_linsolve_accuracy(shape, facts, params.S, seed)

    if condition == "twosided":
        return best_two_sided_accuracy(shape, facts, params.S, seed, threshold)

    if condition.startswith("coin"):
        coin = CoincidenceParams(S=params.S, shrink=1.0, grade=0.0)
        up, primary = coincidence_weights(shape, facts, coin, seed)
    else:
        up, _ = hand_coded_weights(shape, facts, params, seed)
        primary = silence_base(
            get_connection_matrix(
                D=shape.d_mlp, T=shape.output_vocab_size, S=params.S, seed=seed
            )
        )

    hidden = hidden_activations(up, facts["inputs"])
    targets = facts["targets"]
    use_primary = condition.endswith("tiebreak")

    best = 0.0
    for alpha in RIDGE_ALPHAS:
        down = ridge_down(hidden, targets, shape.output_vocab_size, alpha)
        if use_primary:
            down = tiebreak_down(primary, down, hidden)
        best = max(best, accuracy(up, down, facts))
        if best == 1.0:
            break
    return best


def can_store(
    condition: str,
    shape: ModelShape,
    n_facts: int,
    threshold: float,
    n_attempts: int,
    grid: SweepGrid | None = None,
    prior: HandCodedParams | None = None,
    fact_seed: int = 42,
) -> Attempt:
    """Can `condition` store `n_facts` facts at >= `threshold` accuracy?"""
    facts = generate_facts(n_facts, shape.input_vocab_size, shape.output_vocab_size, fact_seed)

    if condition in CONSTRUCTED:
        candidates = grid.cells(prior)
    else:
        candidates: list[HandCodedParams | None] = [None]

    best_acc, best_params = 0.0, None
    for params in candidates:
        for attempt in range(n_attempts):
            acc = _score_once(condition, shape, facts, 1000 + attempt, params, threshold)
            if acc > best_acc:
                best_acc, best_params = acc, params
            if acc >= threshold:  # "any" aggregation -- one success is enough
                return Attempt(True, best_acc, best_params)

    return Attempt(False, best_acc, best_params)


@dataclass
class SearchResult:
    condition: str
    d: int
    threshold: float
    max_facts: int
    best_params: HandCodedParams | None
    n_evaluations: int = 0
    seconds: float = 0.0
    trace: list[tuple[int, bool]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "condition": self.condition,
            "d": self.d,
            "threshold": self.threshold,
            "max_facts": self.max_facts,
            "best_S": self.best_params.S if self.best_params else None,
            "best_top_fraction": (
                self.best_params.top_fraction if self.best_params else None
            ),
            "n_evaluations": self.n_evaluations,
            "seconds": round(self.seconds, 1),
            "trace": self.trace,
        }


def find_max_facts(
    condition: str,
    d: int,
    threshold: float,
    n_attempts: int = 3,
    verbose: bool = True,
) -> SearchResult:
    """Largest storable fact count for one condition and d.

    Ramp up by doubling from a small fact count until the first failure, then
    binary-search the bracket to the authors' 2% relative precision. Starting
    from the top of the range instead (as a plain binary search does) would
    spend its first and most expensive evaluations on fact counts far above
    capacity -- for the trained conditions those are whole training runs on
    tens of thousands of facts, so the ramp is what makes large d affordable.
    """
    shape = ModelShape.from_d(d)
    grid = SweepGrid.for_d(d, condition)
    started = time.time()

    state = {"best": 0, "best_params": None, "prior": None, "n_evals": 0}
    trace: list[tuple[int, bool]] = []

    def probe(n_facts: int) -> bool:
        result = can_store(
            condition, shape, n_facts, threshold, n_attempts, grid, state["prior"]
        )
        state["n_evals"] += 1
        trace.append((n_facts, result.passed))
        if verbose:
            mark = "ok  " if result.passed else "fail"
            extra = f" [{result.best_params}]" if result.best_params else ""
            print(
                f"    {mark} n_facts={n_facts:<6d} acc={result.best_accuracy:.3f}"
                f"{extra}  ({time.time() - started:.0f}s)",
                flush=True,
            )
        if result.passed:
            state["best"] = max(state["best"], n_facts)
            state["best_params"] = result.best_params
            state["prior"] = result.best_params or state["prior"]
        return result.passed

    # Ramp: double until failure (or until the dataset ceiling is reached).
    lo, hi, n = 0, None, min(8 * d, shape.max_facts)
    while True:
        if probe(n):
            lo = n
            if n == shape.max_facts:
                break
            n = min(2 * n, shape.max_facts)
        else:
            hi = n
            break

    # Refine the (lo, hi) bracket to ~2% relative resolution.
    if hi is not None:
        lo = max(lo, 1)
        while hi - lo >= PRECISION_FRACTION * hi:
            mid = (lo + hi) // 2
            if mid <= lo or mid >= hi:
                break
            if probe(mid):
                lo = mid
            else:
                hi = mid

    return SearchResult(
        condition=condition,
        d=d,
        threshold=threshold,
        max_facts=state["best"],
        best_params=state["best_params"],
        n_evaluations=state["n_evals"],
        seconds=time.time() - started,
        trace=trace,
    )


def fit_scaling(ds: list[int], max_facts: list[int]) -> tuple[float, float]:
    """Least-squares fit of max_facts = a * d^b / ln(d); returns (a, b).

    Linear in log space: ln(max_facts * ln d) = ln a + b ln d.
    """
    x = torch.tensor([math.log(d) for d in ds], dtype=torch.float64)
    y = torch.tensor(
        [math.log(m * math.log(d)) for d, m in zip(ds, max_facts)], dtype=torch.float64
    )
    n = len(x)
    b = ((x * y).sum() - x.sum() * y.sum() / n) / ((x * x).sum() - x.sum() ** 2 / n)
    a = math.exp(float(y.mean() - b * x.mean()))
    return a, float(b)


def predict(a: float, b: float, d: int) -> float:
    return a * d**b / math.log(d)
