"""Inequality-native declarative gates: design the pattern, solve for weights.

The rank argument (what-gd-builds item 2) kills equality codes only: m
equality channels force a rank-m readout and capacity <= 4d^2/m. The untried
move is designing the gate *pattern* forward from a combinatorial object with
distance properties, then handing it to the two-LP ascent -- weights solved
from the pattern, the readout free to spend inequality slack at full rank.

The design space is exact, not heuristic. A pattern is realizable iff each
neuron is an additive threshold 1[a[x0] + b[x1] > 0] (any 2x2 submatrix with
1s on one diagonal and 0s on the other is unrealizable: both diagonal sums
equal a_i + a_i' + b_j + b_j'). So this probe designs per-neuron *score
vectors* (a, b) -- every choice is realizable by construction -- and the open
question is purely whether any declarative score design has a feasible,
high-ceiling pattern, given that Gaussian random scores (init,
random_additive) are infeasible at this load.

Families, all deterministic given their seed:

  hadamard   token signatures from Sylvester Hadamard rows (pairwise distance
             exactly V/2); neurons are AND / NAND gates on signature bits.
             Fact signatures inherit code distance.

  modular    per-neuron affine bijections of the token index mod V, mapped to
             [0,1); fire iff frac_L + frac_R > 1. A universal-hash flavor:
             balanced density 1/2 per neuron, designed pairwise token
             separation, no near-tie structure.

  thermo     the inequality-native digit code. Tokens are base-8 digit pairs;
             neurons are thermometer comparisons (digit sums and differences
             against 8 spread thresholds). Distances between fact signatures
             are L1 distances between digit vectors -- a literal metric code,
             and precisely the construction the rank barrier does not touch.

  shuffle    control: the trained gate's own score columns, permuted across
             tokens per neuron. Matches every first-order per-neuron score
             statistic; destroys only the token co-adaptation. If design-space
             statistics were sufficient, this would be feasible.

    uv run python probe_designedgates.py
    uv run python probe_designedgates.py --families thermo shuffle
"""

import argparse
import json
import os

import numpy as np

import probe_gatequality as gq
from handcode.data import generate_facts
from handcode.model import ModelShape
from probe_ratedistortion import join_uv, pattern_of, split_uv

OUT_PATH = os.path.join(gq.RESULTS_DIR, "designedgates.json")


def merge_out(key: str, entry: dict) -> None:
    data = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            data = json.load(f)
    data.setdefault("cell", {"d": gq.D, "n_facts": gq.N_FACTS,
                             "fact_seed": gq.FACT_SEED})
    data.setdefault("entries", {})[key] = entry
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, OUT_PATH)


# --------------------------------------------------------------------------
# score-space constructions: each returns (a, b), both (V, d)
# --------------------------------------------------------------------------

def sylvester(n: int) -> np.ndarray:
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def build_hadamard(shape, inputs, mode: str):
    """Half the neurons AND gates on Hadamard signature bits, half NAND;
    "mixn2" swaps which half is which. The column roll for the right
    signature is the smallest in 1..8 leaving no fact with an empty active
    set (an empty fact is trivially unstorable). Only "mixn" has such a
    roll — pure AND (44 empty facts), AND/OR, and "mixn2" do not, which is
    why configs() runs "mixn" alone (FINDINGS §26)."""
    V, d = shape.input_vocab_size, shape.d_mlp
    H = sylvester(d)
    sign = np.where(np.arange(V) < d, 1.0, -1.0)[:, None]
    hL = sign * H[np.arange(V) % d]           # (V, d), +/-1 entries
    and_half = (np.arange(d) < d // 2) if mode == "mixn" \
        else (np.arange(d) >= d // 2)
    for roll in range(1, 9):
        hR = sign * np.roll(H, roll, axis=1)[np.arange(V) % d]
        a = np.where(and_half, hL - 0.5, -hL + 0.5)  # AND | NAND halves
        b = np.where(and_half, hR - 0.5, -hR + 0.5)
        if (pattern_of(a, b, inputs).sum(1) > 0).all():
            print(f"  [hadamard/{mode}] roll={roll}", flush=True)
            return a, b
    print(f"  [hadamard/{mode}] no roll in 1..8 avoids empty facts; "
          f"using roll=1", flush=True)
    return a, b


def build_modular(shape, seed: int):
    V, d = shape.input_vocab_size, shape.d_mlp
    rng = np.random.default_rng(seed)
    t = np.arange(V)

    def side():
        alpha = rng.choice(np.arange(1, V, 2), d)          # odd => bijection
        beta = rng.integers(0, V, d)
        return ((alpha * t[:, None] + beta) % V) / V - 0.5  # (V, d)

    return side(), side()


def build_thermo(shape, cross: bool):
    V, d = shape.input_vocab_size, shape.d_mlp
    t = np.arange(V)
    e0 = ((t % 8) - 3.5) / 3.5                 # digits centred to [-1, 1]
    e1 = ((t // 8) % 8 - 3.5) / 3.5
    if cross:
        forms = [(e0, -e1), (e1, -e0), (e0, e1), (e1, e0)]
    else:
        forms = [(e0, -e0), (e1, -e1), (e0, e0), (e1, e1)]
    n_th = d // len(forms)
    thetas = (np.arange(n_th) + 0.5) / n_th * 4 - 2        # spread in (-2, 2)
    a = np.zeros((V, d))
    b = np.zeros((V, d))
    for i, (gL, gR) in enumerate(forms):
        for k, th in enumerate(thetas):
            j = i * n_th + k
            a[:, j] = gL - th
            b[:, j] = gR
    return a, b


def build_shuffle(shape, facts, seed: int):
    _, up_t, _ = gq.get_gate("trained", shape, facts)
    u, v = split_uv(up_t, shape.input_vocab_size)
    rng = np.random.default_rng(seed)
    us, vs = u.copy(), v.copy()
    for j in range(u.shape[1]):
        us[:, j] = u[rng.permutation(len(u)), j]
        vs[:, j] = v[rng.permutation(len(v)), j]
    return us, vs


def configs(shape, facts):
    inputs = facts["inputs"].numpy()
    return {
        "hadamard": [("mixn", lambda: build_hadamard(shape, inputs, "mixn"))],
        "modular": [(f"s{s}", lambda s=s: build_modular(shape, s))
                    for s in (0, 1)],
        "thermo": [("axis", lambda: build_thermo(shape, False)),
                   ("cross", lambda: build_thermo(shape, True))],
        "shuffle": [(f"s{s}", lambda s=s: build_shuffle(shape, facts, s))
                    for s in (0, 1)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="*",
                        default=["hadamard", "modular", "thermo", "shuffle"])
    parser.add_argument("--out", default=None)
    gq.add_cell_args(parser)
    args = parser.parse_args()
    gq.configure(args.d, args.n_facts, args.fact_seed)
    if args.out:
        global OUT_PATH
        OUT_PATH = args.out

    shape = ModelShape.from_d(gq.D)
    facts = generate_facts(gq.N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, gq.FACT_SEED)
    inputs = facts["inputs"].numpy()
    print(f"[cell] d={gq.D} n_facts={gq.N_FACTS} fact_seed={gq.FACT_SEED}\n"
          f"[out] {OUT_PATH}", flush=True)

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            done = set(json.load(f).get("entries", {}))

    all_cfg = configs(shape, facts)
    for fam in args.families:
        for tag, build in all_cfg[fam]:
            key = f"{fam}/{tag}"
            if key in done:
                print(f"[skip] {key}", flush=True)
                continue
            a, b = build()
            pattern = pattern_of(a, b, inputs)
            active = pattern.sum(1)
            print(f"== {key}: density={pattern.mean():.3f} "
                  f"min_active={int(active.min())} "
                  f"empty_facts={int((active == 0).sum())}", flush=True)
            up0 = join_uv(a, b)
            entry = {
                "density": float(pattern.mean()),
                "min_active": int(active.min()),
                "empty_facts": int((active == 0).sum()),
                "gate_metrics": gq.gate_metrics(pattern, inputs),
                "predict": gq.predict_one(key, pattern, up0, None, facts,
                                          shape),
            }
            merge_out(key, entry)
            print(f"[saved] {key}", flush=True)


if __name__ == "__main__":
    main()
