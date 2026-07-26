"""Is the trained model the max-margin point of its own geometry?

Freeze a ReLU pattern and a readout. The remaining degrees of freedom are the
embeddings, and everything that matters is *linear* in them: each fact's
margin (through the frozen pattern and readout), and the pattern-consistency
conditions (active units keep nonnegative pre-activations, inactive ones
nonpositive -- at which point the frozen-pattern linearization IS the real
ReLU model). Maximizing the minimum margin subject to a weight box is
therefore a linear program, solved here exactly with HiGHS.

Two conditions:

  reconstruct    pattern and readout from a model trained to 100% at its own
                 capacity. The trained embeddings are a feasible point, so
                 gamma* >= the trained min margin by construction; the
                 question is how much headroom gamma* has, and whether the
                 LP solution's robustness (sigma90, per-fact radii) matches
                 the trained model's. "Trained = max-margin point of its own
                 active-set geometry" predicts: close.

  scratch        pattern from a random density-matched initialization,
                 readout a fixed random codebook -- no gradient descent
                 anywhere in the pipeline. If this is feasible with a healthy
                 margin at trained capacity, inequality-solving alone
                 recovers the trained (capacity, robustness) point, and the
                 residual ~30x of docs/what-gd-builds.md 2 is closed
                 constructively. (Whether an LP counts as a rules-legal
                 entry is argued in that doc; as analysis it needs no
                 defense.)

    uv run python probe_maxmargin.py --d 32
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

from handcode.data import generate_facts
from handcode.model import ModelShape, accuracy, hidden_activations, random_init
from probe_digitcode import noise_curve, sigma90
from probe_reachability import run_adam
from probe_robustness import margin_stats, rms
from probe_structure import per_fact_radii

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def solve_max_margin(
    active: np.ndarray,  # (n, d) bool -- the frozen pattern
    inputs: np.ndarray,  # (n, 2) token ids
    targets: np.ndarray,  # (n,)
    readout: np.ndarray,  # (L, d) frozen
    n_vocab: int,
    box: float,
    wrong_sets: list | None = None,  # per-fact wrong-class subsets (cutting plane)
    pattern_rows: bool = True,  # False: masked-linear margins only, signs free
    spread_tau: float | None = None,  # per-fact capped-sum objective (see below)
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Max-min-margin embeddings for a frozen (pattern, readout). Returns
    (u, v, gamma*, info); u and v are (n_vocab, d).

    spread_tau switches the objective from the shared min margin to spread
    pressure: one margin variable m_f per fact, the same rows written against
    m_f instead of gamma, bounds m_f <= tau (free below), maximize sum m_f.
    Every fact then pulls until it reaches tau, instead of all pressure
    concentrating on the single worst fact. Returns (u, v, sum m_f, info)
    with the per-fact m in info["m"]."""
    n, d = active.shape
    n_labels = readout.shape[0]
    emb_cols = 2 * n_vocab * d
    nvar = emb_cols + (n if spread_tau is not None else 1)  # u, v, margins
    gamma_col = nvar - 1

    rows, cols, data = [], [], []
    row_count = 0

    # Margin rows: for fact f and wrong class c,
    #   sum_{i active} (W[c,i] - W[l,i]) (u[a,i] + v[b,i]) + gamma <= 0.
    for f in range(n):
        act = np.flatnonzero(active[f])
        if len(act) == 0:
            continue
        a, b, lab = inputs[f, 0], inputs[f, 1], targets[f]
        if wrong_sets is not None:
            wrong = np.asarray(wrong_sets[f])
        else:
            wrong = np.array([c for c in range(n_labels) if c != lab])
        if len(wrong) == 0:
            continue
        wdiff = readout[wrong][:, act] - readout[lab][act]  # (L-1, k)
        k = len(act)
        n_rows = len(wrong)
        r = row_count + np.repeat(np.arange(n_rows), 2 * k + 1)
        cu = a * d + act
        cv = n_vocab * d + b * d + act
        margin_col = gamma_col if spread_tau is None else emb_cols + f
        c_all = np.concatenate([cu, cv, [margin_col]])
        cc = np.tile(c_all, n_rows)
        dd = np.concatenate(
            [wdiff, wdiff, np.ones((n_rows, 1))], axis=1
        ).ravel()
        rows.append(r)
        cols.append(cc)
        data.append(dd)
        row_count += n_rows

    # Pattern rows: active -> -(u+v) <= 0, inactive -> +(u+v) <= 0.
    if pattern_rows:
        f_idx, i_idx = np.nonzero(np.ones_like(active))
        sign = np.where(active[f_idx, i_idx], -1.0, 1.0)
        r = row_count + np.arange(len(f_idx))
        cu = inputs[f_idx, 0] * d + i_idx
        cv = n_vocab * d + inputs[f_idx, 1] * d + i_idx
        rows.append(np.repeat(r, 2))
        cols.append(np.stack([cu, cv], axis=1).ravel())
        data.append(np.repeat(sign, 2))
        row_count += len(f_idx)

    A = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(row_count, nvar),
    )
    b_ub = np.zeros(row_count)
    c = np.zeros(nvar)
    if spread_tau is None:
        c[gamma_col] = -1.0  # maximize gamma
        bounds = [(-box, box)] * (nvar - 1) + [(None, None)]
    else:
        c[emb_cols:] = -1.0  # maximize sum of per-fact margins
        bounds = [(-box, box)] * emb_cols + [(None, spread_tau)] * n

    t = time.time()
    res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method="highs-ipm",
                  options={"time_limit": 900})
    info = {
        "status": res.status, "message": res.message,
        "seconds": round(time.time() - t, 1),
        "n_rows": row_count, "nnz": int(A.nnz),
    }
    if res.status != 0:
        return None, None, float("-inf"), info
    x = res.x
    u = x[: n_vocab * d].reshape(n_vocab, d)
    v = x[n_vocab * d : 2 * n_vocab * d].reshape(n_vocab, d)
    if spread_tau is not None:
        info["m"] = x[emb_cols:].copy()
    return u, v, float(-res.fun), info


def solve_with_cuts(
    active: np.ndarray,
    inputs: np.ndarray,
    targets: np.ndarray,
    readout: np.ndarray,
    n_vocab: int,
    box: float,
    k0: int = 8,
    max_rounds: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Cutting-plane max-margin: start with each fact's k0 most-confusable
    wrong classes (by readout-row similarity), solve, then add any class that
    beats the solved margin and re-solve. The returned gamma is the *true*
    min margin of the final solution over all classes, so a truncated
    constraint set can only make it conservative, never wrong."""
    n, d = active.shape
    n_labels = readout.shape[0]
    sim = readout @ readout.T  # confusability proxy
    hint = sim[targets].copy()
    hint[np.arange(n), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]

    total = {"seconds": 0.0, "rounds": 0}
    for _ in range(max_rounds):
        u, v, gamma, info = solve_max_margin(
            active, inputs, targets, readout, n_vocab, box, wrong_sets=wrong_sets
        )
        total["seconds"] += info["seconds"]
        total["rounds"] += 1
        total.update({k: info[k] for k in ("status", "message", "n_rows", "nnz")})
        if u is None:
            return None, None, float("-inf"), total
        # True margins of the solved model, all classes.
        h = active * (u[inputs[:, 0]] + v[inputs[:, 1]])  # frozen-pattern hidden
        logits = h @ readout.T
        correct = logits[np.arange(n), targets]
        logits[np.arange(n), targets] = -np.inf
        margins = correct - logits.max(1)
        true_gamma = float(margins.min())
        # Any (fact, class) pair inside the solved gamma that we left out?
        gaps = correct[:, None] - logits  # (n, L); target col is +inf-ish
        violated = gaps < gamma - 1e-7
        new = 0
        for f in np.flatnonzero(violated.any(1)):
            for c in np.flatnonzero(violated[f]):
                if c != targets[f] and c not in wrong_sets[f]:
                    wrong_sets[f].append(int(c))
                    new += 1
        total["added_cuts"] = new
        if new == 0:
            return u, v, true_gamma, total
    return u, v, true_gamma, total


def to_model(u: np.ndarray, v: np.ndarray, readout: np.ndarray):
    up = torch.from_numpy(np.concatenate([u.T, v.T], axis=1)).float()
    down = torch.from_numpy(readout).float()
    return up, down


def evaluate(up, down, facts, label: str) -> dict:
    entry = {
        "accuracy": accuracy(up, down, facts),
        "margins": margin_stats(up, down, facts),
        "sigma90_weight": sigma90(noise_curve(up, down, facts)),
        "radii": per_fact_radii(up, down, facts, n_sample=256),
        "rms_up": rms(up),
        "rms_down": rms(down),
    }
    m, r = entry["margins"], entry["radii"]
    s90 = entry["sigma90_weight"]
    print(
        f"{label:26s} acc={entry['accuracy']:.4f} "
        f"margin med={m['median']:.3g} min={m['min']:.3g} "
        f"sigma90={s90 if s90 is None else f'{s90:.2e}'} "
        f"radius med={r['radius_median']:.2e} min={r['radius_min']:.2e}",
        flush=True,
    )
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--n-facts", type=int, default=None)
    parser.add_argument("--scratch-ns", type=int, nargs="*", default=None)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "maxmargin.json"))
    args = parser.parse_args()

    d = args.d
    shape = ModelShape.from_d(d)
    trained_capacity = {16: 496, 32: 2080}[d]
    n = args.n_facts or trained_capacity
    facts = generate_facts(n, shape.input_vocab_size, shape.output_vocab_size, 42)
    inputs = facts["inputs"].numpy()
    targets = facts["targets"].numpy()
    results = {"d": d, "n_facts": n}

    # -- condition A: reconstruct the trained model's own geometry ----------
    rand_up, rand_down = random_init(shape, 1000)
    _, _, t_up, t_down = run_adam(rand_up, rand_down, facts, n_epochs=5000, lr=1e-2)
    results["trained"] = evaluate(t_up, t_down, facts, "trained (post budget)")

    with torch.no_grad():
        pre = hidden_activations(t_up, facts["inputs"])
    pattern = (pre > 0).numpy()
    readout = t_down.numpy().astype(np.float64)
    box = float(np.abs(t_up.numpy()).max())

    u, v, gamma, info = solve_with_cuts(
        pattern, inputs, targets, readout, shape.input_vocab_size, box
    )
    results["reconstruct_lp"] = {"gamma": gamma, "box": box, **info}
    print(f"LP(reconstruct): gamma*={gamma:.3f} "
          f"(trained min margin {results['trained']['margins']['min']:.3f}) "
          f"[{info['seconds']}s, {info['n_rows']} rows]", flush=True)
    if u is not None:
        lp_up, lp_down = to_model(u, v, readout)
        results["reconstruct"] = evaluate(lp_up, lp_down, facts, "max-margin (same geometry)")

    # -- decomposition: which frozen ingredient carries the capacity? -------
    gen0 = torch.Generator().manual_seed(7)
    offset = float(torch.special.ndtri(torch.tensor(0.53))) / 2**0.5
    u0 = torch.randn(shape.input_vocab_size, d, generator=gen0) + offset
    v0 = torch.randn(shape.input_vocab_size, d, generator=gen0) + offset
    pre0 = u0[facts["inputs"][:, 0]] + v0[facts["inputs"][:, 1]]
    rand_pattern = (pre0 > 0).numpy()
    rand_codebook = torch.randn(
        shape.output_vocab_size, d, generator=gen0
    ).numpy().astype(np.float64)

    for name, pat, ro in (
        ("trained-pattern+random-codebook", pattern, rand_codebook),
        ("random-pattern+trained-readout", rand_pattern, readout),
    ):
        uu, vv, g, inf = solve_with_cuts(
            pat, inputs, targets, ro, shape.input_vocab_size, box=box
        )
        row = {"gamma": g, **inf}
        print(f"LP({name}): gamma*={g:.4f} [{inf['seconds']}s]", flush=True)
        if uu is not None and g > 0:
            m_up, m_down = to_model(uu, vv, ro)
            row["metrics"] = evaluate(m_up, m_down, facts, name)
        results[name] = row

    # -- condition B: no gradient descent anywhere --------------------------
    results["scratch"] = []
    for n_s in args.scratch_ns or (n,):
        facts_s = generate_facts(
            n_s, shape.input_vocab_size, shape.output_vocab_size, 42
        )
        gen = torch.Generator().manual_seed(7)
        offset = float(torch.special.ndtri(torch.tensor(0.53))) / 2**0.5
        u0 = torch.randn(shape.input_vocab_size, d, generator=gen) + offset
        v0 = torch.randn(shape.input_vocab_size, d, generator=gen) + offset
        pre0 = u0[facts_s["inputs"][:, 0]] + v0[facts_s["inputs"][:, 1]]
        pattern0 = (pre0 > 0).numpy()
        codebook = torch.randn(
            shape.output_vocab_size, d, generator=gen
        ).numpy().astype(np.float64)

        u, v, gamma, info = solve_with_cuts(
            pattern0, facts_s["inputs"].numpy(), facts_s["targets"].numpy(),
            codebook, shape.input_vocab_size, box=6.0,
        )
        row = {"n_facts": n_s, "gamma": gamma, **info}
        print(f"LP(scratch, n={n_s}): gamma*={gamma:.4f} [{info['seconds']}s]",
              flush=True)
        if u is not None and gamma > 0:
            s_up, s_down = to_model(u, v, codebook)
            row["metrics"] = evaluate(s_up, s_down, facts_s, f"scratch LP n={n_s}")
        results["scratch"].append(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
