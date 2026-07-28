"""Which flips build the gate? The policy inside the window.

FINDINGS.md 16 located gate construction in a ~50-epoch window at the end of
fitting: infeasible through train acc 0.77 (epoch 150), 4.5x the best
construction by acc 0.90 (epoch 200), 92% of final at first acc=1 (355) --
crossing only ~4.6% of pattern bits. This probe asks what distinguishes the
flips in that window from the churn that surrounds them, in three phases:

  dense     re-run the trajectory, snapshot every 20 epochs across the
            window (140..360) plus a null pair (500, 700) in the churn
            regime and the epoch-5000 endpoint, and LP-ascend each window
            checkpoint in parallel. Resolves the transition's shape between
            the coarse points. (Torch CPU training is not bit-deterministic
            -- parallel-reduction order -- and the trajectory is chaotic, so
            this is an independent micro-realization of the same run: macro
            observables match the coarse trajectory, bit-level patterns
            diverge ~1%. All within-probe comparisons use this run's own
            snapshots, including its own epoch-5000 for flip survival.)

  stats     characterize the flipped bits of each consecutive pair against
            the pair's unflipped bits and against the null pair: are they
            near-ties (|preactivation| at the earlier checkpoint), are they
            concentrated on then-misclassified facts (enrichment over the
            error-fact bit share), which direction do they flip, do they
            survive to epoch 5000, and how concentrated on neurons are they.

  interp    the causal probe. Per-bit hybrids of two patterns are not
            additively realizable, so interpolate in *embedding* space --
            u_t = (1-t) u_a + t u_b is always a real embedding -- and ascend
            each interpolate's pattern. Does the ceiling arrive gradually in
            t, or at a threshold where the decisive flips switch in?

  direction the constructive test. From one infeasible edge state (epoch
            180 of a fresh short run), take the SAME flip-budget-matched
            capped step along seven directions -- the realized 20-epoch
            training delta, the realized one-epoch Adam step, the raw loss
            gradient, fit-pressure spread LPs (tau = 0.1, 0.5), the max-min
            margin LP vertex, and a random control -- and LP-ascend each
            stepped gate. Same step mechanics everywhere; only the direction
            differs. Also characterizes the realized direction's decisive
            flipped bits.

  stride    the process thesis, tested constructively. Iterate the section
            15 drift machinery unchanged (spread LP -> order-statistic
            capped step -> re-read pattern -> readout-LP refit, no memory)
            but seeded from the INFEASIBLE epoch-180 fitting state with the
            flip cap at gradient descent's own stride (~0.5% of bits per
            full-batch step) -- i.e., simulate the process with an exact-
            solve direction oracle at matched granularity. Snapshots are
            LP-ascended afterwards in parallel: does any round's pattern
            become storage-feasible, as GD's does within ~20 steps?

  softseed  the missing cell of the seed problem (FINDINGS 20). That section
            found the flow needs half a fit AND a soft geometry, but its
            constructed seed (digit) had full fit and stiff geometry -- the
            confound was never broken. Build ridge-only seeds spanning
            (fit, softness) -- handcode/softseed.py: heavy step-ridge keeps
            the geometry near its maximally soft init while fit climbs as
            far as capped motion carries it -- and report each seed's train
            accuracy and near-tie profile against the GD-edge and digit
            references. Seeds are saved as softseed_<tag>.npz; feed one to
            the flow with --stride-seed soft:<tag>. If any cell bootstraps,
            the constructive account closes end to end with no GD anywhere.

    uv run python probe_flippolicy.py --phase dense --workers 12
    uv run python probe_flippolicy.py --phase stats
    uv run python probe_flippolicy.py --phase interp --pair 180 200
    uv run python probe_flippolicy.py --phase direction
    uv run python probe_flippolicy.py --phase stride --rounds 40
    uv run python probe_flippolicy.py --phase stride --stride-seed scratch \
        --rounds 100 --snap-every 10       # theory.md problem 6': no GD at all
    uv run python probe_flippolicy.py --phase softseed
    uv run python probe_flippolicy.py --phase stride --stride-seed soft:s0 \
        --oracle fw-full --rounds 400      # the softness hypothesis, GD-free
"""

import argparse
import json
import os

import numpy as np
import torch

from handcode.data import generate_facts
from handcode.model import ModelShape, hidden_activations
import probe_gatequality as pgq
from probe_gatequality import (
    RESULTS_DIR,
    ascend_best,
    evaluate_full,
    train_curve_checkpoints,
)

WINDOW_EPOCHS = tuple(range(140, 361, 20))
NULL_EPOCHS = (500, 700)
CURVE_CKPT_PATH = pgq.CURVE_CKPT_PATH
CKPT_PATH = os.path.join(pgq.GATES_DIR, "window_checkpoints.npz")
EDGE_PATH = os.path.join(pgq.GATES_DIR, "edge_state.npz")
OUT_PATH = os.path.join(RESULTS_DIR, "flippolicy.json")


def refresh_paths() -> None:
    """Re-derive every path after pgq.configure() moved the cell."""
    global CKPT_PATH, EDGE_PATH, OUT_PATH, CURVE_CKPT_PATH
    tag = ("" if (pgq.D, pgq.N_FACTS, pgq.FACT_SEED) == (32, 1584, 42)
           else f"_d{pgq.D}_n{pgq.N_FACTS}_s{pgq.FACT_SEED}")
    CURVE_CKPT_PATH = pgq.CURVE_CKPT_PATH
    CKPT_PATH = os.path.join(pgq.GATES_DIR, "window_checkpoints.npz")
    EDGE_PATH = os.path.join(pgq.GATES_DIR, "edge_state.npz")
    OUT_PATH = os.path.join(RESULTS_DIR, f"flippolicy{tag}.json")
EDGE_EPOCH = 180
EDGE_T = 0.125  # the interp step that crossed the edge (FINDINGS.md 17)


def merge_out(update: dict) -> None:
    data = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            data = json.load(f)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def setup():
    shape = ModelShape.from_d(pgq.D)
    facts = generate_facts(pgq.N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, pgq.FACT_SEED)
    return shape, facts


def load_ckpt(z, epoch, facts):
    """(up, down, preacts, pattern, wrong-fact mask) at a snapshot epoch."""
    up = torch.from_numpy(z[f"up_{epoch}"]).float()
    down = torch.from_numpy(z[f"down_{epoch}"]).float()
    with torch.no_grad():
        n_vocab = up.shape[1] // 2
        emb = up.T
        pre = (emb[facts["inputs"][:, 0]]
               + emb[facts["inputs"][:, 1] + n_vocab]).numpy()
        logits = hidden_activations(up, facts["inputs"]) @ down.T
        wrong = (logits.argmax(-1) != facts["targets"]).numpy()
    return up, down, pre, pre > 0, wrong


def ascend_worker(tag: str, up_np: np.ndarray, down_np: np.ndarray) -> dict:
    """Ascend the gate of the given model (runs in a worker process)."""
    shape, facts = setup()
    up = torch.from_numpy(up_np).float()
    down = torch.from_numpy(down_np).float()
    with torch.no_grad():
        pattern = (hidden_activations(up, facts["inputs"]) > 0).numpy()
    best, history = ascend_best(pattern, facts, shape, up, down, label=tag)
    entry = {"best_gamma": best["gamma"]}
    entry["metrics"] = evaluate_full(best["u"], best["v"], best["W"], facts,
                                     f"{tag} ceiling")
    return entry


def run_ascents(jobs, workers):
    """jobs: list of (tag, up_np, down_np). Returns {tag: entry}, merging
    into OUT_PATH under 'ascent' as each finishes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = {}
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=pgq.configure,
                             initargs=(pgq.D, pgq.N_FACTS,
                                       pgq.FACT_SEED)) as ex:
        futures = {ex.submit(ascend_worker, *job): job[0] for job in jobs}
        for fut in as_completed(futures):
            tag = futures[fut]
            rows[tag] = fut.result()
            merge_out({"ascent": dict(rows)})
            m = rows[tag]["metrics"]
            print(f"== {tag}: ceiling sigma90={m['sigma90_weight']} "
                  f"acc={m['accuracy']:.3f} ({len(rows)}/{len(jobs)} done)",
                  flush=True)
    return rows


def pair_stats(z, a, b, facts, pat_final):
    """What distinguishes the a->b flipped bits from the rest."""
    _, _, pre_a, P_a, wrong_a = load_ckpt(z, a, facts)
    _, _, _, P_b, _ = load_ckpt(z, b, facts)
    flipped = P_a != P_b
    n_flips = int(flipped.sum())
    if n_flips == 0:
        return {"n_flips": 0}
    abspre = np.abs(pre_a)
    wrong_bit_share = float(wrong_a.mean())
    flips_on_wrong = float(flipped[wrong_a].sum() / n_flips)
    top4 = np.sort(flipped.sum(0))[-4:].sum() / n_flips
    return {
        "n_flips": n_flips,
        "flip_frac": float(flipped.mean()),
        "train_acc_a": float(1 - wrong_a.mean()),
        "median_abspre_flipped": float(np.median(abspre[flipped])),
        "median_abspre_all": float(np.median(abspre)),
        "wrong_fact_bit_share": wrong_bit_share,
        "flips_on_wrong_facts": flips_on_wrong,
        "wrong_fact_enrichment": (flips_on_wrong / wrong_bit_share
                                  if wrong_bit_share > 0 else None),
        "off_to_on_frac": float(P_b[flipped].mean()),
        "survive_to_final": float((P_b[flipped] == pat_final[flipped]).mean()),
        "top4_neuron_share": float(top4),
    }


def readout_lp_spread(h, targets, n_labels, box, tau, k0=12, max_rounds=4,
                      wrong_sets=None, solver_options=None):
    """Capped-sum margins over the readout, activations frozen -- the
    fit-pressure analog of probe_geometry_ascent.readout_lp. The max-min
    readout refit is degenerate off the feasible set (its optimum is the
    do-nothing W = 0 whenever some fact cannot be fixed; theory.md Prop 1a,
    observed as a death spiral in the first stride run), so the stride
    process needs spread pressure in this block too."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, d = h.shape
    emb_cols = n_labels * d
    nvar = emb_cols + n

    if wrong_sets is None:
        means = np.stack([h[targets == c].mean(0) if (targets == c).any()
                          else np.zeros(d) for c in range(n_labels)])
        hint = h @ means.T
        hint[np.arange(n), targets] = -np.inf
        wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :k0]]

    W = m = None
    for _ in range(max_rounds):
        rows, cols, data = [], [], []
        row_count = 0
        for f in range(n):
            lab = targets[f]
            hf = h[f]
            nz = np.flatnonzero(hf)
            for c in wrong_sets[f]:
                r = row_count
                rows.extend([r] * (2 * len(nz) + 1))
                cols.extend((lab * d + nz).tolist())
                data.extend((-hf[nz]).tolist())
                cols.extend((c * d + nz).tolist())
                data.extend(hf[nz].tolist())
                cols.append(emb_cols + f)
                data.append(1.0)
                row_count += 1
        A = sparse.csr_matrix(
            (np.array(data), (np.array(rows), np.array(cols))),
            shape=(row_count, nvar),
        )
        c_obj = np.zeros(nvar)
        c_obj[emb_cols:] = -1.0
        bounds = [(-box, box)] * emb_cols + [(None, tau)] * n
        res = linprog(c_obj, A_ub=A, b_ub=np.zeros(row_count), bounds=bounds,
                      method="highs-ipm",
                      options={"time_limit": 300, **(solver_options or {})})
        if res.status != 0:
            return W, m, {"status": res.status, "message": res.message}
        W = res.x[:emb_cols].reshape(n_labels, d)
        m = res.x[emb_cols:]
        logits = h @ W.T
        correct = logits[np.arange(n), targets]
        logits[np.arange(n), targets] = -np.inf
        gaps = correct[:, None] - logits
        new = 0
        for f in np.flatnonzero((gaps < m[:, None] - 1e-7).any(1)):
            for c in np.flatnonzero(gaps[f] < m[f] - 1e-7):
                if c != targets[f] and c not in wrong_sets[f]:
                    wrong_sets[f].append(int(c))
                    new += 1
        if new == 0:
            break
    return W, m, {"status": 0}


def fw_targets(pattern, inputs, targets, W, u, v, tau, box_e, box_w):
    """The cheapest possible direction oracle: linearize the capped-sum
    (spread) objective at the current point and solve the linearization
    over the weight box exactly -- which is sign-snapping, a matvec. The
    binding wrong class c* per still-pulling fact gives the subgradient;
    coordinates it doesn't touch keep their current value. Returns the
    box-vertex targets for embeddings and readout plus (min gap, mean
    capped gap) for logging."""
    n = len(targets)
    h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
    logits = h @ W.T
    correct = logits[np.arange(n), targets]
    lg = logits.copy()
    lg[np.arange(n), targets] = -np.inf
    cstar = lg.argmax(1)
    gap = correct - lg[np.arange(n), cstar]
    active = gap < tau

    coef = pattern[active] * (W[targets[active]] - W[cstar[active]])
    Gu = np.zeros_like(u)
    Gv = np.zeros_like(v)
    np.add.at(Gu, inputs[active, 0], coef)
    np.add.at(Gv, inputs[active, 1], coef)
    u_star = np.where(Gu != 0, box_e * np.sign(Gu), u)
    v_star = np.where(Gv != 0, box_e * np.sign(Gv), v)

    Gw = np.zeros_like(W)
    np.add.at(Gw, targets[active], h[active])
    np.add.at(Gw, cstar[active], -h[active])
    W_star = np.where(Gw != 0, box_w * np.sign(Gw), W)
    return (u_star, v_star, W_star, float(gap.min()),
            float(np.minimum(gap, tau).mean()))


def split_uv(up_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(d, 2V) up matrix -> (V, d) u and v, float64."""
    d = up_np.shape[0]
    n_vocab = up_np.shape[1] // 2
    u = up_np[:, :n_vocab].T.astype(np.float64)
    v = up_np[:, n_vocab:].T.astype(np.float64)
    return u, v


def join_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([u.T, v.T], axis=1)


def step_to_flip_budget(u, v, du, dv, inputs, k):
    """Scale s along (du, dv) so exactly k pre-activation signs flip --
    the order-statistic step of the drift experiments, with the budget
    fixed instead of the fraction. Returns (u', v', s) or None if the
    direction cannot flip k bits at any scale."""
    pre = u[inputs[:, 0]] + v[inputs[:, 1]]
    dpre = du[inputs[:, 0]] + dv[inputs[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -pre / dpre
    ts = np.sort(t[np.isfinite(t) & (t > 0)])
    if len(ts) < k:
        return None
    s = float(ts[k - 1]) * (1 + 1e-9)
    return u + s * du, v + s * dv, s


def build_directions(z, facts, shape):
    """All seven direction fields at the epoch-180 edge state, as
    (du, dv) in (V, d) coordinates."""
    import torch.nn.functional as F

    from probe_maxmargin import solve_max_margin

    u0, v0 = split_uv(z[f"up_{EDGE_EPOCH}"])
    down0 = z[f"down_{EDGE_EPOCH}"].astype(np.float64)
    inputs = facts["inputs"].numpy()
    targets = facts["targets"].numpy()
    pattern = (u0[inputs[:, 0]] + v0[inputs[:, 1]]) > 0
    box = max(6.0, 1.05 * float(np.abs(np.concatenate([u0, v0])).max()))

    dirs = {}
    for name, e in (("realized20", 200), ("adam1", EDGE_EPOCH + 1)):
        ue, ve = split_uv(z[f"up_{e}"])
        dirs[name] = (ue - u0, ve - v0)

    up_t = torch.from_numpy(z[f"up_{EDGE_EPOCH}"]).float().requires_grad_(True)
    down_t = torch.from_numpy(z[f"down_{EDGE_EPOCH}"]).float()
    logits = hidden_activations(up_t, facts["inputs"]) @ down_t.T
    F.cross_entropy(logits, facts["targets"]).backward()
    gu, gv = split_uv(up_t.grad.numpy())
    dirs["gradient"] = (-gu, -gv)

    h = pattern * (u0[inputs[:, 0]] + v0[inputs[:, 1]])
    hint = h @ down0.T
    hint[np.arange(len(targets)), targets] = -np.inf
    wrong_sets = [list(row) for row in np.argsort(-hint, axis=1)[:, :12]]
    for name, tau in (("fitlp_tau0.1", 0.1), ("fitlp_tau0.5", 0.5),
                      ("minmax", None)):
        u_s, v_s, val, info = solve_max_margin(
            pattern, inputs, targets, down0, shape.input_vocab_size, box,
            wrong_sets=wrong_sets, pattern_rows=False, spread_tau=tau,
        )
        if u_s is None:
            print(f"  [direction] {name} LP failed: {info.get('message')}",
                  flush=True)
            continue
        dirs[name] = (u_s - u0, v_s - v0)
        print(f"  [direction] {name}: objective {val:.3f} "
              f"[{info['seconds']}s]", flush=True)

    rng = np.random.default_rng(0)
    dirs["random"] = (rng.standard_normal(u0.shape),
                      rng.standard_normal(v0.shape))
    return u0, v0, down0, pattern, dirs


def decisive_bits_report(flipped, pre, inputs, wrong):
    """Relational profile of a flipped-bit set."""
    f_idx, j_idx = np.nonzero(flipped)
    n_bits = len(f_idx)
    return {
        "n_bits": n_bits,
        "n_facts_touched": int(len(np.unique(f_idx))),
        "wrong_fact_share": float(wrong[f_idx].mean()),
        "wrong_fact_base": float(wrong.mean()),
        "median_abspre": float(np.median(np.abs(pre[flipped]))),
        "off_to_on_frac": float((pre[flipped] < 0).mean()),
        "distinct_first_token_cells": int(
            len(set(zip(inputs[f_idx, 0].tolist(), j_idx.tolist())))),
        "distinct_second_token_cells": int(
            len(set(zip(inputs[f_idx, 1].tolist(), j_idx.tolist())))),
    }


def main() -> None:
    global OUT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",
                        choices=("dense", "stats", "interp", "direction",
                                 "stride", "softseed"),
                        default="dense")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--cap", type=float, default=0.005,
                        help="flip cap per stride round; 0.005 is gradient "
                             "descent's own per-step gross flip rate")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--snap-every", type=int, default=4)
    parser.add_argument("--stride-seed", default="edge",
                        help="edge: the infeasible epoch-180 GD fitting "
                             "state (FINDINGS 19). scratch: gradient "
                             "descent's own random init, seed 1000 -- no "
                             "GD anywhere (theory.md problem 6'). epN "
                             "(e.g. ep100): the coarse curve run's epoch-N "
                             "checkpoint -- the GD-prefix sweep. soft:TAG: "
                             "a ridge-built soft seed from --phase softseed "
                             "(softseed_TAG.npz) -- GD-free end to end.")
    parser.add_argument("--resume", action="store_true",
                        help="continue a previous stride run of the same "
                             "seed from its saved final state")
    parser.add_argument("--k0", type=int, default=12,
                        help="initial wrong classes per fact in the stride "
                             "LPs; the cut loop repairs anything missed, so "
                             "smaller is faster and stays exact")
    parser.add_argument("--ipm-tol", type=float, default=None,
                        help="loosen HiGHS ipm_optimality_tolerance for the "
                             "stride LPs (e.g. 1e-3): the step only uses the "
                             "direction toward the optimum, so a half-"
                             "converged point serves; ~2-4x per solve")
    parser.add_argument("--oracle", choices=("lp", "fw", "fw-full"),
                        default="lp",
                        help="stride direction oracle. lp: the exact spread "
                             "LPs (FINDINGS 19). fw: emb target = box vertex "
                             "of the linearized objective (a matvec, no "
                             "solver), readout refit still exact. fw-full: "
                             "both blocks first-order -- the cheapest "
                             "possible oracle, rounds cost milliseconds")
    parser.add_argument("--fw-tw", type=float, default=0.05,
                        help="fw-full only: step fraction toward the "
                             "readout's box-vertex target per round")
    parser.add_argument("--no-ascend", action="store_true",
                        help="stride only: skip the LP ascents of the "
                             "snapshots (cheap smoke runs / tuning; the "
                             "history and state npz are still written)")
    parser.add_argument("--soft-mu", type=float, nargs="*",
                        default=(1e-3, 1e-2, 1e-1, 1.0),
                        help="softseed sweep: step-ridge values (the "
                             "fit-vs-softness knob)")
    parser.add_argument("--soft-lam", type=float, nargs="*",
                        default=(1e-1,),
                        help="softseed sweep: readout ridge values")
    parser.add_argument("--soft-rounds", type=int, default=40)
    parser.add_argument("--soft-cap", type=float, default=0.02,
                        help="softseed: per-round flip cap (0.05 reaches the "
                             "fit plateau ~2x faster than twosided's 0.02)")
    parser.add_argument("--soft-soften", type=float, default=None,
                        help="softseed: construct the near-tie reservoir to "
                             "this q0.5%%/median ratio (GD edge states "
                             "measure ~1.1e-3; unset leaves the ~6-9e-3 the "
                             "ridge rounds land at)")
    parser.add_argument("--soft-wrms", type=float, default=None,
                        help="softseed: rescale the seed readout to this rms "
                             "(GD's epoch-180 readout measures 1.28; the raw "
                             "ridge readout is ~19x smaller, which both lets "
                             "box-scale vertex steps swamp it and keeps every "
                             "fact below tau)")
    parser.add_argument("--soft-acc-stop", type=float, default=None,
                        help="softseed: stop a build at this train accuracy "
                             "(targets a fit band instead of a plateau)")
    parser.add_argument("--soft-init-seed", type=int, default=2000)
    parser.add_argument("--out", default=None,
                        help="override the results JSON (lets concurrent "
                             "stride runs avoid write races; merge after)")
    parser.add_argument("--pair", type=int, nargs=2, default=(160, 240))
    parser.add_argument("--ts", type=float, nargs="*",
                        default=(0.25, 0.5, 0.75))
    parser.add_argument("--edge-t", type=float, default=EDGE_T,
                        help="calibration point on the realized 180->200 "
                             "path: the flip budget k is what the realized "
                             "direction spends by this t. The edge's "
                             "location varies by realization; pick the "
                             "smallest t whose interpolate is feasible.")
    parser.add_argument("--workers", type=int, default=12)
    pgq.add_cell_args(parser)
    args = parser.parse_args()
    pgq.configure(args.d, args.n_facts, args.fact_seed)
    refresh_paths()
    print(f"[cell] d={pgq.D} n_facts={pgq.N_FACTS} "
          f"fact_seed={pgq.FACT_SEED}\n[cell] gates={pgq.GATES_DIR}\n"
          f"[cell] out={OUT_PATH}", flush=True)
    if args.out:
        OUT_PATH = args.out

    shape, facts = setup()
    if not os.path.exists(CKPT_PATH):
        train_curve_checkpoints(shape, facts,
                                epochs=WINDOW_EPOCHS + NULL_EPOCHS + (5000,),
                                n_epochs=5000, path=CKPT_PATH)
    z = np.load(CKPT_PATH)
    if os.path.exists(CURVE_CKPT_PATH):  # macro sanity vs the coarse run
        zc = np.load(CURVE_CKPT_PATH)
        acc = dict(zip(z["epochs"].tolist(), z["acc"].tolist()))
        accc = dict(zip(zc["epochs"].tolist(), zc["acc"].tolist()))
        print(f"macro check vs coarse run: acc@200 {acc[200]:.4f} vs "
              f"{accc[200]:.4f}, first_acc1 {int(z['first_acc1'])} vs "
              f"{int(zc['first_acc1'])} (bit-level divergence expected)",
              flush=True)

    if args.phase == "dense":
        jobs = []
        for e in WINDOW_EPOCHS:
            up, down, _, _, _ = load_ckpt(z, e, facts)
            jobs.append((f"ep{e}", up.numpy(), down.numpy()))
        rows = run_ascents(jobs, args.workers)
        merge_out({"dense_epochs": list(WINDOW_EPOCHS)})
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "stats":
        _, _, _, pat_final, _ = load_ckpt(z, 5000, facts)
        rows = {}
        pairs = list(zip(WINDOW_EPOCHS[:-1], WINDOW_EPOCHS[1:]))
        pairs.append(NULL_EPOCHS)
        for a, b in pairs:
            key = f"{a}-{b}"
            rows[key] = pair_stats(z, a, b, facts, pat_final)
            r = rows[key]
            enrich = r["wrong_fact_enrichment"]
            print(f"{key:>9}: flips={r['n_flips']:>6} "
                  f"|pre| flip/all={r['median_abspre_flipped']:.3f}"
                  f"/{r['median_abspre_all']:.3f} "
                  f"wrong-enrich="
                  f"{'n/a' if enrich is None else f'{enrich:.2f}'} "
                  f"off->on={r['off_to_on_frac']:.2f} "
                  f"survive={r['survive_to_final']:.2f} "
                  f"top4={r['top4_neuron_share']:.2f}", flush=True)
        merge_out({"pair_stats": rows})
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "interp":
        a, b = args.pair
        up_a, down_a, _, P_a, _ = load_ckpt(z, a, facts)
        up_b, down_b, _, P_b, _ = load_ckpt(z, b, facts)
        jobs = []
        meta = {}
        for t in args.ts:
            up_t = ((1 - t) * up_a + t * up_b).numpy()
            down_t = ((1 - t) * down_a + t * down_b).numpy()
            with torch.no_grad():
                P_t = (hidden_activations(torch.from_numpy(up_t).float(),
                                          facts["inputs"]) > 0).numpy()
            tag = f"interp{a}-{b}@{t}"
            meta[tag] = {"t": t,
                         "dist_to_a": float((P_t != P_a).mean()),
                         "dist_to_b": float((P_t != P_b).mean())}
            jobs.append((tag, up_t, down_t))
        merge_out({"interp_meta": meta})
        run_ascents(jobs, args.workers)
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "direction":
        if not os.path.exists(EDGE_PATH):
            train_curve_checkpoints(
                shape, facts, epochs=(EDGE_EPOCH, EDGE_EPOCH + 1, 200),
                n_epochs=200, path=EDGE_PATH,
            )
        ze = np.load(EDGE_PATH)
        u0, v0, down0, pattern0, dirs = build_directions(ze, facts, shape)
        inputs = facts["inputs"].numpy()
        pre0 = u0[inputs[:, 0]] + v0[inputs[:, 1]]
        _, _, _, _, wrong0 = load_ckpt(ze, EDGE_EPOCH, facts)

        # The flip budget: what the realized direction spends at --edge-t.
        et = args.edge_t
        du, dv = dirs["realized20"]
        dpre = du[inputs[:, 0]] + dv[inputs[:, 1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cross = -pre0 / dpre
        k = int((np.isfinite(t_cross) & (t_cross > 0)
                 & (t_cross <= et)).sum())
        print(f"  [direction] flip budget k={k} "
              f"(realized20 at t={et})", flush=True)

        sfx = f"@t{et}"
        jobs = [(f"base180{sfx}", join_uv(u0, v0), down0)]
        meta, flips = {}, {}
        for name, (du, dv) in dirs.items():
            stepped = step_to_flip_budget(u0, v0, du, dv, inputs, k)
            if stepped is None:
                print(f"  [direction] {name}: cannot flip {k} bits", flush=True)
                continue
            u_s, v_s, s = stepped
            pat_s = (u_s[inputs[:, 0]] + v_s[inputs[:, 1]]) > 0
            flips[name] = pat_s != pattern0
            h_s = np.maximum(u_s[inputs[:, 0]] + v_s[inputs[:, 1]], 0.0)
            acc_s = float(((h_s @ down0.T).argmax(1)
                           == facts["targets"].numpy()).mean())
            meta[name] = {"scale": s, "n_flips": int(flips[name].sum()),
                          "stepped_acc": acc_s}
            jobs.append((name + sfx, join_uv(u_s, v_s), down0))
            if name == "realized20":
                # readout-start sensitivity: same embeddings, the ascent
                # seeded with the t-interpolated readout instead
                down_i = (1 - et) * down0 + et * ze["down_200"].astype(
                    np.float64)
                jobs.append((f"realized20_ro{sfx}", join_uv(u_s, v_s), down_i))
        if "realized20" in flips:
            merge_out({"decisive_bits": decisive_bits_report(
                flips["realized20"], pre0, inputs, wrong0)})
            for name in meta:
                meta[name]["overlap_with_realized"] = float(
                    (flips[name] & flips["realized20"]).sum() / max(k, 1))
        merge_out({"direction_meta": {f"t{et}": {"k": k, **meta}}})
        run_ascents(jobs, args.workers)
        print(f"\nwrote {OUT_PATH}")

    elif args.phase == "stride":
        from probe_gatequality import capped_step, lp_spread

        seed_name = args.stride_seed
        run_name = (seed_name if args.oracle == "lp"
                    else f"{seed_name}_{args.oracle.replace('-', '')}")
        run_name = run_name.replace(":", "-")
        state_path = os.path.join(pgq.GATES_DIR, f"stride_{run_name}_state.npz")
        r0 = 0
        u0_seed = None
        if args.resume and os.path.exists(state_path):
            zs = np.load(state_path)
            u, v = split_uv(zs["up"])
            W = zs["down"].astype(np.float64)
            u0_seed = split_uv(zs["up0"])
            r0 = int(zs["round"])
            print(f"  [stride] resuming {seed_name} from round {r0}",
                  flush=True)
        elif seed_name == "scratch":
            from handcode.model import random_init

            up0, down0 = random_init(shape, 1000)
            u, v = split_uv(up0.numpy())
            W = down0.numpy().astype(np.float64)
        elif seed_name == "digit":
            # The fully constructive seed: the pedestal-optimized digit
            # construction -- ridge solves only, no gradient descent
            # anywhere in its ancestry. With the stride process also
            # GD-free, this pipeline is GD-free end to end.
            from probe_gatequality import get_gate

            pattern_d, up_d, down_d = get_gate("digit_m2", shape, facts)
            u, v = split_uv(up_d.numpy())
            W = down_d.numpy().astype(np.float64)
        elif seed_name.startswith("soft"):
            # A ridge-built soft seed from --phase softseed: with the fw
            # oracles this pipeline has no gradient descent anywhere.
            tag = seed_name.split(":", 1)[1] if ":" in seed_name else "s0"
            zs = np.load(os.path.join(pgq.GATES_DIR, f"softseed_{tag}.npz"))
            u = zs["u"].astype(np.float64)
            v = zs["v"].astype(np.float64)
            W = zs["W"].astype(np.float64)
        elif seed_name.startswith("ep") and seed_name != "edge":
            e = int(seed_name[2:])
            zc = np.load(CURVE_CKPT_PATH)
            u, v = split_uv(zc[f"up_{e}"])
            W = zc[f"down_{e}"].astype(np.float64)
        else:
            if not os.path.exists(EDGE_PATH):
                train_curve_checkpoints(
                    shape, facts, epochs=(EDGE_EPOCH, EDGE_EPOCH + 1, 200),
                    n_epochs=200, path=EDGE_PATH,
                )
            ze = np.load(EDGE_PATH)
            u, v = split_uv(ze[f"up_{EDGE_EPOCH}"])
            W = ze[f"down_{EDGE_EPOCH}"].astype(np.float64)
        inputs = facts["inputs"].numpy()
        targets = facts["targets"].numpy()
        if u0_seed is None:
            up0_np = join_uv(u, v)
            pattern0 = (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0
        else:
            up0_np = np.asarray(np.load(state_path)["up0"])
            uu0, vv0 = u0_seed
            pattern0 = (uu0[inputs[:, 0]] + vv0[inputs[:, 1]]) > 0
        box_e = max(6.0, 1.05 * float(np.abs(np.concatenate([u, v])).max()))
        box_w = max(6.0, 1.05 * float(np.abs(W).max()))

        key = ("stride" if run_name == "edge" else f"stride_{run_name}")
        tagbase = ("stride" if run_name == "edge"
                   else f"stride_{run_name}")
        history = []
        if r0 and os.path.exists(OUT_PATH):
            with open(OUT_PATH) as f:
                history = json.load(f).get(key, {}).get("history", [])
        snaps = {} if r0 else {0: (join_uv(u, v), W.copy())}
        pattern = (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0
        ws_emb = ws_ro = None  # cut sets carried across rounds (grow-only)
        for r in range(r0 + 1, r0 + args.rounds + 1):
            h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
            sopts = ({"ipm_optimality_tolerance": args.ipm_tol}
                     if args.ipm_tol else None)
            if args.oracle == "lp":
                if ws_emb is None:
                    hint = h @ W.T
                    hint[np.arange(len(targets)), targets] = -np.inf
                    ws_emb = [list(row) for row in
                              np.argsort(-hint, axis=1)[:, :args.k0]]
                u_star, v_star, obj, m, info = lp_spread(
                    pattern, inputs, targets, W, shape.input_vocab_size,
                    box_e, logits_hint=h @ W.T, tau=args.tau, k0=args.k0,
                    wrong_sets=ws_emb, solver_options=sopts,
                )
                if u_star is None:
                    print(f"  [stride] LP failed at round {r}: "
                          f"{info.get('message')}", flush=True)
                    break
                mean_m = obj / len(targets)
            else:
                u_star, v_star, W_star, min_gap, mean_m = fw_targets(
                    pattern, inputs, targets, W, u, v, args.tau, box_e,
                    box_w)
            u, v, t, flips = capped_step(u, v, u_star, v_star, inputs,
                                         args.cap)
            pattern = (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0
            h = np.maximum(u[inputs[:, 0]] + v[inputs[:, 1]], 0.0)
            if args.oracle == "fw-full":
                W = W + args.fw_tw * (W_star - W)
            else:
                if ws_ro is None:
                    means = np.stack([h[targets == c].mean(0)
                                      if (targets == c).any()
                                      else np.zeros(h.shape[1])
                                      for c in range(shape.output_vocab_size)])
                    hint = h @ means.T
                    hint[np.arange(len(targets)), targets] = -np.inf
                    ws_ro = [list(row) for row in
                             np.argsort(-hint, axis=1)[:, :args.k0]]
                W_new, _, _ = readout_lp_spread(
                    h, targets, shape.output_vocab_size, box_w,
                    tau=args.tau, k0=args.k0, wrong_sets=ws_ro,
                    solver_options=sopts,
                )
                if W_new is not None:
                    W = W_new
            acc = float(((h @ W.T).argmax(1) == targets).mean())
            row = {"round": r, "mean_m": mean_m, "step": t,
                   "flips": flips, "acc": acc,
                   "net_drift": float((pattern != pattern0).mean())}
            history.append(row)
            print(f"  [stride] {r}: mean_m={row['mean_m']:.4f} step={t:.4f} "
                  f"flips={flips} acc={acc:.4f} "
                  f"net_drift={row['net_drift']:.4f}", flush=True)
            # Write-as-you-go: history to the JSON every round, resumable
            # state every snapshot -- a killed run loses at most snap_every
            # rounds and no data (the write-at-completion trap, retired).
            merge_out({key: {"cap": args.cap, "tau": args.tau,
                             "seed": seed_name, "oracle": args.oracle,
                             "history": history}})
            if r % args.snap_every == 0 or r == r0 + args.rounds:
                snaps[r] = (join_uv(u, v), W.copy())
                np.savez(state_path, up=join_uv(u, v), down=W, up0=up0_np,
                         round=np.array(r))
        if args.no_ascend:
            print(f"\nwrote {OUT_PATH} (snapshot ascents skipped)")
        else:
            jobs = [(f"{tagbase}_r{r}", up_np, W_np)
                    for r, (up_np, W_np) in sorted(snaps.items())]
            run_ascents(jobs, args.workers)
            print(f"\nwrote {OUT_PATH}")

    elif args.phase == "softseed":
        from handcode.softseed import (
            SoftSeedParams,
            softness_report,
            solve_soft_seed,
        )

        inputs = facts["inputs"].numpy()

        def soft_line(name, rep, acc=None):
            acc_s = "" if acc is None else f" acc={acc:.4f}"
            print(f"  [softseed] {name:14s}{acc_s} q005={rep['q005']:.2e} "
                  f"median={rep['median']:.3f} ratio={rep['ratio']:.2e} "
                  f"density={rep['density']:.3f}", flush=True)

        refs = {}
        if os.path.exists(EDGE_PATH):
            ze = np.load(EDGE_PATH)
            u_e, v_e = split_uv(ze[f"up_{EDGE_EPOCH}"])
            refs[f"ep{EDGE_EPOCH}_gd"] = softness_report(
                u_e[inputs[:, 0]] + v_e[inputs[:, 1]])
        digit_path = os.path.join(pgq.GATES_DIR, "digit_m2.npz")
        if os.path.exists(digit_path):
            zd = np.load(digit_path, allow_pickle=True)
            if zd["up"].size:
                u_d, v_d = split_uv(zd["up"].astype(np.float64))
                refs["digit_m2"] = softness_report(
                    u_d[inputs[:, 0]] + v_d[inputs[:, 1]])
        for name, rep in refs.items():
            soft_line(name, rep)

        merge_out({"softseed_refs": refs})
        rows = {}
        for mu in args.soft_mu:
            for lam in args.soft_lam:
                # Tags carry the full config, so repeated invocations with
                # different grids accumulate distinct seeds in the JSON and
                # the gate cache instead of overwriting s0, s1, ...
                tag = (f"mu{mu:g}-lam{lam:g}-r{args.soft_rounds}"
                       f"-c{args.soft_cap:g}")
                if args.soft_soften is not None:
                    tag += f"-s{args.soft_soften:g}"
                if args.soft_wrms is not None:
                    tag += f"-w{args.soft_wrms:g}"
                if args.soft_acc_stop is not None:
                    tag += f"-a{args.soft_acc_stop:g}"
                params = SoftSeedParams(mu=mu, lam=lam,
                                        rounds=args.soft_rounds,
                                        flip_cap=args.soft_cap,
                                        soften_ratio=args.soft_soften,
                                        w_rms=args.soft_wrms,
                                        acc_stop=args.soft_acc_stop)
                res = solve_soft_seed(shape, facts, params,
                                      seed=args.soft_init_seed)
                np.savez(os.path.join(pgq.GATES_DIR, f"softseed_{tag}.npz"),
                         u=res.u, v=res.v, W=res.W)
                rows[tag] = {"mu": mu, "lam": lam,
                             "rounds": args.soft_rounds,
                             "flip_cap": args.soft_cap,
                             "soften_ratio": args.soft_soften,
                             "w_rms": args.soft_wrms,
                             "acc_stop": args.soft_acc_stop,
                             "init_seed": args.soft_init_seed,
                             "rounds_run": len(res.history),
                             "accuracy": res.accuracy,
                             "softness": res.softness}
                soft_line(tag, res.softness, acc=res.accuracy)
                merge_out({"softseed": dict(rows)})
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
