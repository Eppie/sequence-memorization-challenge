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

    uv run python probe_flippolicy.py --phase dense --workers 12
    uv run python probe_flippolicy.py --phase stats
    uv run python probe_flippolicy.py --phase interp --pair 180 200
    uv run python probe_flippolicy.py --phase direction
    uv run python probe_flippolicy.py --phase stride --rounds 40
"""

import argparse
import json
import os

import numpy as np
import torch

from handcode.data import generate_facts
from handcode.model import ModelShape, hidden_activations
from probe_gatequality import (
    CURVE_CKPT_PATH,
    D,
    GATES_DIR,
    N_FACTS,
    RESULTS_DIR,
    ascend_best,
    evaluate_full,
    train_curve_checkpoints,
)

WINDOW_EPOCHS = tuple(range(140, 361, 20))
NULL_EPOCHS = (500, 700)
CKPT_PATH = os.path.join(GATES_DIR, "window_checkpoints.npz")
EDGE_PATH = os.path.join(GATES_DIR, "edge_state.npz")
OUT_PATH = os.path.join(RESULTS_DIR, "flippolicy.json")
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
    shape = ModelShape.from_d(D)
    facts = generate_facts(N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, 42)
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
    with ProcessPoolExecutor(max_workers=workers) as ex:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",
                        choices=("dense", "stats", "interp", "direction",
                                 "stride"),
                        default="dense")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--cap", type=float, default=0.005,
                        help="flip cap per stride round; 0.005 is gradient "
                             "descent's own per-step gross flip rate")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--snap-every", type=int, default=4)
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
    args = parser.parse_args()

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
        from probe_geometry_ascent import readout_lp

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
        pattern0 = (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0
        box_e = max(6.0, 1.05 * float(np.abs(np.concatenate([u, v])).max()))
        box_w = max(6.0, 1.05 * float(np.abs(W).max()))

        history = []
        snaps = {0: (join_uv(u, v), W.copy())}
        pattern = pattern0.copy()
        for r in range(1, args.rounds + 1):
            h = pattern * (u[inputs[:, 0]] + v[inputs[:, 1]])
            u_star, v_star, obj, m, info = lp_spread(
                pattern, inputs, targets, W, shape.input_vocab_size, box_e,
                logits_hint=h @ W.T, tau=args.tau,
            )
            if u_star is None:
                print(f"  [stride] LP failed at round {r}: "
                      f"{info.get('message')}", flush=True)
                break
            u, v, t, flips = capped_step(u, v, u_star, v_star, inputs,
                                         args.cap)
            pattern = (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0
            h = np.maximum(u[inputs[:, 0]] + v[inputs[:, 1]], 0.0)
            W_new, g_w, _ = readout_lp(h, targets, shape.output_vocab_size,
                                       box_w)
            if W_new is not None:
                W = W_new
            acc = float(((h @ W.T).argmax(1) == targets).mean())
            row = {"round": r, "mean_m": obj / len(targets), "step": t,
                   "flips": flips, "acc": acc,
                   "net_drift": float((pattern != pattern0).mean())}
            history.append(row)
            print(f"  [stride] {r}: mean_m={row['mean_m']:.4f} step={t:.4f} "
                  f"flips={flips} acc={acc:.4f} "
                  f"net_drift={row['net_drift']:.4f}", flush=True)
            if r % args.snap_every == 0 or r == args.rounds:
                snaps[r] = (join_uv(u, v), W.copy())
        merge_out({"stride": {"cap": args.cap, "tau": args.tau,
                              "history": history}})
        jobs = [(f"stride_r{r}", up_np, W_np)
                for r, (up_np, W_np) in sorted(snaps.items())]
        run_ascents(jobs, args.workers)
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
