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

    uv run python probe_flippolicy.py --phase dense --workers 12
    uv run python probe_flippolicy.py --phase stats
    uv run python probe_flippolicy.py --phase interp --pair 160 240
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
OUT_PATH = os.path.join(RESULTS_DIR, "flippolicy.json")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dense", "stats", "interp"),
                        default="dense")
    parser.add_argument("--pair", type=int, nargs=2, default=(160, 240))
    parser.add_argument("--ts", type=float, nargs="*",
                        default=(0.25, 0.5, 0.75))
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


if __name__ == "__main__":
    main()
