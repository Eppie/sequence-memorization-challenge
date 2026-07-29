"""Rate-distortion of the trained gate: how few bits name a good pattern?

The declarative program (FINDINGS 14-23) asked whether any first-pass
statistic separates the trained gate from constructed ones, and every answer
was no. This probe asks the quantitative version instead: compress the
trained solution's *description* -- quantize the pre-activation embeddings
that generate the pattern, truncate their rank, cluster the token vectors,
sparsify, or flip the pattern's near-tie bits directly -- and measure what
each compressed description still buys, both as a pattern (ceiling sigma90
under the two-LP ascent, FINDINGS 14's measure, weights re-solved from the
pattern alone) and as a deployable model (accuracy and weight-noise sigma90
of the compressed weights themselves).

The output is a curve: description length in bits against the best ceiling
that description reaches. The endpoints are known -- the digit code's formula
(a few hundred bits) reaches 2.85e-3 and the float32 trained embeddings
(131,072 bits) reach 4.40e-2. A knee at a few bits/param means compressible
structure the statistics missed; a smooth decay toward the float32 point
means the pattern's shortest description is "the weights, rounded" -- the
quantitative form of theory 6'''.

Protocol notes: ceilings use probe_gatequality.ascend_best (3 rounds) started
from the variant's own embeddings with the trained readout, matching the
predict-phase protocol; quantization-induced pattern flips concentrate on
near-tie bits by construction, so the quant sweep doubles as an interpolation
of the near-tie perturbation result out to large flip fractions, with random
flips as the control arm.

    uv run python probe_ratedistortion.py --families direct
    uv run python probe_ratedistortion.py                      # everything
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

import probe_gatequality as gq
from handcode.data import generate_facts
from handcode.model import ModelShape, accuracy
from probe_digitcode import noise_curve, sigma90

OUT_PATH = os.path.join(gq.RESULTS_DIR, "ratedistortion.json")


def done_keys() -> set:
    """Entries already present in any ratedistortion output file, so parallel
    processes (each with its own --out) never redo or clobber work."""
    keys = set()
    for path in glob.glob(os.path.join(gq.RESULTS_DIR, "ratedistortion*.json")):
        try:
            with open(path) as f:
                keys |= set(json.load(f).get("entries", {}))
        except (json.JSONDecodeError, OSError):
            pass
    return keys

QUANT_KS = (2, 3, 4, 6, 8, 16, 32)
DIRECT_KS = (2, 3, 4, 6, 8, 16, 32, 64)
LOWRANK_RS = (2, 4, 8, 16, 24)
W_RANK_RS = (1, 2, 4, 8, 16)
CODEBOOK_CS = (4, 8, 16, 32, 64)
SPARSE_FS = (0.9, 0.75, 0.5, 0.25, 0.1)
SPARSE_LP_FS = (0.5, 0.25)
TIEFLIP_FRACS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
RANDFLIP_FRACS = (0.005, 0.02)
FLOAT_BITS = 32


# --------------------------------------------------------------------------
# state plumbing
# --------------------------------------------------------------------------

def split_uv(up: torch.Tensor, V: int):
    u = up[:, :V].T.numpy().astype(np.float64)
    v = up[:, V:].T.numpy().astype(np.float64)
    return u, v


def join_uv(u: np.ndarray, v: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([u.T, v.T], axis=1)).float()


def pattern_of(u: np.ndarray, v: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    return (u[inputs[:, 0]] + v[inputs[:, 1]]) > 0


# --------------------------------------------------------------------------
# quantizers and description-length accounting
# --------------------------------------------------------------------------

def quant_linear(x: np.ndarray, k: int):
    lo, hi = float(x.min()), float(x.max())
    step = (hi - lo) / (k - 1)
    idx = np.rint((x - lo) / step).astype(int)
    centers = lo + np.arange(k) * step
    return centers[idx], idx


def quant_quantile(x: np.ndarray, k: int, iters: int = 25):
    """1-D Lloyd refinement from quantile-seeded centers."""
    flat = x.ravel()
    centers = np.quantile(flat, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        idx = np.abs(flat[:, None] - centers[None, :]).argmin(1)
        for j in range(k):
            m = idx == j
            if m.any():
                centers[j] = flat[m].mean()
    idx = np.abs(flat[:, None] - centers[None, :]).argmin(1)
    return centers[idx].reshape(x.shape), idx.reshape(x.shape)


QUANTIZERS = {"linear": quant_linear, "quantile": quant_quantile}


def entropy_bits(idx: np.ndarray, k: int) -> float:
    """Shannon bound for the level stream (codebook charged separately)."""
    counts = np.bincount(idx.ravel(), minlength=k).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) * idx.size


def quant_bits(idx: np.ndarray, k: int) -> dict:
    return {
        "uniform": idx.size * float(np.log2(k)) + k * FLOAT_BITS,
        "entropy": entropy_bits(idx, k) + k * FLOAT_BITS,
    }


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def direct_metrics(up: torch.Tensor, down: torch.Tensor, facts) -> dict:
    s90 = sigma90(noise_curve(up, down, facts))
    return {"accuracy": accuracy(up, down, facts), "sigma90_weight": s90}


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


class Probe:
    def __init__(self, shape, facts):
        self.shape, self.facts = shape, facts
        self.done = done_keys()
        self.inputs = facts["inputs"].numpy()
        pattern, up, down = gq.get_gate("trained", shape, facts)
        self.pattern_t, self.up_t, self.down_t = pattern, up, down
        self.u_t, self.v_t = split_uv(up, shape.input_vocab_size)
        self.W_t = down.numpy().astype(np.float64)
        self.pre_t = (self.u_t[self.inputs[:, 0]]
                      + self.v_t[self.inputs[:, 1]])
        regen = pattern_of(self.u_t, self.v_t, self.inputs)
        mismatch = int((regen != pattern).sum())
        print(f"[setup] trained gate loaded; regenerated pattern differs on "
              f"{mismatch}/{pattern.size} bits", flush=True)

    def ceiling(self, tag, pattern, up_start, down_start) -> dict:
        """Two-LP ascent ceiling of a pattern, published protocol."""
        best, history = gq.ascend_best(pattern, self.facts, self.shape,
                                       up_start, down_start, label=tag)
        entry = {"best_gamma": best["gamma"],
                 "infeasible": best["gamma"] <= 1e-9}
        entry["metrics"] = gq.evaluate_full(best["u"], best["v"], best["W"],
                                            self.facts, tag)
        entry["n_lp_steps"] = len(history) - 1
        return entry

    def record(self, family, tag, entry):
        merge_out(f"{family}/{tag}", entry)
        print(f"[saved] {family}/{tag}", flush=True)

    def skip(self, family, tag) -> bool:
        if f"{family}/{tag}" in self.done:
            print(f"[skip] {family}/{tag} already done", flush=True)
            return True
        return False

    def flip_frac(self, pattern) -> float:
        return float((pattern != self.pattern_t).mean())

    # -- families ----------------------------------------------------------

    def fam_baseline(self):
        if self.skip("baseline", "trained_float32"):
            return
        entry = {
            "bits": {"uniform": float(2 * self.shape.input_vocab_size
                                      * self.shape.d_mlp * FLOAT_BITS)},
            "flip_frac": 0.0,
            "direct": direct_metrics(self.up_t, self.down_t, self.facts),
            "ceiling": self.ceiling("baseline", self.pattern_t, self.up_t,
                                    self.down_t),
        }
        self.record("baseline", "trained_float32", entry)

    def fam_direct(self):
        """No-LP sweep: quantize / rank-truncate / sparsify the weights and
        evaluate them as-deployed (which side carries the bits?)."""
        for scheme, quant in QUANTIZERS.items():
            for k in DIRECT_KS:
                uq, iu = quant(self.u_t, k)
                vq, iv = quant(self.v_t, k)
                Wq, iw = quant(self.W_t, k)
                up_q = join_uv(uq, vq)
                down_q = torch.from_numpy(Wq).float()
                pat = pattern_of(uq, vq, self.inputs)
                bits_e = quant_bits(np.concatenate([iu.ravel(), iv.ravel()]),
                                    k)
                entry = {
                    "k": k, "scheme": scheme,
                    "bits_emb": bits_e,
                    "bits_readout": quant_bits(iw, k),
                    "flip_frac": self.flip_frac(pat),
                    "emb_only": direct_metrics(up_q, self.down_t, self.facts),
                    "readout_only": direct_metrics(self.up_t, down_q,
                                                   self.facts),
                    "both": direct_metrics(up_q, down_q, self.facts),
                }
                self.record("direct_quant", f"{scheme}_k{k}", entry)

        for r in W_RANK_RS:
            U, S, Vt = np.linalg.svd(self.W_t, full_matrices=False)
            Wr = (U[:, :r] * S[:r]) @ Vt[:r]
            entry = {
                "r": r,
                "bits": r * sum(self.W_t.shape) * FLOAT_BITS,
                "readout_only": direct_metrics(
                    self.up_t, torch.from_numpy(Wr).float(), self.facts),
            }
            self.record("direct_wrank", f"r{r}", entry)

        for f in SPARSE_FS:
            entry = {"keep_frac": f}
            entry.update(self._sparse_state(f, direct_only=True))
            self.record("direct_sparse", f"f{f}", entry)

    def _quant_state(self, scheme, k):
        quant = QUANTIZERS[scheme]
        uq, iu = quant(self.u_t, k)
        vq, iv = quant(self.v_t, k)
        pat = pattern_of(uq, vq, self.inputs)
        bits = quant_bits(np.concatenate([iu.ravel(), iv.ravel()]), k)
        return uq, vq, pat, bits

    def fam_quant(self, schemes):
        """The headline curve: ceiling of the pattern generated by k-level
        embeddings, i.e. bits needed to *name* a good pattern."""
        for scheme in schemes:
            for k in QUANT_KS:
                tag = f"{scheme}_k{k}"
                if self.skip("quant", tag):
                    continue
                uq, vq, pat, bits = self._quant_state(scheme, k)
                entry = {
                    "k": k, "scheme": scheme, "bits": bits,
                    "flip_frac": self.flip_frac(pat),
                    "direct": direct_metrics(join_uv(uq, vq), self.down_t,
                                             self.facts),
                    "ceiling": self.ceiling(tag, pat, join_uv(uq, vq),
                                            self.down_t),
                }
                self.record("quant", tag, entry)

    def fam_lowrank(self):
        tok = np.concatenate([self.u_t, self.v_t], axis=0)  # (2V, d)
        U, S, Vt = np.linalg.svd(tok, full_matrices=False)
        V = self.shape.input_vocab_size
        for r in LOWRANK_RS:
            tag = f"r{r}"
            if self.skip("lowrank", tag):
                continue
            tok_r = (U[:, :r] * S[:r]) @ Vt[:r]
            ur, vr = tok_r[:V], tok_r[V:]
            pat = pattern_of(ur, vr, self.inputs)
            entry = {
                "r": r,
                "bits": r * (tok.shape[0] + tok.shape[1] + 1) * FLOAT_BITS,
                "flip_frac": self.flip_frac(pat),
                "direct": direct_metrics(join_uv(ur, vr), self.down_t,
                                         self.facts),
                "ceiling": self.ceiling(f"lowrank_{tag}", pat,
                                        join_uv(ur, vr), self.down_t),
            }
            self.record("lowrank", tag, entry)

    def fam_codebook(self):
        tok = np.concatenate([self.u_t, self.v_t], axis=0)
        V = self.shape.input_vocab_size
        for c in CODEBOOK_CS:
            tag = f"c{c}"
            if self.skip("codebook", tag):
                continue
            centers, labels = kmeans2(tok, c, minit="++", seed=0)
            tok_c = centers[labels]
            uc, vc = tok_c[:V], tok_c[V:]
            pat = pattern_of(uc, vc, self.inputs)
            entry = {
                "c": c,
                "bits": c * tok.shape[1] * FLOAT_BITS
                + tok.shape[0] * float(np.log2(c)),
                "flip_frac": self.flip_frac(pat),
                "direct": direct_metrics(join_uv(uc, vc), self.down_t,
                                         self.facts),
                "ceiling": self.ceiling(f"codebook_{tag}", pat,
                                        join_uv(uc, vc), self.down_t),
            }
            self.record("codebook", tag, entry)

    def _sparse_state(self, f, direct_only=False):
        tok = np.concatenate([self.u_t.ravel(), self.v_t.ravel()])
        thresh = np.quantile(np.abs(tok), 1 - f)
        us = np.where(np.abs(self.u_t) >= thresh, self.u_t, 0.0)
        vs = np.where(np.abs(self.v_t) >= thresh, self.v_t, 0.0)
        pat = pattern_of(us, vs, self.inputs)
        nnz = int((us != 0).sum() + (vs != 0).sum())
        out = {
            "nnz": nnz,
            "bits": nnz * (FLOAT_BITS + float(np.log2(tok.size))),
            "flip_frac": self.flip_frac(pat),
            "direct": direct_metrics(join_uv(us, vs), self.down_t,
                                     self.facts),
        }
        if not direct_only:
            out["ceiling"] = self.ceiling(f"sparse_f{f}", pat,
                                          join_uv(us, vs), self.down_t)
        return out

    def fam_sparse(self):
        for f in SPARSE_LP_FS:
            if self.skip("sparse", f"f{f}"):
                continue
            entry = {"keep_frac": f}
            entry.update(self._sparse_state(f))
            self.record("sparse", f"f{f}", entry)

    def fam_tieflip(self):
        """Flip the q fraction of pattern bits nearest zero (the near-tie
        direction), vs random flips of the same count -- the perturbation
        result as a full curve. No bits column: this is tolerance, not a
        code."""
        order = np.argsort(np.abs(self.pre_t).ravel())
        rng = np.random.default_rng(0)
        for q in TIEFLIP_FRACS:
            tag = f"tie_q{q}"
            if self.skip("tieflip", tag):
                continue
            n_flip = round(q * self.pattern_t.size)
            mask = np.zeros(self.pattern_t.size, dtype=bool)
            mask[order[:n_flip]] = True
            pat = self.pattern_t ^ mask.reshape(self.pattern_t.shape)
            entry = {"q": q, "n_flip": n_flip, "kind": "near_tie",
                     "ceiling": self.ceiling(tag, pat, self.up_t,
                                             self.down_t)}
            self.record("tieflip", tag, entry)
        for q in RANDFLIP_FRACS:
            tag = f"rand_q{q}"
            if self.skip("tieflip", tag):
                continue
            n_flip = round(q * self.pattern_t.size)
            mask = np.zeros(self.pattern_t.size, dtype=bool)
            mask[rng.choice(self.pattern_t.size, n_flip, replace=False)] = True
            pat = self.pattern_t ^ mask.reshape(self.pattern_t.shape)
            entry = {"q": q, "n_flip": n_flip, "kind": "random",
                     "ceiling": self.ceiling(tag, pat, self.up_t,
                                             self.down_t)}
            self.record("tieflip", tag, entry)


FAMILIES = ("baseline", "direct", "quant", "lowrank", "codebook", "sparse",
            "tieflip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES),
                        choices=FAMILIES)
    parser.add_argument("--schemes", nargs="*",
                        default=list(QUANTIZERS),
                        choices=list(QUANTIZERS))
    parser.add_argument("--out", default=None,
                        help="output JSON (default results/ratedistortion.json;"
                        " give each parallel process its own file)")
    parser.add_argument("--lp-time-limit", type=float, default=None,
                        help="override HiGHS time_limit (seconds). A first-"
                        "round emb-LP timeout records as infeasible with "
                        "nothing proven -- near-edge instances under "
                        "concurrent LP load need this raised")
    gq.add_cell_args(parser)
    args = parser.parse_args()
    gq.configure(args.d, args.n_facts, args.fact_seed)
    if args.out:
        global OUT_PATH
        OUT_PATH = args.out
    if args.lp_time_limit is not None:
        # probe_geometry_ascent imports linprog function-locally, so patching
        # the scipy attribute reaches every solve.
        import scipy.optimize
        orig = scipy.optimize.linprog

        def patched(*a, **kw):
            opts = dict(kw.get("options") or {})
            if "time_limit" in opts:
                opts["time_limit"] = args.lp_time_limit
            kw["options"] = opts
            return orig(*a, **kw)

        scipy.optimize.linprog = patched

    shape = ModelShape.from_d(gq.D)
    facts = generate_facts(gq.N_FACTS, shape.input_vocab_size,
                           shape.output_vocab_size, gq.FACT_SEED)
    print(f"[cell] d={gq.D} n_facts={gq.N_FACTS} fact_seed={gq.FACT_SEED}\n"
          f"[out] {OUT_PATH}", flush=True)

    probe = Probe(shape, facts)
    for fam in args.families:
        print(f"== family {fam}", flush=True)
        if fam == "quant":
            probe.fam_quant(args.schemes)
        else:
            getattr(probe, f"fam_{fam}")()


if __name__ == "__main__":
    main()
