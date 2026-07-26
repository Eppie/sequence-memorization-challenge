"""Their open question: why is MLP+Norms+NoRes+NoBias+ReLU so bad?

The post's appendix B flags one architecture combination that does far worse
than its parts predict -- "almost as bad as having no MLP at all" -- and says:
"We don't know why. Specifically, we don't know if the limitation is due to
training dynamics or due to what is possible for this architecture."

This probe runs their own vendored model (reference/models.py) at their own
experiment scale (V=32, d_res=d_ff=16, L=16) and tests the two mechanisms our
other measurements suggest:

  collapse   with Res=0 and Norms=1 the head reads RMSNorm(ff_out). A fact
             whose ReLU units all go dead has ff_out ~ 0, and RMSNorm of a
             near-zero vector amplifies numerical noise -- the fact becomes
             undecodable and its gradients spike. Measured: fraction of dead
             units, and of facts with an all-dead hidden layer.
  plateaus   the same architecture pins how cross-entropy can grow logits
             (no residual bypass around the norm), which is the plateau
             signature FINDINGS.md 7 found for the fixed quadratic decode:
             4% accuracy under the post's patience, full accuracy with a
             long one. Measured: the same model under a short and a 10x
             training budget. If the gap closes with budget, their answer is
             "training dynamics"; if not, "architecture".

One-setting-flipped controls (GELU, Bias=1, Res=1, Norms=0) are run alongside,
since the post reports each of those variants is fine.

    uv run python probe_badcombo.py
"""

import json
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
sys.path.insert(0, REFERENCE_DIR)
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from models import MemoryToyModel, ModelSettings  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_VALUES = (256, 384, 512, 640)
BASE = dict(
    input_vocab_size=32, output_vocab_size=16, d_residual=16, d_ff=16,
    attention=False,  # the 2Emb mixing, their most stable variant
)
CONDITIONS = {
    "bad (norms,noB,noRes,ReLU)": dict(norms=True, bias=False, ff_residual=False,
                                       ff_activation_type="ReLU"),
    "flip act -> GELU": dict(norms=True, bias=False, ff_residual=False,
                             ff_activation_type="GELU"),
    "flip bias -> on": dict(norms=True, bias=True, ff_residual=False,
                            ff_activation_type="ReLU"),
    "flip res -> on": dict(norms=True, bias=False, ff_residual=True,
                           ff_activation_type="ReLU"),
    "flip norms -> off": dict(norms=False, bias=False, ff_residual=False,
                              ff_activation_type="ReLU"),
}


def train(model, n_epochs, patience, lr=1e-2):
    inputs, targets = model.facts["inputs"], model.facts["targets"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, since, best_state = 0.0, 0, None
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
        with torch.no_grad():
            acc = float((model(inputs).argmax(-1) == targets).float().mean())
        if acc > best:
            best, since = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if best == 1.0 or since >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def diagnostics(model) -> dict:
    with torch.no_grad():
        idx = model.facts["inputs"]
        x = sum(emb(idx[:, i]) for i, emb in enumerate(model.token_emb))
        pre = model.ff[0](model.ln2(x))
        h = model.ff[1](pre)  # after the activation
        logits = model(idx)
        active = (h > 0).float()
        return {
            "dead_unit_fraction": float((active.sum(0) == 0).float().mean()),
            "all_dead_fact_fraction": float((active.sum(1) == 0).float().mean()),
            "mean_density": float(active.mean()),
            "logit_rms": float(logits.pow(2).mean().sqrt()),
        }


def main() -> None:
    records = []
    for name, flags in CONDITIONS.items():
        for n in N_VALUES:
            budgets = {"short (5k, pat 100)": (5000, 100)}
            if name.startswith("bad") or name.startswith("flip norms"):
                budgets["long (60k, pat 10k)"] = (60000, 10000)
            for bname, (epochs, patience) in budgets.items():
                best, best_model, t = 0.0, None, time.time()
                for seed in (0, 1):
                    torch.manual_seed(seed)
                    model = MemoryToyModel(ModelSettings(n_facts=n, **BASE, **flags))
                    acc = train(model, epochs, patience)
                    if acc > best:
                        best, best_model = acc, model
                    if best == 1.0:
                        break
                diag = diagnostics(best_model)
                records.append({"condition": name, "n_facts": n, "budget": bname,
                                "accuracy": best, **diag,
                                "seconds": round(time.time() - t, 1)})
                print(f"{name:28s} n={n:<4d} {bname:20s} acc={best:.4f} "
                      f"dead_units={diag['dead_unit_fraction']:.2f} "
                      f"all_dead_facts={diag['all_dead_fact_fraction']:.3f} "
                      f"logit_rms={diag['logit_rms']:.1f} "
                      f"({records[-1]['seconds']}s)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "badcombo.json"), "w") as f:
        json.dump(records, f, indent=2)
    print("\nwrote results/badcombo.json")


if __name__ == "__main__":
    main()
