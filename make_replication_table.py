"""Assemble the replication table (FINDINGS.md 22) from the result JSONs.

Reads results/gatequality*.json and results/flippolicy_s4*.json and prints the
per-row verdict against the published d=32/seed-42 numbers. Regenerating this
after any further run keeps the table and the data in step.

    uv run python make_replication_table.py
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def jload(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def ceilings(fname):
    """{round: sigma90} from a stride run's ascents, plus its history."""
    d = jload(fname)
    if not d:
        return {}, []
    a = d.get("ascent", {})
    ce = {int(k.split("_r")[1]): a[k]["metrics"]["sigma90_weight"]
          for k in a if "_r" in k}
    k = next((x for x in d if x.startswith("stride")), None)
    return ce, (d.get(k, {}).get("history", []) if k else [])


def feas(s):
    """The LPs return sigma90=1e-5 exactly when gamma*=0 (theory.md Prop 1a),
    i.e. infeasibility of any positive margin -- not a small robustness."""
    return s is not None and s > 1.1e-5


def fmt(s):
    return "infeasible" if not feas(s) else f"{s:.3g}"


print("=" * 78)
print("SECTION 14 -- the gate-zoo feasibility split")
print("=" * 78)
cells = [("published d=32/s42", "gatequality.json"),
         ("d=32 seed 43", "gatequality_d32_n1584_s43.json"),
         ("d=32 seed 44", "gatequality_d32_n1584_s44.json"),
         ("d=64 seed 42 (n=6336)", "gatequality_d64_n6336_s42.json")]
gates = ("trained", "digit_m2", "random_additive", "init")
print(f"{'cell':<24}" + "".join(f"{g:>18}" for g in gates))
for label, f in cells:
    d = jload(f)
    if not d or "predict" not in d:
        print(f"{label:<24}{'(not run)':>18}")
        continue
    row = f"{label:<24}"
    for g in gates:
        v = d["predict"].get(g)
        row += f"{fmt(v['metrics']['sigma90_weight']) if v else '-':>18}"
    print(row)

print()
print("=" * 78)
print("SECTION 19/20/21 -- stride ceilings by round")
print("=" * 78)
runs = [("S19  edge      ", "flippolicy_s43_edge.json", "s43"),
        ("S19  edge      ", "flippolicy_s44_edge.json", "s44"),
        ("S20  ep50      ", "flippolicy_s43_ep50.json", "s43"),
        ("S20  ep50      ", "flippolicy_s44_ep50.json", "s44"),
        ("S20  ep100     ", "flippolicy_s43_ep100.json", "s43"),
        ("S20  ep100     ", "flippolicy_s44_ep100.json", "s44"),
        ("S20  ep150     ", "flippolicy_s43_ep150.json", "s43"),
        ("S21  fw        ", "flippolicy_s43_fw.json", "s43"),
        ("S21  fwfull .2%", "flippolicy_s43_fwfull02.json", "s43"),
        ("S21  fwfull 5% ", "flippolicy_s43_fwfull5pct.json", "s43"),
        ("S21  fwfull 5% ", "flippolicy_s43_fwfull5pct_long.json", "s43 long")]
for label, f, seed in runs:
    ce, h = ceilings(f)
    if not ce and not h:
        continue
    ks = sorted(ce)
    cross = next((r for r in ks if feas(ce[r])), None)
    tail = " ".join(f"r{r}={fmt(ce[r])}" for r in ks[-3:]) if ks else ""
    print(f"{label} {seed}  rounds={len(h):>3} "
          f"crossed@={'never' if cross is None else f'r{cross}':>6}  "
          f"peak={fmt(max(ce.values(), key=lambda x: x or 0)) if ce else '-':>10}  {tail}")

print()
print("=" * 78)
print("SECTION 17 -- the dense ceiling curve (seed 43 vs published seed 42)")
print("=" * 78)
pub = {200: 1.39e-2, 220: 2.0e-2, 240: 2.8e-2, 260: 3.4e-2, 280: 3.7e-2}
d = jload("flippolicy_s43_dense.json")
if d:
    a = d.get("ascent", {})
    rows = sorted((int(k[2:]), v["metrics"]["sigma90_weight"])
                  for k, v in a.items() if k.startswith("ep"))
    print(f"{'epoch':>7}{'seed 42 (published)':>22}{'seed 43':>14}{'ratio':>9}")
    for e, s in rows:
        p = pub.get(e)
        r = f"{s/p:.2f}" if (p and feas(s)) else "-"
        print(f"{e:>7}{(f'{p:.3g}' if p else '-'):>22}{fmt(s):>14}{r:>9}")
