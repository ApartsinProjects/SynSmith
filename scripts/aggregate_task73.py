"""Aggregate task #73 Banking77 results: SynSmith vs SynSmith+sibling-rejection."""
import json, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [17, 23, 41, 53, 89]

print(f"{'seed':>6}  {'SynSmith':>10}  {'SynSmith+sib':>14}  {'delta':>8}")
print("-" * 46)

fa_vals, fs_vals = [], []
for s in SEEDS:
    path = REPO / f"experiments/task73_v2_10_1_seed{s}/all_summaries.json"
    if not path.exists():
        print(f"{s:>6}  missing all_summaries.json")
        continue
    with open(path) as f:
        runs = json.load(f)
    by_cond = {r["condition"]: r for r in runs}
    a_run = by_cond.get("full_attrforge", {}).get("final_downstream", {})
    b_run = by_cond.get("full_attrforge_sibling", {}).get("final_downstream", {})
    a = a_run.get("macro_f1") if isinstance(a_run, dict) else None
    b = b_run.get("macro_f1") if isinstance(b_run, dict) else None
    if a is not None and b is not None:
        fa_vals.append(a)
        fs_vals.append(b)
        delta = b - a
        print(f"{s:>6}  {a:>10.3f}  {b:>14.3f}  {delta:>+8.3f}")
    else:
        print(f"{s:>6}  a={a}  b={b}")

if len(fa_vals) >= 2 and len(fs_vals) >= 2:
    print()
    print(f"{'mean':>6}  {st.mean(fa_vals):>10.3f}  {st.mean(fs_vals):>14.3f}  "
          f"{st.mean(fs_vals) - st.mean(fa_vals):>+8.3f}")
    print(f"{'stdev':>6}  {st.stdev(fa_vals):>10.3f}  {st.stdev(fs_vals):>14.3f}")
    # paired-t hand calc
    diffs = [b - a for a, b in zip(fa_vals, fs_vals)]
    if len(diffs) >= 2:
        d_mean = st.mean(diffs)
        d_sd = st.stdev(diffs)
        n = len(diffs)
        t = d_mean / (d_sd / (n ** 0.5))
        print(f"\npaired diffs n={n}, mean={d_mean:+.4f}, sd={d_sd:.4f}, t={t:.3f}")
