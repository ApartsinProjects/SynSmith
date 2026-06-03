"""Aggregate task #69 customer-support results: SynSmith vs no_pack vs no_pack_vs.

Three conditions evaluated on customer-support intent classification:
- full_attrforge  : SynSmith reference (all 7 critics)
- no_pack         : SynSmith with the Pack Discriminator removed
- no_pack_vs      : SynSmith with Pack removed AND Verbalized Sampling generator
"""
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [17, 23, 41, 53]
CONDS = ["full_attrforge", "no_pack", "no_pack_vs"]

per_cond = {c: [] for c in CONDS}
print(f"{'seed':>6}  " + "  ".join(f"{c:>16}" for c in CONDS))
print("-" * (6 + 18 * len(CONDS)))

for s in SEEDS:
    path = REPO / f"experiments/task69_v2_10_1_seed{s}/all_summaries.json"
    if not path.exists():
        print(f"{s:>6}  missing")
        continue
    runs = json.load(open(path))
    by_cond = {r["condition"]: r for r in runs}
    row = []
    for c in CONDS:
        run = by_cond.get(c, {})
        fd = run.get("final_downstream") if isinstance(run, dict) else None
        if isinstance(fd, dict):
            mf1 = fd.get("macro_f1")
            wf1 = fd.get("worst_class_f1") or fd.get("per_class_f1_min")
            if mf1 is not None:
                per_cond[c].append(mf1)
                row.append(f"{mf1:.3f}/{wf1:.3f}" if wf1 else f"{mf1:.3f}/  -  ")
            else:
                row.append("  -   /  -  ")
        else:
            row.append("  -   /  -  ")
    print(f"{s:>6}  " + "  ".join(f"{x:>16}" for x in row))

print()
print(f"{'mean':>6}  " + "  ".join(
    f"{st.mean(v):>16.3f}" if len(v) >= 1 else f"{'-':>16}" for v in per_cond.values()))
print(f"{'stdev':>6}  " + "  ".join(
    f"{st.stdev(v):>16.3f}" if len(v) >= 2 else f"{'-':>16}" for v in per_cond.values()))

# Paired-t for no_pack vs full_attrforge and no_pack_vs vs full_attrforge
def paired_t(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [bi - ai for ai, bi in zip(a, b)]
    n = len(diffs)
    return {
        "n": n,
        "mean_diff": st.mean(diffs),
        "sd": st.stdev(diffs),
        "t": st.mean(diffs) / (st.stdev(diffs) / (n ** 0.5)) if st.stdev(diffs) > 0 else float("inf"),
    }

fa = per_cond["full_attrforge"]
np_ = per_cond["no_pack"]
npv = per_cond["no_pack_vs"]

print()
for label, b in [("no_pack vs SynSmith", np_), ("no_pack_vs vs SynSmith", npv)]:
    res = paired_t(fa, b)
    if res:
        print(f"{label:>26}: n={res['n']}, mean_diff={res['mean_diff']:+.4f}, "
              f"sd={res['sd']:.4f}, t={res['t']:+.3f}")
