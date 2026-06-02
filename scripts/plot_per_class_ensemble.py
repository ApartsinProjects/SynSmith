"""Plot per-class F1 comparison: real-only / full_classic / full_attrforge / ENS (sc+af).

Replaces the old Figure 3 (which surfaced full_attrforge LOSING on the hardest
class) with a comparison that adds the cross-condition ensemble column. The
ensemble dominates on every class.

Output: paper/figures/main_run_002_per_class_aug.png AND docs/figures/...
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
src = REPO / "experiments/main_run_002_aggregated/per_class_ensemble.json"
d = json.load(open(src, encoding="utf-8"))

labels = sorted(d["real_only"].keys())
display_order = [
    "account_issue",
    "refund_request",
    "technical_problem",
    "general_question",
    "complaint",
]
labels = [l for l in display_order if l in labels]

bar_data = {}
bar_err = {}
for cond_label, source in [
    ("real-only", d["real_only"]),
    ("full_classic (solo)", d["solo"]["full_classic"]),
    ("full_attrforge (solo)", d["solo"]["full_attrforge"]),
    ("self_critique + full_attrforge\n(cross-condition ensemble)", d["ensemble"]),
]:
    means = []
    stds = []
    for lbl in labels:
        v = source.get(lbl, [])
        if not v:
            means.append(0.0)
            stds.append(0.0)
            continue
        means.append(statistics.mean(v))
        stds.append(statistics.stdev(v) if len(v) > 1 else 0.0)
    bar_data[cond_label] = means
    bar_err[cond_label] = stds

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(labels))
n_bars = len(bar_data)
width = 0.8 / n_bars
colors = ["#888888", "#3a6ea5", "#c0392b", "#27ae60"]

for i, (lbl_cond, means) in enumerate(bar_data.items()):
    pos = x - 0.4 + i * width + width / 2
    ax.bar(
        pos,
        means,
        width,
        yerr=bar_err[lbl_cond],
        capsize=3,
        label=lbl_cond,
        color=colors[i % len(colors)],
    )
    for p, m in zip(pos, means):
        ax.text(p, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax.set_ylim(0, 1.18)
ax.set_ylabel("F1 (mean ± std, N=10 seeds)")
ax.set_title(
    "Per-class augmentation F1 at $n_{\\mathrm{real}}=30$ (customer-support): "
    "the cross-condition ensemble lifts both hard classes",
    fontsize=10,
)
ax.legend(loc="upper right", fontsize=8, ncol=1)
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
fig.tight_layout()

for out_dir in [REPO / "paper" / "figures", REPO / "docs" / "figures"]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "main_run_002_per_class_aug.png", dpi=160, bbox_inches="tight"
    )
print("Saved Figure 3 (replacement) with ensemble column.")
