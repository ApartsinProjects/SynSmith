"""
Render the cross-task relative-ratio bracket figure for the SynSmith paper.

Data source: Table 12 in docs/index.html (synth-only cross-task headline).
- SST-2: ratio 1.04, 95% CI [1.00, 1.07], real/class=30
- Banking77: ratio 0.92, 95% CI [0.91, 0.93], real/class=30
- TREC: ratio 1.00, 95% CI [0.92, 1.08], real/class=10

Output: docs/figures/cross_task_ratio.png (and .pdf for reference).
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA = [
    ("SST-2",      1.04, 1.00, 1.07, 30, 2),
    ("Banking77",  0.92, 0.91, 0.93, 30, 10),
    ("TREC",       1.00, 0.92, 1.08, 10, 6),
]

fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=200)

ys = list(range(len(DATA)))[::-1]
labels = [f"{d[0]}\n($K={d[5]}$, real/cls=${d[4]}$)" for d in DATA]

for y, (name, ratio, lo, hi, n_real, n_cls) in zip(ys, DATA):
    # bracket = horizontal line with end caps
    color = "#14385c"
    ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="butt", zorder=3)
    ax.plot([lo, lo], [y - 0.18, y + 0.18], color=color, lw=2.5, zorder=3)
    ax.plot([hi, hi], [y - 0.18, y + 0.18], color=color, lw=2.5, zorder=3)
    # ratio mean = solid circle
    ax.scatter([ratio], [y], s=46, color=color, zorder=4, edgecolor="white", linewidths=0.7)
    # value annotation: ratio + bracket
    ax.text(hi + 0.012, y, f"{ratio:.2f}  [{lo:.2f}, {hi:.2f}]",
            va="center", ha="left", fontsize=9.5, color="#111418")

# reference line at 1.0 (real-only)
ax.axvline(1.0, color="#888c93", linestyle="--", lw=1.0, zorder=1)
ax.text(1.0, len(DATA) - 0.45, "real-only", color="#5a626c", fontsize=8.5,
        rotation=0, ha="center", va="bottom")

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlim(0.86, 1.18)
ax.set_ylim(-0.6, len(DATA) - 0.3)
ax.set_xlabel(r"Relative ratio  $F1_{\mathrm{SynSmith}} / F1_{\mathrm{real\text{-}only}}$",
              fontsize=10.5)
ax.tick_params(axis="x", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#d1d4d8")
ax.spines["bottom"].set_color("#d1d4d8")
ax.grid(axis="x", color="#e8eaed", lw=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
out_png = OUT_DIR / "cross_task_ratio.png"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
out_pdf = OUT_DIR / "cross_task_ratio.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Wrote {out_png} ({out_png.stat().st_size} bytes)")
print(f"Wrote {out_pdf} ({out_pdf.stat().st_size} bytes)")
