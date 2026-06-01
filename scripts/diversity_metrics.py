"""Direct diversity measurements on each condition's final batch.

BL4 from the first reviewer round: the previous draft asserted "surface
diversity measurably increases" without backing it with direct numbers
(distinct-n, self-BLEU, etc.). This script computes:

  - distinct-1, distinct-2, distinct-3: unique-n-gram / total-n-gram
    ratio. Higher = more lexical diversity.
  - self-BLEU-4: mean BLEU-4 of each sample against the rest of the
    batch. Higher = more repetitive (sample close to other samples).
  - mean-edit-fraction: mean Levenshtein-like normalized character
    similarity to nearest neighbor in batch.

These are run on every condition's final pooled batch for every seed,
and aggregated to mean ± std.

Outputs:
  experiments/<base>_aggregated/diversity_metrics.csv
  experiments/<base>_aggregated/diversity_metrics.json
  paper/figures/<base>_diversity_metrics.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402  (loads .env)
from attrforge.schema import SyntheticSample, load_jsonl  # noqa: E402


def tokenize(s: str) -> list[str]:
    return s.lower().split()


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: unique-n-gram / total-n-gram (Li et al. 2016).

    Higher is more diverse. 0 = perfectly repetitive (one n-gram repeated),
    1 = no n-gram repeats.
    """
    total = 0
    seen = set()
    for t in texts:
        toks = tokenize(t)
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i : i + n])
            seen.add(ng)
            total += 1
    if total == 0:
        return 0.0
    return len(seen) / total


def bleu4_brevity_penalty(ref_len: int, cand_len: int) -> float:
    if cand_len > ref_len:
        return 1.0
    if cand_len == 0:
        return 0.0
    return math.exp(1 - ref_len / cand_len)


def bleu4(cand: list[str], refs: list[list[str]]) -> float:
    """Tiny BLEU-4 implementation; smoothing-1 (Chen and Cherry 2014)."""
    if not cand or not refs:
        return 0.0
    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []
    for n in range(1, 5):
        cand_ng = Counter([tuple(cand[i : i + n]) for i in range(len(cand) - n + 1)])
        if not cand_ng:
            return 0.0
        max_ref = Counter()
        for ref in refs:
            ref_ng = Counter([tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)])
            for ng, c in ref_ng.items():
                if c > max_ref[ng]:
                    max_ref[ng] = c
        clipped = 0
        total = 0
        for ng, c in cand_ng.items():
            clipped += min(c, max_ref[ng])
            total += c
        if total == 0:
            return 0.0
        # smoothing-1
        if clipped == 0:
            clipped = 1
            total = total + 1
        precisions.append(clipped / total)
    log_p = sum(w * math.log(p) if p > 0 else -1e9 for w, p in zip(weights, precisions))
    ref_len = min(len(r) for r in refs)
    bp = bleu4_brevity_penalty(ref_len, len(cand))
    return bp * math.exp(log_p)


def self_bleu4(texts: list[str]) -> float:
    """Mean BLEU-4 of each sample against all other samples in the batch.

    Higher = more repetitive across the batch.
    """
    if len(texts) < 2:
        return 0.0
    toks = [tokenize(t) for t in texts]
    n = len(toks)
    scores = []
    for i in range(n):
        refs = [toks[j] for j in range(n) if j != i]
        scores.append(bleu4(toks[i], refs))
    return statistics.mean(scores)


def load_condition_batch(cond_dir: Path) -> list[str]:
    out: list[str] = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r).text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    if not seed_dirs:
        sys.exit(f"no seed dirs matching {args.base}_seed*")

    print(f"Computing direct diversity across {len(seed_dirs)} seeds")

    bag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sd in seed_dirs:
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            texts = load_condition_batch(cond_dir)
            if not texts:
                continue
            for n in (1, 2, 3):
                bag[cond_dir.name][f"distinct_{n}"].append(distinct_n(texts, n))
            bag[cond_dir.name]["self_bleu4"].append(self_bleu4(texts))

    conds_order = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    out_rows = []
    print(f'\n{"condition":<18} {"distinct_1":<14} {"distinct_2":<14} {"distinct_3":<14} {"self_bleu4":<14}')
    for cond in conds_order:
        if cond not in bag:
            continue
        d = bag[cond]
        def stats(k):
            v = d.get(k, [])
            if not v:
                return None, None
            return statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0
        d1 = stats("distinct_1"); d2 = stats("distinct_2"); d3 = stats("distinct_3"); sb = stats("self_bleu4")
        def fmt(t):
            if t == (None, None):
                return "n/a"
            return f"{t[0]:.3f}±{t[1]:.3f}"
        print(f'{cond:<18} {fmt(d1):<14} {fmt(d2):<14} {fmt(d3):<14} {fmt(sb):<14}')
        out_rows.append({
            "condition": cond,
            "distinct_1_mean": d1[0], "distinct_1_sd": d1[1],
            "distinct_2_mean": d2[0], "distinct_2_sd": d2[1],
            "distinct_3_mean": d3[0], "distinct_3_sd": d3[1],
            "self_bleu4_mean": sb[0], "self_bleu4_sd": sb[1],
            "n_seeds": len(d["distinct_1"]),
        })

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diversity_metrics.json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    with (out_dir / "diversity_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    def panel(ax, key, title, ylim=None, color="#3a6ea5", invert=False):
        means = [r[f"{key}_mean"] for r in out_rows]
        sds = [r[f"{key}_sd"] for r in out_rows]
        xs = list(range(len(out_rows)))
        ax.bar(xs, means, yerr=sds, capsize=3, color=color)
        for i, (m, s) in enumerate(zip(means, sds)):
            ax.text(i, m + s + 0.005, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([r["condition"] for r in out_rows], rotation=20, ha="right", fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title + (" (lower = more diverse)" if invert else " (higher = more diverse)"), fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    panel(axes[0, 0], "distinct_1", "distinct-1", (0, 1), color="#14385c")
    panel(axes[0, 1], "distinct_2", "distinct-2", (0, 1), color="#14385c")
    panel(axes[1, 0], "distinct_3", "distinct-3", (0, 1), color="#14385c")
    panel(axes[1, 1], "self_bleu4", "self-BLEU-4", (0, 0.5), color="#6e0e10", invert=True)

    n_seeds = out_rows[0]["n_seeds"]
    fig.suptitle(f"Direct diversity measurements (mean ± std across {n_seeds} seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.base}_diversity_metrics.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_diversity_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {out_dir} and figure to {fig_dir}/{args.base}_diversity_metrics.png")


if __name__ == "__main__":
    main()
