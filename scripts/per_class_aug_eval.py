"""Per-class augmentation analysis.

Disaggregate augmentation F1 by class to look for class-specific wins.
Hypothesis: AttrForge's surface-diversity gains might help on the hardest
class (general_question) more than on easier classes with strong keyword
signatures.

For each real-train size and each condition, compute per-class F1 averaged
across 5 seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def stratified_subsample(reals, n, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in reals:
        by_label[r.label].append(r)
    labels = sorted(by_label.keys())
    per_class = max(1, n // len(labels))
    out = []
    for lbl in labels:
        items = list(by_label[lbl])
        rng.shuffle(items)
        out.extend(items[:per_class])
    rng.shuffle(out)
    return out[:n]


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def per_class_f1(X_train, y_train, X_test, y_test, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    out = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            out[lbl] = 0.0; continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        out[lbl] = 2 * p * r / (p + r) if (p + r) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20, 30])
    args = ap.parse_args()

    from scripts._splits_resolver import resolve_splits
    _real_train_path, _real_test_path = resolve_splits(args.base)
    real_all = [RealExample.model_validate(r) for r in load_jsonl(_real_train_path)]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(_real_test_path)]
    y_test = np.array([r.label for r in real_test])
    labels = sorted(set(y_test.tolist()))
    test_texts = [r.text for r in real_test]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_test_st = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # bag[condition][size][label] -> list[per-seed F1]
    bag = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    real_only = defaultdict(lambda: defaultdict(list))

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for n in args.sizes:
            sub = stratified_subsample(real_all, n, seed=seed)
            sub_texts = [r.text for r in sub]
            sub_labels = np.array([r.label for r in sub])
            X_tr = enc.encode(sub_texts, normalize_embeddings=True, show_progress_bar=False)
            f1s = per_class_f1(X_tr, sub_labels, X_test_st, y_test, labels)
            for lbl in labels:
                real_only[n][lbl].append(f1s[lbl])

        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth: continue
            synth_texts = [s.text for s in synth]
            synth_labels = np.array([s.requested_attributes.get("intent", "?") for s in synth])
            X_synth = enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False)

            for n in args.sizes:
                sub = stratified_subsample(real_all, n, seed=seed)
                X_sub = enc.encode([r.text for r in sub], normalize_embeddings=True, show_progress_bar=False)
                X_tr = np.concatenate([X_sub, X_synth], axis=0)
                y_tr = np.concatenate([np.array([r.label for r in sub]), synth_labels])
                f1s = per_class_f1(X_tr, y_tr, X_test_st, y_test, labels)
                for lbl in labels:
                    bag[cond_dir.name][n][lbl].append(f1s[lbl])

    # Headline display: per-class at n=30
    n_focus = 30 if 30 in args.sizes else args.sizes[-1]
    print(f"\nPer-class augmentation F1 at n_real={n_focus} (mean ± std, 5 seeds):\n")
    print(f"{'condition':<18}", end="")
    for lbl in labels:
        print(f" {lbl:<22}", end="")
    print()
    print(f"{'real-only':<18}", end="")
    for lbl in labels:
        vals = real_only[n_focus][lbl]
        m = statistics.mean(vals); s = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f" {m:.2f}±{s:.2f}              ", end="")
    print()
    for cond in ["full_classic", "full_attrforge"]:
        print(f"{cond:<18}", end="")
        for lbl in labels:
            vals = bag[cond][n_focus][lbl]
            m = statistics.mean(vals); s = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f" {m:.2f}±{s:.2f}              ", end="")
        print()

    # Find class where AttrForge wins largest over full_classic
    print(f"\nMean per-class augmentation F1 at n=30 (full_attrforge - full_classic):")
    for lbl in labels:
        a = bag["full_attrforge"][n_focus][lbl]
        c = bag["full_classic"][n_focus][lbl]
        if a and c:
            diffs = [aa - cc for aa, cc in zip(a, c)]
            md = statistics.mean(diffs)
            sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
            print(f"  {lbl:<20} diff = {md:+.3f} ± {sd:.3f}")

    # Save
    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sizes": args.sizes,
        "labels": labels,
        "real_only": {n: {lbl: real_only[n][lbl] for lbl in labels} for n in args.sizes},
        "augmented": {c: {n: {lbl: bag[c][n][lbl] for lbl in labels} for n in args.sizes} for c in bag},
    }
    (out_dir / "per_class_aug.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plot: per-class F1 at n=30 for real-only / full_classic / full_attrforge
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = list(range(len(labels)))
    width = 0.25
    real_means = [statistics.mean(real_only[n_focus][lbl]) for lbl in labels]
    real_sds = [statistics.stdev(real_only[n_focus][lbl]) if len(real_only[n_focus][lbl]) > 1 else 0 for lbl in labels]
    fc_means = [statistics.mean(bag["full_classic"][n_focus][lbl]) if bag["full_classic"][n_focus][lbl] else 0 for lbl in labels]
    fc_sds = [statistics.stdev(bag["full_classic"][n_focus][lbl]) if len(bag["full_classic"][n_focus][lbl]) > 1 else 0 for lbl in labels]
    fa_means = [statistics.mean(bag["full_attrforge"][n_focus][lbl]) if bag["full_attrforge"][n_focus][lbl] else 0 for lbl in labels]
    fa_sds = [statistics.stdev(bag["full_attrforge"][n_focus][lbl]) if len(bag["full_attrforge"][n_focus][lbl]) > 1 else 0 for lbl in labels]
    ax.bar([x - width for x in xs], real_means, width, yerr=real_sds, capsize=3, color="#444444", label="real-only")
    ax.bar(xs, fc_means, width, yerr=fc_sds, capsize=3, color="#3a6ea5", label="full_classic aug")
    ax.bar([x + width for x in xs], fa_means, width, yerr=fa_sds, capsize=3, color="#c0392b", label="full_attrforge aug")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("per-class F1")
    ax.set_title(f"Per-class augmentation F1 at n_real={n_focus} (sentence-transformer + LR; mean ± std, 5 seeds)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(REPO / "paper" / "figures" / f"{args.base}_per_class_aug.png", dpi=160)
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_per_class_aug.png", dpi=160)
    print(f"\nSaved figure: paper/figures/{args.base}_per_class_aug.png")


if __name__ == "__main__":
    main()
