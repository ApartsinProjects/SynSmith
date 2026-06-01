"""Scarce-real augmentation: subsample real-train, augment with synthetic, test on real.

When real data is abundant (30 examples here), a strong classifier already scores
F1=0.89; synthetic data has little headroom. The realistic use of LLM-generated
synthetic data is the case where real data is SCARCE. We subsample the real-train
pool to sizes {5, 10, 20, 30} stratified across classes, augment with each
condition's synthetic batch, and measure downstream F1.

Hypothesis: as real-train shrinks, the diversity of the synthetic data matters
more, and AttrForge's adversaries should win at the smallest real-train sizes.

Outputs:
    experiments/<base>_aggregated/scarce_real.json
    paper/figures/<base>_scarce_real.png
"""
from __future__ import annotations

import argparse
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


def fit_eval_f1(X_train, y_train, X_test, y_test, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1s = []
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s))


def st_encode(texts, enc):
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20, 30])
    args = ap.parse_args()

    real_all = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")]
    y_test = np.array([r.label for r in real_test])
    labels = sorted(set(y_test.tolist()))
    test_texts = [r.text for r in real_test]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_test_st = st_encode(test_texts, enc)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # Results: condition -> size -> [f1 per seed]
    aug_st = defaultdict(lambda: defaultdict(list))
    real_only_st = defaultdict(list)

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        # Real-only baselines at each size
        for n in args.sizes:
            sub = stratified_subsample(real_all, n, seed=seed)
            sub_texts = [r.text for r in sub]
            sub_labels = np.array([r.label for r in sub])
            X_tr_st = st_encode(sub_texts, enc)
            real_only_st[n].append(fit_eval_f1(X_tr_st, sub_labels, X_test_st, y_test, labels))

        # Augmentation per condition
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth: continue
            synth_texts = [s.text for s in synth]
            synth_labels = np.array([s.requested_attributes.get("intent", "?") for s in synth])
            X_synth_st = st_encode(synth_texts, enc)

            for n in args.sizes:
                sub = stratified_subsample(real_all, n, seed=seed)
                sub_texts = [r.text for r in sub]
                sub_labels = np.array([r.label for r in sub])
                X_tr_st = np.concatenate([st_encode(sub_texts, enc), X_synth_st], axis=0)
                y_tr = np.concatenate([sub_labels, synth_labels])
                aug_st[cond_dir.name][n].append(fit_eval_f1(X_tr_st, y_tr, X_test_st, y_test, labels))

    conds = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    print(f"{'size':<6} {'real-only':<12}", end="")
    for c in conds:
        print(f" {c:<14}", end="")
    print()
    for n in args.sizes:
        r = real_only_st[n]
        r_m = statistics.mean(r); r_s = statistics.stdev(r) if len(r) > 1 else 0
        print(f"{n:<6} {r_m:.3f}±{r_s:.3f}", end="")
        for c in conds:
            v = aug_st[c][n]
            if not v:
                print(f" {'n/a':<14}", end=""); continue
            m = statistics.mean(v); sd = statistics.stdev(v) if len(v) > 1 else 0
            print(f" {m:.3f}±{sd:.3f}  ", end="")
        print()

    # Paired statistics: full_attrforge vs full_classic at each size
    print("\nPaired stats (full_attrforge - full_classic) on ST under augmentation:")
    try:
        from scipy import stats as st
        for n in args.sizes:
            fc = aug_st["full_classic"][n]
            fa = aug_st["full_attrforge"][n]
            if not fc or not fa: continue
            diffs = [a - b for a, b in zip(fa, fc)]
            t, p = st.ttest_rel(fa, fc)
            w, wp = st.wilcoxon(fa, fc, zero_method="zsplit")
            print(f"  size={n}: mean diff = {statistics.mean(diffs):+.3f} ± {(statistics.stdev(diffs) if len(diffs)>1 else 0):.3f}; paired-t p={p:.3f}; Wilcoxon p={wp:.3f}")
    except Exception as e:
        print(f"  scipy: {e}")

    # Save and plot
    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sizes": args.sizes,
        "real_only_st": {n: real_only_st[n] for n in args.sizes},
        "augmented_st": {c: {n: aug_st[c][n] for n in args.sizes} for c in conds if c in aug_st},
    }
    (out_dir / "scarce_real.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sizes_arr = list(args.sizes)
    real_mean = [statistics.mean(real_only_st[n]) for n in sizes_arr]
    real_sd = [statistics.stdev(real_only_st[n]) if len(real_only_st[n]) > 1 else 0 for n in sizes_arr]
    ax.errorbar(sizes_arr, real_mean, yerr=real_sd, marker="s", linewidth=2, capsize=4,
                color="#444444", label="real-only (no augmentation)", linestyle="--")
    colors = {"full_classic": "#3a6ea5", "full_attrforge": "#c0392b",
              "naive": "#999999", "few_shot": "#999999",
              "self_critique": "#bbbbbb", "realism_only": "#bbbbbb", "diversity_only": "#bbbbbb"}
    for c in ["full_classic", "full_attrforge"]:
        m = [statistics.mean(aug_st[c][n]) if aug_st[c][n] else 0 for n in sizes_arr]
        s = [statistics.stdev(aug_st[c][n]) if aug_st[c][n] and len(aug_st[c][n]) > 1 else 0 for n in sizes_arr]
        ax.errorbar(sizes_arr, m, yerr=s, marker="o", linewidth=2, capsize=4, color=colors[c], label=c)
    ax.set_xlabel("number of real training examples (stratified subsample)")
    ax.set_ylabel("downstream macro F1 (sentence-transformer + LR)")
    ax.set_title("Scarce-real augmentation: F1 vs real-train size (mean ± std, 5 seeds)")
    ax.set_xticks(sizes_arr)
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.base}_scarce_real.png", dpi=160)
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_scarce_real.png", dpi=160)
    print(f"\nSaved figure: {fig_dir}/{args.base}_scarce_real.png")


if __name__ == "__main__":
    main()
