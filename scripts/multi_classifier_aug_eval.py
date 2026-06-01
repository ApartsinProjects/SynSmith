"""Multi-classifier augmentation evaluation.

Reviewer MA-NEW-5 (round-4): the augmentation experiment is single-classifier
(sentence-transformer + LR). Section 7.2/7.3 of the paper demonstrates that the
diversity-utility tradeoff is classifier-dependent in the isolated train-on-
synthetic protocol. This script extends the augmentation evaluation to the same
three downstream classifiers used in Section 7.3:

  - TF-IDF word unigrams+bigrams + LR
  - TF-IDF char n-grams of size 3-5 + LR
  - Sentence-transformer (MiniLM-L6-v2) embeddings + LR

For each combination of (real_train_size n, condition, classifier), we train on
the union of a stratified real subsample of size n and the condition's full
synthetic batch (48 samples for iterated conditions, 16 for naive/few_shot),
then test on the held-out 10-item real test set. Aggregated across 5 random
seeds.

This addresses whether the augmentation headline is classifier-dependent.
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


def featurize_tfidf_word(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def featurize_tfidf_char(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30])
    args = ap.parse_args()

    real_all = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")]
    y_test = np.array([r.label for r in real_test])
    labels = sorted(set(y_test.tolist()))
    test_texts = [r.text for r in real_test]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_test_st = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # Results: classifier -> condition -> size -> [f1 per seed]
    bag = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    real_only = defaultdict(lambda: defaultdict(list))

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        # Real-only baselines at each size, each classifier
        for n in args.sizes:
            sub = stratified_subsample(real_all, n, seed=seed)
            sub_texts = [r.text for r in sub]
            sub_labels = np.array([r.label for r in sub])
            # tfidf_word
            Xtr, Xte = featurize_tfidf_word(sub_texts, test_texts)
            real_only["tfidf_word"][n].append(fit_eval_f1(Xtr, sub_labels, Xte, y_test, labels))
            # tfidf_char
            Xtr, Xte = featurize_tfidf_char(sub_texts, test_texts)
            real_only["tfidf_char"][n].append(fit_eval_f1(Xtr, sub_labels, Xte, y_test, labels))
            # st
            Xtr_st = enc.encode(sub_texts, normalize_embeddings=True, show_progress_bar=False)
            real_only["st"][n].append(fit_eval_f1(Xtr_st, sub_labels, X_test_st, y_test, labels))

        # Augmentation per condition, each classifier
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth: continue
            synth_texts = [s.text for s in synth]
            synth_labels = np.array([s.requested_attributes.get("intent", "?") for s in synth])

            for n in args.sizes:
                sub = stratified_subsample(real_all, n, seed=seed)
                sub_texts = [r.text for r in sub]
                sub_labels = np.array([r.label for r in sub])
                aug_texts = sub_texts + synth_texts
                aug_labels = np.concatenate([sub_labels, synth_labels])

                # tfidf_word
                Xtr, Xte = featurize_tfidf_word(aug_texts, test_texts)
                bag["tfidf_word"][cond_dir.name][n].append(fit_eval_f1(Xtr, aug_labels, Xte, y_test, labels))
                # tfidf_char
                Xtr, Xte = featurize_tfidf_char(aug_texts, test_texts)
                bag["tfidf_char"][cond_dir.name][n].append(fit_eval_f1(Xtr, aug_labels, Xte, y_test, labels))
                # st
                Xtr_st = np.concatenate([
                    enc.encode(sub_texts, normalize_embeddings=True, show_progress_bar=False),
                    enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False),
                ], axis=0)
                bag["st"][cond_dir.name][n].append(fit_eval_f1(Xtr_st, aug_labels, X_test_st, y_test, labels))

    conds = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    classifiers = ["tfidf_word", "tfidf_char", "st"]
    cls_names = {"tfidf_word": "TF-IDF word", "tfidf_char": "TF-IDF char 3-5", "st": "Sent-trans MiniLM"}

    for cls in classifiers:
        print(f"\n========== {cls_names[cls]} ==========")
        print(f"{'size':<6} {'real-only':<14}", end="")
        for c in conds:
            print(f" {c:<16}", end="")
        print()
        for n in args.sizes:
            r = real_only[cls][n]
            r_m = statistics.mean(r); r_s = statistics.stdev(r) if len(r) > 1 else 0
            print(f"{n:<6} {r_m:.3f}±{r_s:.3f}  ", end="")
            for c in conds:
                v = bag[cls][c][n]
                if not v:
                    print(f" {'n/a':<16}", end=""); continue
                m = statistics.mean(v); sd = statistics.stdev(v) if len(v) > 1 else 0
                print(f" {m:.3f}±{sd:.3f}    ", end="")
            print()

        # Paired stats: full_attrforge vs full_classic
        print(f"\n  Paired stats full_attrforge - full_classic ({cls_names[cls]}):")
        try:
            from scipy import stats as st
            for n in args.sizes:
                fc = bag[cls]["full_classic"][n]
                fa = bag[cls]["full_attrforge"][n]
                if not fc or not fa: continue
                diffs = [a - b for a, b in zip(fa, fc)]
                t, p = st.ttest_rel(fa, fc)
                print(f"    n={n}: mean diff = {statistics.mean(diffs):+.3f} ± {statistics.stdev(diffs):.3f}; paired-t p={p:.3f}; per-seed: {[round(d,3) for d in diffs]}")
        except Exception as e:
            print(f"    scipy: {e}")

    # Save
    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sizes": args.sizes,
        "classifiers": classifiers,
        "real_only": {cls: {n: real_only[cls][n] for n in args.sizes} for cls in classifiers},
        "augmented": {cls: {c: {n: bag[cls][c][n] for n in args.sizes} for c in conds if c in bag[cls]} for cls in classifiers},
    }
    (out_dir / "scarce_real_multi_classifier.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 3-panel figure: F1 vs n for each classifier
    fig_dir = REPO / "paper" / "figures"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, cls in zip(axes, classifiers):
        sizes_arr = list(args.sizes)
        rm = [statistics.mean(real_only[cls][n]) for n in sizes_arr]
        rs = [statistics.stdev(real_only[cls][n]) if len(real_only[cls][n]) > 1 else 0 for n in sizes_arr]
        ax.errorbar(sizes_arr, rm, yerr=rs, marker="s", linewidth=2, capsize=4, color="#444444", linestyle="--", label="real-only")
        for c, col, label in [("full_classic", "#3a6ea5", "3-critic loop"), ("full_attrforge", "#c0392b", "7-critic loop (AttrForge)")]:
            m = [statistics.mean(bag[cls][c][n]) if bag[cls][c][n] else 0 for n in sizes_arr]
            s = [statistics.stdev(bag[cls][c][n]) if bag[cls][c][n] and len(bag[cls][c][n]) > 1 else 0 for n in sizes_arr]
            ax.errorbar(sizes_arr, m, yerr=s, marker="o", linewidth=2, capsize=4, color=col, label=label)
        ax.set_title(cls_names[cls])
        ax.set_xlabel("number of real training examples")
        ax.set_xticks(sizes_arr)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("downstream macro F1")
    fig.suptitle(f"Augmentation macro F1 vs real-train size, across 3 downstream classifiers (mean ± std, 5 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.base}_scarce_real_multi_classifier.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_scarce_real_multi_classifier.png", dpi=160, bbox_inches="tight")
    print(f"\nSaved figure: {fig_dir}/{args.base}_scarce_real_multi_classifier.png")


if __name__ == "__main__":
    main()
