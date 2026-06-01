"""Data-augmentation downstream evaluation: train on real + synthetic, test on held-out real.

The default train-on-synthetic / test-on-real protocol gives an unfair handicap to
diversity-inducing methods: the synthetic data replaces the small real anchor that
a low-capacity classifier needs. The realistic deployment of LLM-generated synthetic
data is data augmentation: concatenate the small real set with synthetic samples
and train on the union. This script implements that protocol.

Outputs:
    experiments/<base>_aggregated/augmentation.csv
    experiments/<base>_aggregated/augmentation.json
    paper/figures/<base>_augmentation.png
"""
from __future__ import annotations

import argparse
import csv
import json
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
import attrforge  # noqa: E402  (loads .env)
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_synth(condition_dir: Path) -> list[SyntheticSample]:
    out: list[SyntheticSample] = []
    for iter_dir in sorted(condition_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def fit_eval(X_train, y_train, X_test, y_test, labels) -> dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=17
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    per_class_f1 = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            per_class_f1[lbl] = 0.0
            continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        per_class_f1[lbl] = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "accuracy": float(np.mean(y_pred == y_test)),
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "per_class_f1": per_class_f1,
    }


def featurize_tfidf(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def featurize_st(train_texts, test_texts, enc):
    return (
        enc.encode(train_texts, normalize_embeddings=True, show_progress_bar=False),
        enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    real_train_path = REPO / "experiments" / "_splits" / "real_train.jsonl"
    real_test_path = REPO / "experiments" / "_splits" / "real_test.jsonl"
    real_train = [RealExample.model_validate(r) for r in load_jsonl(real_train_path)]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(real_test_path)]

    # Real-only baseline: train on the 30 real-train samples, test on the 10 held-out.
    real_texts = [r.text for r in real_train]
    real_labels = np.array([r.label for r in real_train])
    test_texts = [r.text for r in real_test]
    test_labels = np.array([r.label for r in real_test])
    labels = sorted(set(test_labels.tolist()))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")

    real_only_tfidf = []
    real_only_st = []
    # Real-only depends only on real-train (no seed variation). Compute once.
    X_tr, X_te = featurize_tfidf(real_texts, test_texts)
    r = fit_eval(X_tr, real_labels, X_te, test_labels, labels)
    real_only_tfidf = [r["macro_f1"]]
    X_tr, X_te = featurize_st(real_texts, test_texts, enc)
    r = fit_eval(X_tr, real_labels, X_te, test_labels, labels)
    real_only_st = [r["macro_f1"]]
    print(f"Real-only baseline (30 train + 10 test): "
          f"TF-IDF F1 = {real_only_tfidf[0]:.3f}, ST F1 = {real_only_st[0]:.3f}\n")

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    bag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for sd in seed_dirs:
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples = load_synth(cond_dir)
            if not samples:
                continue
            synth_texts = [s.text for s in samples]
            synth_labels = np.array([s.requested_attributes.get("intent", "?") for s in samples])

            # Concatenate real + synthetic
            aug_texts = real_texts + synth_texts
            aug_labels = np.concatenate([real_labels, synth_labels])

            # TF-IDF
            X_tr, X_te = featurize_tfidf(aug_texts, test_texts)
            r = fit_eval(X_tr, aug_labels, X_te, test_labels, labels)
            bag[cond_dir.name]["aug_tfidf_f1"].append(r["macro_f1"])

            # Sentence-transformer
            X_tr, X_te = featurize_st(aug_texts, test_texts, enc)
            r = fit_eval(X_tr, aug_labels, X_te, test_labels, labels)
            bag[cond_dir.name]["aug_st_f1"].append(r["macro_f1"])

    conds_order = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    out_rows = []
    print(f'{"condition":<18} {"aug TF-IDF":<20} {"aug ST":<20}')
    print(f'{"real-only":<18} {real_only_tfidf[0]:<.3f}                {real_only_st[0]:<.3f}')
    for cond in conds_order:
        if cond not in bag:
            continue
        d = bag[cond]
        def stats(k):
            v = d.get(k, [])
            if not v:
                return None, None
            return statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0
        t = stats("aug_tfidf_f1")
        s = stats("aug_st_f1")
        def fmt(t):
            if t == (None, None):
                return "n/a"
            return f"{t[0]:.3f}±{t[1]:.3f}"
        print(f'{cond:<18} {fmt(t):<20} {fmt(s):<20}')
        out_rows.append({
            "condition": cond,
            "aug_tfidf_f1_mean": t[0], "aug_tfidf_f1_sd": t[1],
            "aug_st_f1_mean": s[0], "aug_st_f1_sd": s[1],
            "n_seeds": len(d["aug_tfidf_f1"]),
        })

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "real_only_tfidf_f1": real_only_tfidf[0],
        "real_only_st_f1": real_only_st[0],
        "augmented": out_rows,
    }
    (out_dir / "augmentation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "augmentation.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Paired t-test full_classic vs full_attrforge under augmentation
    fc_tfidf = bag["full_classic"]["aug_tfidf_f1"]
    fa_tfidf = bag["full_attrforge"]["aug_tfidf_f1"]
    fc_st = bag["full_classic"]["aug_st_f1"]
    fa_st = bag["full_attrforge"]["aug_st_f1"]
    print(f"\nfull_classic aug TF-IDF: {[round(x,3) for x in fc_tfidf]}")
    print(f"full_attrforge aug TF-IDF: {[round(x,3) for x in fa_tfidf]}")
    diffs_tfidf = [a - b for a, b in zip(fa_tfidf, fc_tfidf)]
    diffs_st = [a - b for a, b in zip(fa_st, fc_st)]
    print(f"paired diffs (attrforge - classic) TF-IDF: {[round(d,3) for d in diffs_tfidf]}")
    print(f"paired diffs (attrforge - classic) ST:     {[round(d,3) for d in diffs_st]}")
    try:
        from scipy import stats as st
        t, p = st.ttest_rel(fa_tfidf, fc_tfidf)
        w, wp = st.wilcoxon(fa_tfidf, fc_tfidf)
        print(f"TF-IDF augmentation paired-t: t={t:.3f}, p={p:.4f}; Wilcoxon: W={w:.1f}, p={wp:.4f}")
        t, p = st.ttest_rel(fa_st, fc_st)
        w, wp = st.wilcoxon(fa_st, fc_st)
        print(f"ST augmentation paired-t:     t={t:.3f}, p={p:.4f}; Wilcoxon: W={w:.1f}, p={wp:.4f}")
    except Exception as e:
        print(f"scipy: {e}")

    # Plot
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    xs = list(range(len(out_rows) + 1))
    width = 0.4
    cats = ["real-only"] + [r["condition"] for r in out_rows]
    tf_means = [real_only_tfidf[0]] + [r["aug_tfidf_f1_mean"] for r in out_rows]
    tf_sds   = [0.0]               + [r["aug_tfidf_f1_sd"]   for r in out_rows]
    st_means = [real_only_st[0]]   + [r["aug_st_f1_mean"]    for r in out_rows]
    st_sds   = [0.0]               + [r["aug_st_f1_sd"]      for r in out_rows]
    ax.bar([x - width/2 for x in xs], tf_means, width, yerr=tf_sds, capsize=3, color="#3a6ea5", label="TF-IDF + LR")
    ax.bar([x + width/2 for x in xs], st_means, width, yerr=st_sds, capsize=3, color="#2e715a", label="Sentence-transformer + LR")
    ax.axhline(real_only_tfidf[0], color="#3a6ea5", linestyle=":", alpha=0.5)
    ax.axhline(real_only_st[0], color="#2e715a", linestyle=":", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("downstream macro F1 (held-out real test)")
    ax.set_title("Data-augmentation downstream F1: train on 30 real + N synthetic, test on 10 real (mean ± std, 5 seeds)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.base}_augmentation.png", dpi=160)
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_augmentation.png", dpi=160)
    print(f"\nSaved figure: {fig_dir}/{args.base}_augmentation.png")


if __name__ == "__main__":
    main()
