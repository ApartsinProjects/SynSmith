"""Compute per-seed multi-classifier F1 to enable paired statistical tests.

Reviewer round 3: verify the sentence-transformer claim
   full_attrforge 0.71 vs full_classic 0.67 (gap 0.04)
is statistically meaningful or within noise at N=3 seeds.
"""
from __future__ import annotations
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402  (loads .env)
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_synth_for_condition(condition_dir):
    out = []
    for iter_dir in sorted(condition_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def fit_eval(X_train, y_train, X_test, y_test, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(np.mean(y_pred == y_test))
    f1s = []
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            p = tp / (tp + fp); r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), acc


def featurize_tfidf(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def featurize_char(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def featurize_st(train_texts, test_texts):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    return (
        enc.encode(train_texts, normalize_embeddings=True),
        enc.encode(test_texts, normalize_embeddings=True),
    )


def main():
    test_path = REPO / "experiments" / "_splits" / "real_test.jsonl"
    test = [RealExample.model_validate(r) for r in load_jsonl(test_path)]
    test_texts = [t.text for t in test]
    test_labels = np.array([t.label for t in test])
    labels = sorted(set(test_labels.tolist()))

    seed_dirs = sorted((REPO / "experiments").glob("main_run_002_seed*"))
    # condition -> classifier -> seed_idx -> macro_f1
    per_seed = defaultdict(lambda: defaultdict(dict))

    for sd in seed_dirs:
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples = load_synth_for_condition(cond_dir)
            if not samples:
                continue
            train_texts = [s.text for s in samples]
            train_labels = np.array([s.requested_attributes.get("intent", "?") for s in samples])

            for fname, featfn in [
                ("tfidf_word", featurize_tfidf),
                ("tfidf_char_3_5", featurize_char),
                ("st_minilm", featurize_st),
            ]:
                try:
                    X_train, X_test = featfn(train_texts, test_texts)
                    f1, _ = fit_eval(X_train, train_labels, X_test, test_labels, labels)
                    per_seed[cond_dir.name][fname][sd.name] = f1
                except Exception as e:
                    print(f"  {sd.name}/{cond_dir.name}/{fname} FAILED: {e}")

    # write out
    out = {}
    for cond, by_clf in per_seed.items():
        out[cond] = {}
        for clf, by_seed in by_clf.items():
            out[cond][clf] = by_seed
    with (REPO / "paper" / "_per_seed_f1.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote _per_seed_f1.json")
    # print compact
    print("\nCondition x Classifier x Seed:")
    for cond, by_clf in out.items():
        for clf, by_seed in by_clf.items():
            vals = list(by_seed.values())
            if len(vals) == 3:
                s = statistics.stdev(vals)
                print(f"  {cond:18s} {clf:18s} seeds={[round(v,3) for v in vals]} mean={statistics.mean(vals):.3f} sd={s:.3f}")


if __name__ == "__main__":
    main()
