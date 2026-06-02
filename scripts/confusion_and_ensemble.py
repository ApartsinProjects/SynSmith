"""Confusion matrix analysis and cross-condition classifier ensembling.

Two free-CPU analyses that probe the v2 paper's diversity hypothesis from
angles the v1 macro-F1 cannot see:

1. CONFUSION MATRIX: do AttrForge's synthetic samples reduce the specific
   pairwise class confusions that the v1 worst-class result identified?
   On v1's customer-support task we expect complaint <-> general_question
   confusion to dominate (frustrated user phrases complaints as questions).
   If AttrForge's diversity is real, the augmented classifier should
   confuse those two LESS than the augmented full_classic classifier.

2. CROSS-CONDITION ENSEMBLE: do classifiers trained on different
   conditions' synthetic batches learn DIFFERENT decision boundaries?
   If yes, an ensemble (logit average) over multiple conditions should
   beat the best single condition. If AttrForge's diversity is
   orthogonal to full_classic's keyword-anchoring, AttrForge + classic
   ensemble should beat either alone.

Outputs:
  experiments/<base>_aggregated/confusion_ensemble.json
  paper/figures/<base>_confusion.png
  paper/figures/<base>_ensemble.png
"""
from __future__ import annotations

import argparse
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
import attrforge  # noqa: E402
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def fit_eval_with_proba(X_tr, y_tr, X_te, y_te, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    # align columns to labels order
    col_idx = [list(clf.classes_).index(lbl) for lbl in labels]
    proba = proba[:, col_idx]
    return clf, proba


def confusion_matrix(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[labels.index(t)][labels.index(p)] += 1
    return cm


def macro_f1_and_worst(cm, labels):
    f1s = []
    for i in range(len(labels)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s)), float(min(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    real_train = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")]
    real_train_texts = [r.text for r in real_train]
    real_train_labels = np.array([r.label for r in real_train])
    test_texts = [r.text for r in real_test]
    test_labels = np.array([r.label for r in real_test])
    labels = sorted(set(test_labels.tolist()))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    # bag[condition] -> list per seed with confusion matrix + proba matrix
    bag = defaultdict(list)

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples = load_synth(cond_dir)
            if not samples:
                continue
            texts = [s.text for s in samples]
            slabels = np.array([s.requested_attributes.get("intent", "?") for s in samples])
            X_synth = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            y_tr = np.concatenate([real_train_labels, slabels])
            clf, proba = fit_eval_with_proba(X_tr, y_tr, X_test, test_labels, labels)
            y_pred = np.array([labels[i] for i in proba.argmax(axis=1)])
            cm = confusion_matrix(test_labels.tolist(), y_pred.tolist(), labels)
            macro, worst = macro_f1_and_worst(cm, labels)
            bag[cond_dir.name].append({
                "seed": seed,
                "cm": cm.tolist(),
                "proba": proba.tolist(),
                "macro_f1": macro,
                "worst_f1": worst,
            })

    # ===== 1. Confusion-matrix analysis =====
    print(f"\n=== Confusion analysis ===")
    print(f"Most-confused pair per condition (mean off-diagonal across 5 seeds):\n")
    n_labels = len(labels)
    pair_means_per_cond = {}
    for cond, rows in bag.items():
        cm_sum = np.zeros((n_labels, n_labels))
        for r in rows:
            cm_sum += np.array(r["cm"])
        cm_mean = cm_sum / len(rows)
        pair_means_per_cond[cond] = cm_mean
        # Find top off-diagonal cell
        cm_off = cm_mean.copy(); np.fill_diagonal(cm_off, 0)
        i_max, j_max = np.unravel_index(cm_off.argmax(), cm_off.shape)
        if cm_off[i_max, j_max] > 0:
            print(f"  {cond:<18} top confusion: {labels[i_max]} -> {labels[j_max]}  ({cm_off[i_max, j_max]:.1f} avg samples)")
        else:
            print(f"  {cond:<18} no off-diagonal confusion (perfect)")

    # specific confusion of interest: complaint <-> general_question
    print(f"\n=== complaint <-> general_question confusion (the v1 worst-class story) ===")
    try:
        i_c = labels.index("complaint")
        i_g = labels.index("general_question")
    except ValueError:
        print("  skip (labels not present)")
        i_c = i_g = None
    if i_c is not None:
        print(f"{'condition':<18} {'C->G':<10} {'G->C':<10} {'total cross-pair':<14}")
        for cond, rows in bag.items():
            cgs = [r["cm"][i_c][i_g] for r in rows]
            gcs = [r["cm"][i_g][i_c] for r in rows]
            totals = [c + g for c, g in zip(cgs, gcs)]
            def fmt(v): return f"{statistics.mean(v):.2f}+-{statistics.stdev(v) if len(v)>1 else 0:.2f}"
            print(f"  {cond:<18} {fmt(cgs):<10} {fmt(gcs):<10} {fmt(totals):<14}")

    # ===== 2. Cross-condition ensemble =====
    print(f"\n=== Cross-condition ensemble (logit average) ===")
    # For each seed, ensemble pairs of conditions and compare
    interesting_pairs = [
        ("full_classic", "full_attrforge"),
        ("full_classic", "realism_only"),
        ("full_classic", "diversity_only"),
        ("full_attrforge", "realism_only"),
        ("full_attrforge", "diversity_only"),
    ]
    ensemble_rows = []
    for ca, cb in interesting_pairs:
        rows_a = bag.get(ca, [])
        rows_b = bag.get(cb, [])
        if not rows_a or not rows_b:
            continue
        macros_solo_a, macros_solo_b, macros_ens = [], [], []
        worsts_solo_a, worsts_solo_b, worsts_ens = [], [], []
        for ra, rb in zip(rows_a, rows_b):
            p_a = np.array(ra["proba"]); p_b = np.array(rb["proba"])
            ens = (p_a + p_b) / 2.0
            y_pred = np.array([labels[i] for i in ens.argmax(axis=1)])
            cm = confusion_matrix(test_labels.tolist(), y_pred.tolist(), labels)
            macro_e, worst_e = macro_f1_and_worst(cm, labels)
            macros_solo_a.append(ra["macro_f1"]); macros_solo_b.append(rb["macro_f1"])
            macros_ens.append(macro_e)
            worsts_solo_a.append(ra["worst_f1"]); worsts_solo_b.append(rb["worst_f1"])
            worsts_ens.append(worst_e)
        ensemble_rows.append({
            "pair": (ca, cb),
            "solo_a_macro": statistics.mean(macros_solo_a),
            "solo_b_macro": statistics.mean(macros_solo_b),
            "ensemble_macro": statistics.mean(macros_ens),
            "ensemble_worst": statistics.mean(worsts_ens),
            "macros_solo_a": macros_solo_a,
            "macros_solo_b": macros_solo_b,
            "macros_ens": macros_ens,
            "worsts_solo_a": worsts_solo_a,
            "worsts_solo_b": worsts_solo_b,
            "worsts_ens": worsts_ens,
        })

    print(f"{'pair':<40} {'solo A':<14} {'solo B':<14} {'ensemble':<14} {'ens worst':<14}")
    for row in ensemble_rows:
        ca, cb = row["pair"]
        a = row["macros_solo_a"]; b = row["macros_solo_b"]; e = row["macros_ens"]
        w = row["worsts_ens"]
        def fmt(v): return f"{statistics.mean(v):.3f}+-{statistics.stdev(v):.3f}"
        print(f"  {(ca+' + '+cb):<40} {fmt(a):<14} {fmt(b):<14} {fmt(e):<14} {fmt(w):<14}")

    print(f"\n=== Paired stats: ensemble vs solo ===")
    try:
        from scipy import stats as st
        for row in ensemble_rows:
            ca, cb = row["pair"]
            for solo_label, solo_vals in [("A_solo", row["macros_solo_a"]),
                                          ("B_solo", row["macros_solo_b"])]:
                diffs = [e - s for e, s in zip(row["macros_ens"], solo_vals)]
                t, p = st.ttest_rel(row["macros_ens"], solo_vals)
                md = statistics.mean(diffs); sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
                print(f"  ({ca}+{cb}) ensemble - {solo_label}: macro diff={md:+.3f}+-{sd:.3f} paired-t p={p:.3f}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": labels,
        "confusion": {c: [r["cm"] for r in rows] for c, rows in bag.items()},
        "ensemble": ensemble_rows,
    }
    (out_dir / "confusion_ensemble.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/confusion_ensemble.json")


if __name__ == "__main__":
    main()
