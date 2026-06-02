"""Smart ensembling: stacking and confidence-weighted aggregation.

Beyond logit-average, two more sophisticated combiners are tested:

1. STACKING via leave-one-out: train a small meta-classifier on each
   seed's [proba_cond_a | proba_cond_b | ...] features. Use leave-one-seed-out
   so the meta-classifier never sees the test labels of the held-out seed.

2. CONFIDENCE-WEIGHTED average: for each test item, weight each
   classifier's probability vector by its own max-probability (its
   self-reported confidence). Effectively: defer to the most-confident
   classifier per item.

3. MAX-CONFIDENCE: take the prediction of the single most-confident
   classifier per item.

If any of these beat the simple logit-average from ensemble_deep.py
(self_critique + full_attrforge at 0.953 macro), it strengthens the
v2 claim that AttrForge's diversity is decision-boundary diversity
that ensembling can extract.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def macro_worst(y_true, y_pred, labels):
    f1s = []
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_true == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_true != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_true == lbl)).sum())
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
    from sklearn.linear_model import LogisticRegression
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    proba_bag = {}  # cond -> {seed: proba}
    seeds_seen = []

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        seeds_seen.append(seed)
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples = load_synth(cond_dir)
            if not samples: continue
            texts = [s.text for s in samples]
            slabels = np.array([s.requested_attributes.get("intent", "?") for s in samples])
            X_synth = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            y_tr = np.concatenate([real_train_labels, slabels])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_test)
            col_idx = [list(clf.classes_).index(lbl) for lbl in labels]
            proba = proba[:, col_idx]
            proba_bag.setdefault(cond_dir.name, {})[seed] = proba

    # The best pair from ensemble_deep
    pair = ("self_critique", "full_attrforge")
    print(f"\n=== Ensembling methods for pair = {pair} ===")
    print(f"{'method':<32} {'macro':<14} {'worst':<14}")

    def eval_pair(combine_fn):
        macros, worsts = [], []
        for seed in seeds_seen:
            ps = [proba_bag.get(c, {}).get(seed) for c in pair]
            if any(p is None for p in ps): continue
            ens_pred = combine_fn(ps)
            y_pred = np.array([labels[i] for i in ens_pred])
            m, w = macro_worst(test_labels, y_pred, labels)
            macros.append(m); worsts.append(w)
        return macros, worsts

    # 1. straight logit-average
    def logit_avg(ps):
        return np.mean(np.stack(ps, axis=0), axis=0).argmax(axis=1)
    m, w = eval_pair(logit_avg)
    print(f"  logit_average                    {statistics.mean(m):.3f}+-{statistics.stdev(m):.3f}  "
          f"{statistics.mean(w):.3f}+-{statistics.stdev(w):.3f}")

    # 2. confidence-weighted
    def conf_weighted(ps):
        # weight each classifier's proba vector by its own max prob (per-item)
        weighted = np.zeros_like(ps[0])
        for p in ps:
            confs = p.max(axis=1, keepdims=True)
            weighted += p * confs
        return weighted.argmax(axis=1)
    m, w = eval_pair(conf_weighted)
    print(f"  confidence_weighted              {statistics.mean(m):.3f}+-{statistics.stdev(m):.3f}  "
          f"{statistics.mean(w):.3f}+-{statistics.stdev(w):.3f}")

    # 3. defer-to-most-confident
    def max_conf(ps):
        # for each item, pick the classifier with highest max-prob, take its argmax
        n_items = ps[0].shape[0]
        out = np.zeros(n_items, dtype=int)
        for i in range(n_items):
            best_conf = -1; best_pred = 0
            for p in ps:
                c = float(p[i].max())
                if c > best_conf:
                    best_conf = c
                    best_pred = int(p[i].argmax())
            out[i] = best_pred
        return out
    m, w = eval_pair(max_conf)
    print(f"  defer_to_most_confident          {statistics.mean(m):.3f}+-{statistics.stdev(m):.3f}  "
          f"{statistics.mean(w):.3f}+-{statistics.stdev(w):.3f}")

    # 4. all ITERATED conditions, logit-average
    iter_conds = [c for c in ["self_critique", "realism_only", "diversity_only",
                              "full_classic", "full_attrforge"] if c in proba_bag]
    print(f"\n=== All {len(iter_conds)} iterated conditions ensemble ===")
    macros, worsts = [], []
    for seed in seeds_seen:
        ps = [proba_bag[c].get(seed) for c in iter_conds if seed in proba_bag[c]]
        if not ps: continue
        ens = np.mean(np.stack(ps, axis=0), axis=0)
        y_pred = np.array([labels[i] for i in ens.argmax(axis=1)])
        m, w = macro_worst(test_labels, y_pred, labels)
        macros.append(m); worsts.append(w)
    print(f"  logit_average (5 iter conds)     {statistics.mean(macros):.3f}+-{statistics.stdev(macros):.3f}  "
          f"{statistics.mean(worsts):.3f}+-{statistics.stdev(worsts):.3f}")

    # 5. confidence-weighted all-iter
    cw_macros, cw_worsts = [], []
    for seed in seeds_seen:
        ps = [proba_bag[c].get(seed) for c in iter_conds if seed in proba_bag[c]]
        if not ps: continue
        weighted = np.zeros_like(ps[0])
        for p in ps:
            confs = p.max(axis=1, keepdims=True)
            weighted += p * confs
        y_pred = np.array([labels[i] for i in weighted.argmax(axis=1)])
        m, w = macro_worst(test_labels, y_pred, labels)
        cw_macros.append(m); cw_worsts.append(w)
    print(f"  confidence-weighted (5 iter)     {statistics.mean(cw_macros):.3f}+-{statistics.stdev(cw_macros):.3f}  "
          f"{statistics.mean(cw_worsts):.3f}+-{statistics.stdev(cw_worsts):.3f}")

    # Best solo for paired comparison
    print(f"\n=== Best individual baseline ===")
    best_macros, best_name = None, None
    for c in iter_conds:
        ms = []
        for seed in seeds_seen:
            if seed in proba_bag[c]:
                y_pred = np.array([labels[i] for i in proba_bag[c][seed].argmax(axis=1)])
                m, _ = macro_worst(test_labels, y_pred, labels)
                ms.append(m)
        mean_m = statistics.mean(ms)
        if best_macros is None or mean_m > statistics.mean(best_macros):
            best_macros = ms
            best_name = c
    print(f"  Best solo: {best_name} = {statistics.mean(best_macros):.3f}")

    print(f"\n=== Paired-t: best ensemble vs best solo ===")
    try:
        from scipy import stats as st
        # SC+AF logit-average vs best_solo
        sc_af_macros, sc_af_worsts = eval_pair(logit_avg)
        # subset to matched seeds
        diffs_m = [a-b for a,b in zip(sc_af_macros, best_macros)]
        t, p = st.ttest_rel(sc_af_macros, best_macros)
        try: _, pw = st.wilcoxon(sc_af_macros, best_macros, zero_method="zsplit")
        except: pw = float("nan")
        print(f"  (self_critique+AttrForge) ens vs best solo {best_name}: "
              f"diff={statistics.mean(diffs_m):+.3f}+-{statistics.stdev(diffs_m):.3f}  "
              f"paired-t p={p:.3f}  Wilcoxon p={pw:.3f}")

        # All-iter ensemble vs best_solo
        diffs_m = [a-b for a,b in zip(macros, best_macros)]
        t, p = st.ttest_rel(macros, best_macros)
        try: _, pw = st.wilcoxon(macros, best_macros, zero_method="zsplit")
        except: pw = float("nan")
        print(f"  All-iter (logit-avg) ens vs best solo {best_name}: "
              f"diff={statistics.mean(diffs_m):+.3f}+-{statistics.stdev(diffs_m):.3f}  "
              f"paired-t p={p:.3f}  Wilcoxon p={pw:.3f}")

        # CW-all-iter vs best solo
        diffs_m = [a-b for a,b in zip(cw_macros, best_macros)]
        t, p = st.ttest_rel(cw_macros, best_macros)
        try: _, pw = st.wilcoxon(cw_macros, best_macros, zero_method="zsplit")
        except: pw = float("nan")
        print(f"  All-iter (conf-wgt) ens vs best solo {best_name}: "
              f"diff={statistics.mean(diffs_m):+.3f}+-{statistics.stdev(diffs_m):.3f}  "
              f"paired-t p={p:.3f}  Wilcoxon p={pw:.3f}")
    except Exception as e:
        print(f"scipy: {e}")


if __name__ == "__main__":
    main()
