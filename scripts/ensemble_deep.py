"""Deep ensemble analysis: is AttrForge necessary for the ensemble win, or is
ensembling just a generic regularizer?

The v2 baseline finding: full_classic + full_attrforge ensemble beats
full_classic alone by +0.067 macro F1 at paired-t p=0.038. Two follow-ups:

1. Is AttrForge NECESSARY for the win? Compare:
     - any pair NOT containing AttrForge
     - any pair CONTAINING AttrForge
   If ensembling is generic regularization, all pairs should help equally.
   If AttrForge's diversity is the differentiator, AttrForge-containing
   ensembles should win.

2. What about 3-way and 4-way ensembles? If diversity is the active
   ingredient, adding more diverse conditions monotonically improves.

3. Compute per-condition pair LEAVE-ONE-OUT: ensemble of (all iterated
   conditions minus X) - does dropping AttrForge hurt the most?

Outputs:
  experiments/<base>_aggregated/ensemble_deep.json
  paper/figures/<base>_ensemble_deep.png
"""
from __future__ import annotations

import argparse
import itertools
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
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    # bag[condition][seed_idx] = proba matrix
    proba_bag = {}
    seeds_seen = []

    from sklearn.linear_model import LogisticRegression

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

    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]

    def ensemble_eval(condition_set):
        per_seed_macros, per_seed_worsts = [], []
        for seed in seeds_seen:
            probas = []
            for c in condition_set:
                if c in proba_bag and seed in proba_bag[c]:
                    probas.append(proba_bag[c][seed])
            if not probas: continue
            ens = np.mean(np.stack(probas, axis=0), axis=0)
            y_pred = np.array([labels[i] for i in ens.argmax(axis=1)])
            macro, worst = macro_worst(test_labels, y_pred, labels)
            per_seed_macros.append(macro); per_seed_worsts.append(worst)
        return per_seed_macros, per_seed_worsts

    # All ITERATED conditions for ensembling (skip naive, few_shot)
    iter_conds = [c for c in ["self_critique", "realism_only", "diversity_only",
                              "full_classic", "full_attrforge"] if c in proba_bag]

    print(f"\n=== Solo conditions ===")
    print(f"{'condition':<18} {'macro':<14} {'worst':<14}")
    solo_results = {}
    for c in iter_conds:
        m, w = ensemble_eval([c])
        solo_results[c] = (m, w)
        def fmt(v): return f"{statistics.mean(v):.3f}+-{statistics.stdev(v):.3f}"
        print(f"  {c:<18} {fmt(m):<14} {fmt(w):<14}")

    # 2. ALL PAIRS
    print(f"\n=== All pairs of iterated conditions (sorted by macro F1) ===")
    pair_results = []
    for ca, cb in itertools.combinations(iter_conds, 2):
        m, w = ensemble_eval([ca, cb])
        pair_results.append((ca, cb, m, w))
    pair_results.sort(key=lambda x: -statistics.mean(x[2]))
    print(f"{'pair':<48} {'macro':<14} {'worst':<14} {'gain vs best solo':<14}")
    for ca, cb, m, w in pair_results:
        best_solo = max(statistics.mean(solo_results[ca][0]),
                        statistics.mean(solo_results[cb][0]))
        ens_m = statistics.mean(m)
        gain = ens_m - best_solo
        def fmt(v): return f"{statistics.mean(v):.3f}+-{statistics.stdev(v):.3f}"
        print(f"  {(ca + ' + ' + cb):<48} {fmt(m):<14} {fmt(w):<14} {gain:+.3f}")

    # 3. AttrForge necessary?
    print(f"\n=== Does AttrForge matter? ===")
    af_pairs = [(ca, cb, m, w) for ca, cb, m, w in pair_results if "full_attrforge" in (ca, cb)]
    no_af_pairs = [(ca, cb, m, w) for ca, cb, m, w in pair_results if "full_attrforge" not in (ca, cb)]
    af_macros = [statistics.mean(m) for _, _, m, _ in af_pairs]
    no_af_macros = [statistics.mean(m) for _, _, m, _ in no_af_pairs]
    if af_macros and no_af_macros:
        print(f"  Pairs CONTAINING AttrForge      ({len(af_pairs)} pairs): "
              f"mean macro = {statistics.mean(af_macros):.3f}, "
              f"range [{min(af_macros):.3f}, {max(af_macros):.3f}]")
        print(f"  Pairs NOT containing AttrForge  ({len(no_af_pairs)} pairs): "
              f"mean macro = {statistics.mean(no_af_macros):.3f}, "
              f"range [{min(no_af_macros):.3f}, {max(no_af_macros):.3f}]")

    # 4. 3-way and 4-way ensembles including AttrForge
    print(f"\n=== Higher-order ensembles with AttrForge ===")
    for k in (3, 4, 5):
        if k > len(iter_conds): continue
        # all k-subsets that include full_attrforge
        af_subsets = [s for s in itertools.combinations(iter_conds, k)
                      if "full_attrforge" in s]
        if not af_subsets: continue
        # also all k-subsets that DO NOT include AttrForge
        no_af_subsets = [s for s in itertools.combinations(iter_conds, k)
                         if "full_attrforge" not in s]
        af_means = []
        for s in af_subsets:
            m, _ = ensemble_eval(list(s))
            af_means.append(statistics.mean(m))
        no_af_means = []
        for s in no_af_subsets:
            m, _ = ensemble_eval(list(s))
            no_af_means.append(statistics.mean(m))
        print(f"  k={k}: WITH AttrForge mean = {statistics.mean(af_means):.3f} "
              f"(range [{min(af_means):.3f}, {max(af_means):.3f}], n={len(af_subsets)} subsets)")
        if no_af_means:
            print(f"        NO AttrForge   mean = {statistics.mean(no_af_means):.3f} "
                  f"(range [{min(no_af_means):.3f}, {max(no_af_means):.3f}], n={len(no_af_subsets)} subsets)")

    # 5. ALL ITERATED ensemble vs ALL minus AttrForge (leave-one-out test)
    print(f"\n=== Leave-one-out ensemble of all iterated conditions ===")
    all_iter = iter_conds
    m_all, w_all = ensemble_eval(all_iter)
    print(f"  ALL ({len(all_iter)})    : macro = {statistics.mean(m_all):.3f}+-{statistics.stdev(m_all):.3f}, "
          f"worst = {statistics.mean(w_all):.3f}+-{statistics.stdev(w_all):.3f}")
    for drop in iter_conds:
        ens_set = [c for c in all_iter if c != drop]
        m, w = ensemble_eval(ens_set)
        delta_m = statistics.mean(m_all) - statistics.mean(m)
        delta_w = statistics.mean(w_all) - statistics.mean(w)
        print(f"  - {drop:<18}: macro = {statistics.mean(m):.3f}+-{statistics.stdev(m):.3f}  "
              f"(drop {delta_m:+.3f})  worst = {statistics.mean(w):.3f}+-{statistics.stdev(w):.3f}  "
              f"(drop {delta_w:+.3f})")

    # 6. Paired-t stats: every AttrForge-containing pair vs every non-AttrForge pair
    print(f"\n=== Paired stats: ALL-iterated ensemble vs best solo ===")
    from scipy import stats as st
    best_solo_cond = max(iter_conds, key=lambda c: statistics.mean(solo_results[c][0]))
    bs_macros = solo_results[best_solo_cond][0]
    t, p = st.ttest_rel(m_all, bs_macros)
    md = statistics.mean([a-b for a,b in zip(m_all, bs_macros)])
    print(f"  best solo = {best_solo_cond}, mean = {statistics.mean(bs_macros):.3f}")
    print(f"  all-iter ensemble = {statistics.mean(m_all):.3f}")
    print(f"  diff = {md:+.3f}, paired-t p = {p:.3f}")

    # Save
    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "solo": {c: {"macros": m, "worsts": w} for c, (m, w) in solo_results.items()},
        "pairs": [{"a": ca, "b": cb, "macros": m, "worsts": w} for ca, cb, m, w in pair_results],
        "all_iter_ensemble": {"macros": m_all, "worsts": w_all},
    }
    (out_dir / "ensemble_deep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/ensemble_deep.json")


if __name__ == "__main__":
    main()
