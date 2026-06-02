"""Per-class F1 for the cross-condition ensemble pair vs solo conditions.

Refits the sentence-transformer + LR downstream classifier on real_train + each
condition's synthetic batch (one classifier per condition per seed), then
ensembles the top pair (self_critique + full_attrforge) via logit average,
and reports per-class F1 mean +/- std across N=10 seeds.

The motivation: Figure 3's solo per-class F1 hides the ensemble win on the
hardest class (complaint). This computes the per-class F1 for the ensemble
to surface the win directly.

Saves: experiments/<base>_aggregated/per_class_ensemble.json
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

from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_synth(cond_dir: Path) -> list[SyntheticSample]:
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    f1s = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_true == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_true != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_true == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s[lbl] = 0.0
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s[lbl] = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="main_run_002")
    args = ap.parse_args()

    from scripts._splits_resolver import resolve_splits

    rt_path, te_path = resolve_splits(args.base)
    real_train = [RealExample.model_validate(r) for r in load_jsonl(rt_path)]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(te_path)]
    real_train_texts = [r.text for r in real_train]
    real_train_labels = np.asarray([r.label for r in real_train])
    test_texts = [r.text for r in real_test]
    test_labels = np.asarray([r.label for r in real_test])
    labels = sorted(set(test_labels.tolist()))

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # For each seed, train per-condition classifier; collect per-class F1 for
    # ENS pair (self_critique + full_attrforge) and for solo conditions, plus
    # the real-only baseline.
    target_conds = [
        "self_critique",
        "realism_only",
        "diversity_only",
        "full_classic",
        "full_attrforge",
    ]
    pair = ("self_critique", "full_attrforge")

    real_only_pcf1: dict[str, list[float]] = defaultdict(list)
    solo_pcf1: dict[str, dict[str, list[float]]] = {
        c: defaultdict(list) for c in target_conds
    }
    ens_pcf1: dict[str, list[float]] = defaultdict(list)

    # Real-only classifier (same for every seed, but recompute per-seed RNG).
    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        # Real-only baseline (re-fit per seed for RNG consistency).
        clf_real = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
        )
        clf_real.fit(X_real_train, real_train_labels)
        y_pred_real = clf_real.predict(X_test)
        for k, v in per_class_f1(test_labels, y_pred_real, labels).items():
            real_only_pcf1[k].append(v)

        proba_by_cond: dict[str, np.ndarray] = {}
        for cond in target_conds:
            cond_dir = sd / cond
            if not cond_dir.exists():
                continue
            samples = load_synth(cond_dir)
            if not samples:
                continue
            synth_texts = [s.text for s in samples]
            synth_labels = np.asarray(
                [s.requested_attributes.get("intent", "?") for s in samples]
            )
            X_synth = enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False)
            X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            y_tr = np.concatenate([real_train_labels, synth_labels])
            clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", random_state=17
            )
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_test)
            col_idx = [list(clf.classes_).index(lbl) for lbl in labels]
            proba = proba[:, col_idx]
            proba_by_cond[cond] = proba
            y_pred = np.asarray([labels[i] for i in proba.argmax(axis=1)])
            for k, v in per_class_f1(test_labels, y_pred, labels).items():
                solo_pcf1[cond][k].append(v)

        # Ensemble: logit-average pair
        if all(c in proba_by_cond for c in pair):
            ens = (proba_by_cond[pair[0]] + proba_by_cond[pair[1]]) / 2.0
            y_pred_ens = np.asarray([labels[i] for i in ens.argmax(axis=1)])
            for k, v in per_class_f1(test_labels, y_pred_ens, labels).items():
                ens_pcf1[k].append(v)

    # Print + save
    print()
    print(
        f"{'class':<22} {'real-only':<14} "
        f"{'full_classic':<14} {'full_attrforge':<14} "
        f"{'ENS (sc+af)':<14}"
    )
    out = {"real_only": dict(real_only_pcf1), "solo": {c: dict(v) for c, v in solo_pcf1.items()}, "ensemble": dict(ens_pcf1)}
    for lbl in labels:
        ro = real_only_pcf1[lbl]
        fc = solo_pcf1["full_classic"][lbl]
        fa = solo_pcf1["full_attrforge"][lbl]
        en = ens_pcf1[lbl]
        def fmt(v):
            if not v:
                return "n/a"
            return f"{statistics.mean(v):.2f}+-{statistics.stdev(v) if len(v) > 1 else 0:.2f}"
        print(f"  {lbl:<22} {fmt(ro):<14} {fmt(fc):<14} {fmt(fa):<14} {fmt(en):<14}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_class_ensemble.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_dir}/per_class_ensemble.json")


if __name__ == "__main__":
    main()
