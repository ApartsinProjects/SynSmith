"""Three-way comparison: OLD framework vs v2.9.7 vs v2.9.6 (SynSmith).

Reports per-dataset, per-seed accuracy + macro-F1 using sentence-
transformer + LR (the paper's headline evaluator). Designed to run as
soon as the v297 sweep completes; falls back gracefully when v297 data
is partial.

Output format matches the paper's intended Table X structure:

  Dataset   | Real-only | OLD     | v2.9.7  | v2.9.6  | delta v297-OLD | delta v297-real
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def _load_synth(cond_dir):
    """Load samples.jsonl PLUS samples_regen.jsonl (Fix B re-generations)."""
    out = []
    for sj in sorted(cond_dir.rglob("samples.jsonl")):
        for r in load_jsonl(sj):
            out.append(SyntheticSample.model_validate(r))
    for sj in sorted(cond_dir.rglob("samples_regen.jsonl")):
        for r in load_jsonl(sj):
            out.append(SyntheticSample.model_validate(r))
    return out


def _seed_eval(samples, enc, X_test, y_test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    if len(samples) < 2:
        return None
    texts = [s.text for s in samples]
    labels = np.array([s.requested_attributes.get("intent", "?") for s in samples])
    if len(set(labels)) < 2:
        return None
    X = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
    )
    clf.fit(X, labels)
    y_pred = clf.predict(X_test)
    return (
        float(accuracy_score(y_test, y_pred)),
        float(f1_score(y_test, y_pred, average="macro")),
        len(samples),
    )


def compare_dataset(name, train_path, test_path, run_patterns, seeds, enc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    real_train = [RealExample.model_validate(r) for r in load_jsonl(train_path)]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(test_path)]
    y_test = np.array([r.label for r in real_test])
    X_test = enc.encode(
        [r.text for r in real_test], normalize_embeddings=True, show_progress_bar=False
    )
    X_real = enc.encode(
        [r.text for r in real_train], normalize_embeddings=True, show_progress_bar=False
    )
    y_real = np.array([r.label for r in real_train])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_real, y_real)
    y_pred = clf.predict(X_test)
    acc_ro = float(accuracy_score(y_test, y_pred))
    f1_ro = float(f1_score(y_test, y_pred, average="macro"))

    results = {}
    for label, pattern in run_patterns.items():
        accs, f1s, ns = [], [], []
        for s in seeds:
            cond_dir = REPO / pattern.format(seed=s) / "full_attrforge"
            if not cond_dir.exists():
                continue
            samples = _load_synth(cond_dir)
            res = _seed_eval(samples, enc, X_test, y_test, s)
            if res is None:
                continue
            accs.append(res[0])
            f1s.append(res[1])
            ns.append(res[2])
        results[label] = {"accs": accs, "f1s": f1s, "ns": ns}

    def fmt(arr):
        if not arr:
            return "no data"
        if len(arr) == 1:
            return f"{arr[0]:.3f} (N=1)"
        return f"{np.mean(arr):.3f} +- {np.std(arr, ddof=1):.3f} (N={len(arr)})"

    print(f"\n=== {name} (sentence-transformer + LR, n_test={len(real_test)}) ===")
    print(f"  Real-only baseline:                acc {acc_ro:.3f}   f1 {f1_ro:.3f}")
    for label in run_patterns.keys():
        r = results[label]
        n_synth_mean = int(np.mean(r["ns"])) if r["ns"] else 0
        print(f"  {label:<10} full_attrforge acc:  {fmt(r['accs'])}  (mean n_synth = {n_synth_mean})")
    labels = list(run_patterns.keys())
    if len(labels) >= 2 and results[labels[-1]]["accs"]:
        latest_label = labels[-1]
        latest_mean = float(np.mean(results[latest_label]["accs"]))
        for other in labels[:-1]:
            if results[other]["accs"]:
                d = latest_mean - float(np.mean(results[other]["accs"]))
                print(f"  Delta {latest_label} - {other}:  {d:+.3f}pp")
        d_real = latest_mean - acc_ro
        print(f"  Delta {latest_label} - real-only:  {d_real:+.3f}pp")


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    seeds = [17, 23, 41, 53, 89]
    print("=== Three-way comparison: OLD vs v2.9.7 vs v2.9.6 (SynSmith) ===")
    print("    sentence-transformer + LR headline evaluator")
    for ds_name, train, test in [
        ("SST-2",     "experiments/_splits/sst2_real_train.jsonl",     "experiments/_splits/sst2_real_test.jsonl"),
        ("Banking77", "experiments/_splits/banking77_real_train.jsonl", "experiments/_splits/banking77_real_test.jsonl"),
        ("TREC",      "experiments/_splits/trec_real_train.jsonl",      "experiments/_splits/trec_real_test.jsonl"),
    ]:
        compare_dataset(
            ds_name,
            train,
            test,
            run_patterns={
                "OLD":  "experiments/" + ds_name.lower().replace("-", "") + "_run_001_seed{seed}",
                "v297": "experiments/" + ds_name.lower().replace("-", "") + "_v297_seed{seed}",
                "v297": "experiments/" + ds_name.lower().replace("-", "") + "_v297_seed{seed}",
            },
            seeds=seeds,
            enc=enc,
        )


if __name__ == "__main__":
    main()
