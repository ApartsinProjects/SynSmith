"""Re-audit with bug-fixed Mode-Seeking and Coverage AUROC.

Root causes identified during v2:

1. Mode-Seeking constancy (0.23 +/- 0.01 across all conditions in v1
   Table 9): the v1 critic uses TF-IDF cosine for "text distance" with
   use_embeddings=False (mode_seeking.py default). TF-IDF cosine is
   bounded in a narrow range for short utterances and is blind to
   synonym substitution and structural reordering. AttrForge's diversity
   manifests precisely as synonym + structural variation, which TF-IDF
   does not see. Re-audit with sentence-transformer embeddings should
   recover the differentiating signal.

2. Coverage AUROC saturation at 1.00 across all conditions (v1 Table 9):
   the v1 critic fits an LR on TF-IDF (uni+bigram) features over 78
   docs (30 real + 48 synth) and reports the in-sample AUROC. The
   feature space is high-dimensional; balanced LR with C=1.0 trivially
   memorizes. Re-audit with 5-fold cross-validated AUROC gives a
   generalization-faithful coverage signal.

Outputs:
  experiments/<base>_aggregated/reaudit_fixed.json
  paper/figures/<base>_reaudit_fixed.png
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


def hamming(a: dict, b: dict) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def mode_seeking_emb(batch, enc):
    """MS ratio computed on sentence-transformer embeddings (NOT TF-IDF)."""
    if len(batch) < 2:
        return {"ms_ratio": 0.0, "text_dist_mean": 0.0, "target_dist_mean": 0.0,
                "attribute_sensitivity": {}}
    texts = [s.text for s in batch]
    targets = [s.requested_attributes for s in batch]
    emb = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    n = len(texts)
    text_dists, target_dists = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(emb[i] @ emb[j])
            text_dists.append(1.0 - sim)  # cosine distance
            target_dists.append(hamming(targets[i], targets[j]))
    text_arr = np.asarray(text_dists)
    targ_arr = np.asarray(target_dists)
    nonzero = targ_arr > 0
    ratio = float((text_arr[nonzero] / targ_arr[nonzero]).mean()) if nonzero.any() else 0.0

    # Per-attribute sensitivity: mean text-distance for pairs differing in
    # ONLY one attribute.
    all_attrs = set().union(*targets)
    per_attr = {a: [] for a in all_attrs}
    for i in range(n):
        for j in range(i + 1, n):
            diff = [a for a in all_attrs if targets[i].get(a) != targets[j].get(a)]
            if len(diff) == 1:
                d = 1.0 - float(emb[i] @ emb[j])
                per_attr[diff[0]].append(d)
    sens = {a: (float(np.mean(v)) if v else 0.0) for a, v in per_attr.items()}
    return {"ms_ratio": ratio, "text_dist_mean": float(text_arr.mean()),
            "target_dist_mean": float(targ_arr.mean()),
            "attribute_sensitivity": sens}


def coverage_auroc_cv(real_texts, synth_texts, enc, n_folds=5):
    """Cross-validated AUROC for the real-vs-synth classifier on sentence-
    transformer features. This is a generalization-faithful coverage signal
    (a uniformly-covering synth distribution should produce AUROC near 0.5,
    a separable one near 1.0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    X_real = enc.encode(real_texts, normalize_embeddings=True, show_progress_bar=False)
    X_synth = enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False)
    X = np.concatenate([X_real, X_synth], axis=0)
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_synth))])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=17)
    fold_aurocs = []
    for tr_idx, te_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 random_state=17)
        clf.fit(X[tr_idx], y[tr_idx])
        proba = clf.predict_proba(X[te_idx])[:, list(clf.classes_).index(1)]
        try:
            fold_aurocs.append(roc_auc_score(y[te_idx], proba))
        except Exception:
            continue
    return float(np.mean(fold_aurocs)) if fold_aurocs else 0.5


def vendi_score(emb_matrix):
    """Vendi score (Friedman & Dieng 2023): effective number of distinct
    samples in the batch via the exp-entropy of the eigenvalues of the
    similarity-kernel matrix divided by n. Higher = more semantically
    diverse. Computed on the sentence-transformer Gram matrix.
    """
    K = emb_matrix @ emb_matrix.T
    n = K.shape[0]
    # eigenvalues of K/n, normalized so they sum to 1
    eigvals = np.linalg.eigvalsh(K / n)
    eigvals = eigvals[eigvals > 1e-12]
    eigvals = eigvals / eigvals.sum()
    entropy = -(eigvals * np.log(eigvals)).sum()
    return float(np.exp(entropy))


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    from scripts._splits_resolver import resolve_splits
    _real_train_path, _ = resolve_splits(args.base)
    real_train = [RealExample.model_validate(r) for r in load_jsonl(_real_train_path)]
    real_texts = [r.text for r in real_train]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
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
            synth_texts = [s.text for s in samples]
            # ms ratio with embeddings
            ms = mode_seeking_emb(samples, enc)
            # AUROC with 5-fold CV
            auroc = coverage_auroc_cv(real_texts, synth_texts, enc)
            # Vendi score (extra semantic-diversity signal)
            synth_emb = enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False)
            vendi = vendi_score(synth_emb)
            bag[cond_dir.name].append({
                "seed": seed,
                "ms_emb": ms["ms_ratio"],
                "coverage_auroc_cv": auroc,
                "vendi": vendi,
                "text_dist_mean": ms["text_dist_mean"],
            })

    # Conditions in canonical paper order; the leave-one-out ablations are
    # appended so reaudit_fixed runs on loo_run_001 as well as main_run_002.
    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge",
             "no_pack", "no_mode_seeking", "no_mode_hunter", "no_coverage_hole"]

    print(f"\n=== Re-audit with FIXED metrics (mean +/- std over seeds) ===")
    print(f"{'condition':<18} {'ms (emb)':<14} {'cov AUROC (cv)':<14} {'Vendi score':<14} {'pair-text-dist':<14}")
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        def fmt(key):
            v = [r[key] for r in rows]
            m = statistics.mean(v); sd = statistics.stdev(v) if len(v)>1 else 0
            return f"{m:.3f}+-{sd:.3f}"
        print(f"{c:<18} {fmt('ms_emb'):<14} {fmt('coverage_auroc_cv'):<14} {fmt('vendi'):<14} {fmt('text_dist_mean'):<14}")

    print(f"\n=== Paired stats: full_attrforge vs full_classic ===")
    try:
        from scipy import stats as st
        for metric in ("ms_emb", "coverage_auroc_cv", "vendi", "text_dist_mean"):
            fc = [r[metric] for r in bag.get("full_classic", [])]
            fa = [r[metric] for r in bag.get("full_attrforge", [])]
            if not fc or not fa:
                continue
            diffs = [a-b for a,b in zip(fa, fc)]
            t, p = st.ttest_rel(fa, fc)
            try:
                _, pw = st.wilcoxon(fa, fc, zero_method="zsplit")
            except ValueError:
                pw = float("nan")
            md = statistics.mean(diffs); sd = statistics.stdev(diffs) if len(diffs)>1 else 0
            print(f"  {metric:<22} diff={md:+.4f}+-{sd:.4f}  paired-t p={p:.3f}  Wilcoxon p={pw:.3f}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reaudit_fixed.json").write_text(json.dumps(
        {"augmented": {c: bag[c] for c in conds if c in bag}}, indent=2
    ), encoding="utf-8")
    print(f"\nSaved: {out_dir}/reaudit_fixed.json")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    iter_conds = [c for c in conds if c in bag]
    xs = list(range(len(iter_conds)))
    for ax, (key, title) in zip(axes, [
        ("ms_emb", "Mode-Seeking ratio (embeddings)\nhigher = more attribute-responsive"),
        ("coverage_auroc_cv", "Coverage AUROC (5-fold CV)\nlower = synth covers real better"),
        ("vendi", "Vendi score (semantic diversity)\nhigher = more diverse"),
    ]):
        m = [statistics.mean([r[key] for r in bag[c]]) for c in iter_conds]
        sd = [statistics.stdev([r[key] for r in bag[c]]) if len(bag[c]) > 1 else 0
              for c in iter_conds]
        colors = ["#999"] * 5 + ["#3a6ea5", "#c0392b"]
        ax.bar(xs, m, yerr=sd, capsize=4, color=colors[:len(iter_conds)])
        for i, (mm, s) in enumerate(zip(m, sd)):
            ax.text(i, mm + s + 0.005, f"{mm:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(iter_conds, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Post-hoc adversary audit, fixed metrics (mean +/- std, 5 seeds)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(REPO / "paper" / "figures" / f"{args.base}_reaudit_fixed.png",
                dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_reaudit_fixed.png",
                dpi=160, bbox_inches="tight")
    print(f"Saved figure")


if __name__ == "__main__":
    main()
