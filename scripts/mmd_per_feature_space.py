"""Maximum Mean Discrepancy (MMD) between synthetic and real distributions
under each downstream feature space.

Reviewer MA-NEW-6 (round 4, carried to round 5): the "embedding absorbs
surface diversity" mechanism claim is asserted without direct measurement.
This script provides that direct measurement: for each (condition,
feature_space) pair, we compute the squared MMD with RBF kernel (Gretton
et al. 2012) between the condition's synthetic batch and the held-out
real seed set. The pattern we expect IF the mechanism claim holds:

  under SENTENCE-TRANSFORMER features:
      MMD(full_attrforge, real) ~ MMD(full_classic, real)
      (diversity is "absorbed" as added coverage, so distributional
       distance does not increase)

  under TF-IDF WORD features:
      MMD(full_attrforge, real) > MMD(full_classic, real)
      (diversity moves away from real's keyword distribution)

Bandwidth: median heuristic over pairwise squared distances.

Outputs:
    experiments/<base>_aggregated/mmd_per_feature_space.json
    paper/figures/<base>_mmd_per_feature_space.png
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


def rbf_mmd2(X, Y, sigma=None):
    """Squared MMD with RBF kernel.

    X, Y: (n, d), (m, d) arrays of features.
    sigma: kernel bandwidth. If None, median-heuristic on the combined pairwise sqdists.
    Returns the unbiased MMD^2 estimator.
    """
    # Convert sparse to dense BEFORE np.asarray (np.asarray on sparse creates 0-d object array)
    Xn = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Yn = Y.toarray() if hasattr(Y, "toarray") else np.asarray(Y)
    Xn = np.asarray(Xn, dtype=np.float64)
    Yn = np.asarray(Yn, dtype=np.float64)
    n, m = Xn.shape[0], Yn.shape[0]

    # Pairwise squared distances
    Kxx = pairwise_sqdist(Xn, Xn)
    Kyy = pairwise_sqdist(Yn, Yn)
    Kxy = pairwise_sqdist(Xn, Yn)

    if sigma is None:
        # Median heuristic over combined off-diagonal sqdists
        all_sqd = np.concatenate([
            Kxx[np.triu_indices(n, k=1)],
            Kyy[np.triu_indices(m, k=1)],
            Kxy.flatten(),
        ])
        med = np.median(all_sqd)
        sigma = np.sqrt(max(med / 2.0, 1e-8))

    g = 1.0 / (2.0 * sigma * sigma)
    Kxx_k = np.exp(-g * Kxx)
    Kyy_k = np.exp(-g * Kyy)
    Kxy_k = np.exp(-g * Kxy)

    # Unbiased estimator (Gretton 2012, eq. 3)
    np.fill_diagonal(Kxx_k, 0.0)
    np.fill_diagonal(Kyy_k, 0.0)
    mmd2 = Kxx_k.sum() / (n * (n - 1)) + Kyy_k.sum() / (m * (m - 1)) - 2.0 * Kxy_k.mean()
    return float(max(mmd2, 0.0)), float(sigma)


def pairwise_sqdist(A, B):
    a2 = (A * A).sum(axis=1, keepdims=True)
    b2 = (B * B).sum(axis=1, keepdims=True)
    return np.maximum(a2 + b2.T - 2.0 * A @ B.T, 0.0)


def featurize_tfidf_word(train_texts, synth_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    v.fit(train_texts + synth_texts)
    return v.transform(train_texts), v.transform(synth_texts)


def featurize_tfidf_char(train_texts, synth_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    v.fit(train_texts + synth_texts)
    return v.transform(train_texts), v.transform(synth_texts)


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
    X_real_st = enc.encode(real_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # bag[condition][feature_space] -> list[mmd^2 per seed]
    bag = defaultdict(lambda: defaultdict(list))

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth:
                continue
            synth_texts = [s.text for s in synth]

            # TF-IDF word
            Xr, Xs = featurize_tfidf_word(real_texts, synth_texts)
            mmd, _ = rbf_mmd2(Xr, Xs)
            bag[cond_dir.name]["tfidf_word"].append(mmd)

            # TF-IDF char
            Xr, Xs = featurize_tfidf_char(real_texts, synth_texts)
            mmd, _ = rbf_mmd2(Xr, Xs)
            bag[cond_dir.name]["tfidf_char"].append(mmd)

            # Sentence-transformer
            Xs_st = enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False)
            mmd, _ = rbf_mmd2(X_real_st, Xs_st)
            bag[cond_dir.name]["st"].append(mmd)

    conds = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    fs_names = {"tfidf_word": "TF-IDF word", "tfidf_char": "TF-IDF char 3-5", "st": "Sent-trans MiniLM"}

    print(f"\nSquared MMD (synthetic vs real seed set), mean ± std across 5 seeds, lower = closer to real.\n")
    print(f"{'condition':<18}", end="")
    for fs in fs_names:
        print(f" {fs_names[fs]:<22}", end="")
    print()

    for c in conds:
        if c not in bag:
            continue
        print(f"{c:<18}", end="")
        for fs in fs_names:
            vals = bag[c][fs]
            if not vals:
                print(f" {'n/a':<22}", end=""); continue
            m = statistics.mean(vals); sd = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f" {m:.4f} ± {sd:.4f}     ", end="")
        print()

    # Paired comparisons full_attrforge vs full_classic per feature space
    print("\nPaired comparison full_attrforge - full_classic (signed):")
    try:
        from scipy import stats as st
        for fs in fs_names:
            fc = bag["full_classic"][fs]
            fa = bag["full_attrforge"][fs]
            if not fc or not fa:
                continue
            diffs = [a - b for a, b in zip(fa, fc)]
            md = statistics.mean(diffs); sd = statistics.stdev(diffs)
            t, p = st.ttest_rel(fa, fc)
            print(f"  {fs_names[fs]:<22}: diff = {md:+.5f} ± {sd:.5f}; paired-t p={p:.3f}; per-seed: {[round(d,5) for d in diffs]}")
    except Exception as e:
        print(f"  scipy: {e}")

    # Save
    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_spaces": list(fs_names),
        "mmd": {c: {fs: bag[c][fs] for fs in fs_names if bag[c][fs]} for c in conds if c in bag},
    }
    (out_dir / "mmd_per_feature_space.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/mmd_per_feature_space.json")

    # 3-panel bar chart: MMD per condition, per feature space
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    iter_conds = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    for ax, fs in zip(axes, fs_names):
        means = [statistics.mean(bag[c][fs]) if bag[c][fs] else 0 for c in iter_conds]
        sds = [statistics.stdev(bag[c][fs]) if bag[c][fs] and len(bag[c][fs]) > 1 else 0 for c in iter_conds]
        xs = list(range(len(iter_conds)))
        colors = ["#999999"] * 5 + ["#3a6ea5", "#c0392b"]
        ax.bar(xs, means, yerr=sds, capsize=3, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(iter_conds, rotation=20, ha="right", fontsize=8)
        ax.set_title(fs_names[fs])
        ax.set_ylabel("squared MMD (synth vs real)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Distributional distance synthetic vs real seed set, per feature space (lower = closer to real)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(REPO / "paper" / "figures" / f"{args.base}_mmd_per_feature_space.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_mmd_per_feature_space.png", dpi=160, bbox_inches="tight")
    print(f"Saved figure: paper/figures/{args.base}_mmd_per_feature_space.png")


if __name__ == "__main__":
    main()
