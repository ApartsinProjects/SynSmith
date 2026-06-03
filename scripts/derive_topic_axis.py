"""Derive an empirical topic axis from a real-seed JSONL.

Task #74 fix for the TREC topic-coverage anti-pattern: the schema
constrains attribute axes the user named (intent, difficulty, style,
scenario_type) but does not constrain the TOPIC axis. The generator
produces topic-diverse content that does not overlap the test set's
topic distribution. Fix: cluster the real seed by sentence-transformer
embeddings, label each cluster by top-TF-IDF terms, and emit a new
schema with `topic` added drawn from the empirical clusters.

Usage:
    python scripts/derive_topic_axis.py \
        --seed experiments/_splits/trec_real_train.jsonl \
        --in-schema  examples/trec/schema.yaml \
        --out-schema examples/trec/schema_topic.yaml \
        --k 8

The downstream loop sees `topic` as just another labeled attribute, so
the balanced planner (Fix A) automatically enforces per-topic coverage
the same way it enforces per-class coverage. No code change needed
beyond the schema. The real-seed JSONL is NOT modified, so existing
anchoring stays consistent for the class attribute.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import yaml


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{2,}")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "are", "you",
    "what", "who", "where", "when", "why", "how", "which", "many", "much",
    "did", "does", "has", "have", "had", "can", "could", "would", "should",
    "his", "her", "she", "him", "they", "their", "them", "any", "all",
    "from", "into", "over", "than", "then", "there", "here", "some", "such",
    "name", "names", "called",
}


def load_seed(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def derive_topics(rows: list[dict], k: int, seed: int = 17) -> tuple[list[str], dict[str, int]]:
    """Cluster real-seed texts and return cluster labels + per-row cluster index.

    Strategy: TF-IDF vectorize -> KMeans -> label each cluster with the
    most-distinctive token among in-cluster texts vs out-of-cluster texts
    (a TF-IDF style ratio with stopword and short-token filtering).

    Uses TF-IDF rather than sentence-transformer embeddings deliberately:
    (a) at N=60-300 short questions the surface lexicon dominates topic
    similarity anyway, (b) torch DLL load on Windows is fragile (paging-
    file errors at small heap sizes), (c) the clusterer output is the
    cluster ASSIGNMENT, not a continuous geometry, so the small loss in
    semantic granularity does not change the downstream planner's
    coverage-balancing behavior.
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [r["text"] for r in rows]
    vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[a-zA-Z][a-zA-Z\-']{2,}",
        max_df=0.85,
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    # KMeans on sparse TF-IDF features; cosine-like by L2-normalizing rows.
    from sklearn.preprocessing import normalize
    X = normalize(X, norm="l2", axis=1)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_ix = km.fit_predict(X)

    # Label each cluster by its most-distinctive content tokens.
    cluster_tokens: dict[int, Counter] = {i: Counter() for i in range(k)}
    global_tokens: Counter = Counter()
    for idx, txt in enumerate(texts):
        toks = [t for t in tokenize(txt) if t not in _STOPWORDS]
        cluster_tokens[cluster_ix[idx]].update(toks)
        global_tokens.update(toks)

    labels: list[str] = []
    for i in range(k):
        in_c = cluster_tokens[i]
        # Score: in-cluster freq / (out-of-cluster freq + 1).
        scored: list[tuple[str, float]] = []
        for tok, c_in in in_c.items():
            c_out = global_tokens[tok] - c_in
            score = c_in / (c_out + 1)
            scored.append((tok, score))
        scored.sort(key=lambda x: -x[1])
        # Take the top 2 distinctive tokens, joined by underscore for a
        # schema-safe label. Fallback to "cluster_<i>" if nothing distinct.
        picks = [t for t, s in scored[:6] if s > 0.5][:2]
        label = "_".join(picks) if picks else f"cluster_{i}"
        labels.append(label)

    # Deduplicate (rare): if two clusters resolved to the same label, suffix.
    seen: dict[str, int] = {}
    final_labels: list[str] = []
    for lab in labels:
        n = seen.get(lab, 0)
        seen[lab] = n + 1
        final_labels.append(lab if n == 0 else f"{lab}_{n+1}")

    row_to_topic = {
        rows[i].get("sample_id") or str(i): final_labels[cluster_ix[i]]
        for i in range(len(rows))
    }
    return final_labels, row_to_topic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, type=Path,
                    help="Real-seed JSONL (one {text, label, ...} per line).")
    ap.add_argument("--in-schema", required=True, type=Path,
                    help="Source schema YAML.")
    ap.add_argument("--out-schema", required=True, type=Path,
                    help="Destination schema YAML with topic axis added.")
    ap.add_argument("--k", type=int, default=8,
                    help="Number of topic clusters to derive.")
    ap.add_argument("--cluster-seed", type=int, default=17)
    args = ap.parse_args()

    rows = load_seed(args.seed)
    print(f"Loaded {len(rows)} real-seed examples from {args.seed}", flush=True)

    labels, _row_to_topic = derive_topics(rows, args.k, seed=args.cluster_seed)
    print(f"Derived {len(labels)} topic clusters:")
    for lab in labels:
        print(f"  - {lab}")

    schema = yaml.safe_load(args.in_schema.read_text(encoding="utf-8"))
    # Insert topic right after intent so the planner sees it as a primary axis.
    new_attrs = {}
    for name, values in schema["attributes"].items():
        new_attrs[name] = values
        if name == schema.get("label_attribute", "intent"):
            new_attrs["topic"] = labels
    schema["attributes"] = new_attrs

    args.out_schema.parent.mkdir(parents=True, exist_ok=True)
    args.out_schema.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.out_schema}", flush=True)


if __name__ == "__main__":
    main()
