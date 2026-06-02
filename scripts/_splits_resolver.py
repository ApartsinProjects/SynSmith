"""Resolve real_train.jsonl / real_test.jsonl paths from a --base argument.

Used by every aggregation script that needs to load the real-train / real-
test splits. The pattern: if the --base name contains a known dataset stem
(banking77, trec, ...), use that dataset's splits; otherwise fall back to
the customer-support default ('real_train.jsonl', 'real_test.jsonl').

This avoids the namespace overwrite bug where Banking77 / TREC runs silently
loaded customer-support splits and produced degenerate results.
"""
from __future__ import annotations

from pathlib import Path

DATASET_STEMS = {
    "banking77": ("banking77_real_train.jsonl", "banking77_real_test.jsonl"),
    "trec": ("trec_real_train.jsonl", "trec_real_test.jsonl"),
    "sst2": ("sst2_real_train.jsonl", "sst2_real_test.jsonl"),
}


def resolve_splits(base: str, splits_root: Path | None = None) -> tuple[Path, Path]:
    """Return (real_train_path, real_test_path) for a given --base name."""
    if splits_root is None:
        splits_root = (
            Path(__file__).resolve().parents[1] / "experiments" / "_splits"
        )
    base_lower = base.lower()
    for stem, (tr, te) in DATASET_STEMS.items():
        if stem in base_lower:
            return splits_root / tr, splits_root / te
    return splits_root / "real_train.jsonl", splits_root / "real_test.jsonl"
