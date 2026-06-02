"""Unit tests for the Mode Hunter's deterministic helpers.

The Mode Hunter's main hunt() method calls an LLM. Two helpers
(_count_substring and top_ngrams) are deterministic and used to verify
candidate banned patterns. We test them here without any API key.
"""
from __future__ import annotations

from attrforge.critics.mode_hunter import ModeHunter, ModeHunterConfig


def test_count_substring_counts_corpus_occurrences():
    """_count_substring returns the number of corpus entries containing the substring."""
    corpus = [
        "Hello team, hope you are well.",
        "Hello team, just checking in.",
        "Hello team, quick question.",
        "Different opener entirely.",
    ]
    assert ModeHunter._count_substring("Hello team,", corpus) == 3


def test_count_substring_zero_for_missing_pattern():
    """An LLM-returned candidate that does not actually appear, count = 0."""
    corpus = ["Hello team, hope you are well.", "Hello team, just checking in."]
    assert ModeHunter._count_substring("absolutely never said", corpus) == 0


def test_top_ngrams_returns_most_frequent_synth_ngrams():
    """top_ngrams surfaces frequent n-grams from a text list as (ngram, count) pairs."""
    texts = [
        "Hello team, can I help you?",
        "Hello team, what can I do?",
        "Hello team, is there an issue?",
    ]
    pairs = ModeHunter.top_ngrams(texts, n=2, top_k=5)
    joined = " | ".join(f"{ng}:{ct}" for ng, ct in pairs)
    assert "hello team" in joined.lower()
    # Counts are positive ints.
    assert all(isinstance(ct, int) and ct > 0 for _, ct in pairs)


def test_library_property_starts_empty():
    """A fresh ModeHunter has no remembered findings."""
    hunter = ModeHunter(client=None, config=ModeHunterConfig())
    assert hunter.library == []
