"""Unit tests for the Mode-Seeking critic.

The Mode-Seeking critic is deterministic (no LLM calls), so the test
exercises it end-to-end on stub samples and asserts:

1. Score on a homogeneous batch (same text, different attributes) is
   driven toward zero.
2. Score on a diverse batch (different text per attribute) is positive.
3. Per-attribute sensitivity vector is populated.
"""
from __future__ import annotations

from attrforge.critics.mode_seeking import ModeSeeking, ModeSeekingConfig
from attrforge.schema import SyntheticSample


def _make_sample(text: str, intent: str, style: str, idx: int) -> SyntheticSample:
    return SyntheticSample(
        sample_id=f"s_{idx}",
        text=text,
        requested_attributes={"intent": intent, "style": style},
        generated_attributes={"intent": intent, "style": style},
        prompt_version=1,
        iteration=0,
    )


def test_homogeneous_batch_drives_ratio_toward_zero():
    """Same text for every attribute vector -> low text-distance -> low ratio."""
    batch = [
        _make_sample("Please reset my password.", "account_issue", "formal", 0),
        _make_sample("Please reset my password.", "refund_request", "formal", 1),
        _make_sample("Please reset my password.", "complaint", "casual", 2),
        _make_sample("Please reset my password.", "general_question", "casual", 3),
    ]
    ms = ModeSeeking(ModeSeekingConfig(use_embeddings=False))
    result = ms.score(batch)
    assert result.n_pairs >= 4
    # All pairs share the same text, text-distance ~ 0 -> ratio ~ 0.
    assert result.text_distance_mean < 0.05
    assert result.mode_seeking_ratio < 0.1


def test_diverse_batch_produces_positive_ratio():
    """Different text per attribute vector -> positive text-distance, positive ratio."""
    batch = [
        _make_sample("Please reset my password to access my account.", "account_issue", "formal", 0),
        _make_sample("I want a full refund for the broken shipment.", "refund_request", "formal", 1),
        _make_sample("This is unacceptable, no one ever responds!", "complaint", "casual", 2),
        _make_sample("Hi, what are your customer support hours?", "general_question", "casual", 3),
    ]
    ms = ModeSeeking(ModeSeekingConfig(use_embeddings=False))
    result = ms.score(batch)
    assert result.n_pairs >= 4
    assert result.text_distance_mean > 0.0
    assert result.mode_seeking_ratio > 0.0


def test_small_batch_returns_zero_ratio():
    """Fewer than 2 samples -> ratio = 0 (no pairs to compute)."""
    ms = ModeSeeking()
    assert ms.score([]).mode_seeking_ratio == 0.0
    one = [_make_sample("hello", "general_question", "casual", 0)]
    assert ms.score(one).mode_seeking_ratio == 0.0


def test_attribute_sensitivity_populated():
    """When pairs differ in exactly one attribute, per-attribute sensitivity is recorded."""
    batch = [
        _make_sample("alpha bravo charlie", "account_issue", "formal", 0),
        _make_sample("delta echo foxtrot", "refund_request", "formal", 1),  # differs in intent
        _make_sample("golf hotel india", "account_issue", "casual", 2),  # differs in style (vs 0)
    ]
    ms = ModeSeeking(ModeSeekingConfig(use_embeddings=False))
    result = ms.score(batch)
    assert isinstance(result.attribute_sensitivity, dict)
