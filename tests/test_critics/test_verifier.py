"""Unit tests for the empirical-anchor Verifier.

The Verifier's main verify() method calls an LLM. The deterministic
parts that this test pins:

1. Anchor lookup: real-seed examples bucketed by (attribute, value) and
   sampled when their attribute appears in a requested-attributes dict.
2. Graceful fallback when no real examples are available (the formatter
   returns a clear message and verification still works via schema-only
   interpretation).
3. Real examples are sampled, not exhaustively included (k_real_per_value
   cap respected).
"""
from __future__ import annotations

import json

from attrforge.critics.verifier import AttributeVerifier, VerifierConfig
from attrforge.schema import AttributeSchema, RealExample


def _schema() -> AttributeSchema:
    return AttributeSchema(
        label_attribute="intent",
        attributes={"intent": ["positive", "negative"], "style": ["informal", "formal"]},
    )


def test_format_anchors_returns_per_value_real_examples():
    """Anchor formatter pulls real-seed examples for each requested attribute value."""
    real = [
        RealExample(text="A breathtaking piece of cinema, deeply felt.", label="positive"),
        RealExample(text="A quietly devastating film with stellar pacing.", label="positive"),
        RealExample(text="A by-the-numbers misfire that drags through every scene.", label="negative"),
        RealExample(text="Pretentious and self-indulgent; the script collapses.", label="negative"),
    ]
    v = AttributeVerifier(
        client=None,
        schema=_schema(),
        real_examples=real,
        config=VerifierConfig(k_real_per_value=2, seed=1),
    )
    block = v._format_anchors({"intent": "positive"})
    assert "Real examples of intent='positive'" in block
    # At least one of the positive seeds must appear.
    assert ("breathtaking" in block) or ("devastating" in block)
    # Negative seeds must NOT appear when only positive is requested.
    assert "by-the-numbers" not in block
    assert "Pretentious" not in block


def test_format_anchors_falls_back_when_no_real_examples():
    """No real examples available -> formatter returns explicit fallback message."""
    v = AttributeVerifier(
        client=None,
        schema=_schema(),
        real_examples=None,
        config=VerifierConfig(),
    )
    block = v._format_anchors({"intent": "positive"})
    assert "no real-distribution anchors" in block.lower()


def test_format_anchors_falls_back_when_label_not_in_seed():
    """A requested value with no anchors in the seed -> graceful empty block."""
    real = [
        RealExample(text="A perfectly fine film.", label="positive"),
    ]
    v = AttributeVerifier(
        client=None,
        schema=_schema(),
        real_examples=real,
        config=VerifierConfig(),
    )
    # Request a value that has no anchors (negative is absent from seed).
    block = v._format_anchors({"intent": "negative"})
    assert "no real-distribution anchors" in block.lower()


def test_anchors_disabled_via_config():
    """Disabling enable_real_anchors -> formatter never returns a real-example block."""
    real = [RealExample(text="A masterpiece", label="positive")]
    v = AttributeVerifier(
        client=None,
        schema=_schema(),
        real_examples=real,
        config=VerifierConfig(enable_real_anchors=False),
    )
    block = v._format_anchors({"intent": "positive"})
    assert "no real-distribution anchors" in block.lower()
    assert "masterpiece" not in block


def test_k_real_per_value_caps_the_pool():
    """k_real_per_value controls how many anchors appear in the prompt."""
    real = [RealExample(text=f"positive sample {i}", label="positive") for i in range(20)]
    v = AttributeVerifier(
        client=None,
        schema=_schema(),
        real_examples=real,
        config=VerifierConfig(k_real_per_value=3, seed=42),
    )
    block = v._format_anchors({"intent": "positive"})
    # Bullet lines count - should be exactly 3.
    n_bullets = sum(1 for line in block.splitlines() if line.strip().startswith("- "))
    assert n_bullets == 3
