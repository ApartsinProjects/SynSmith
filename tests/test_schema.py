"""Schema model contract tests."""
from __future__ import annotations

import pytest

from attrforge.schema import AttributeSchema


def _schema() -> AttributeSchema:
    return AttributeSchema(
        attributes={
            "label": ["a", "b", "c"],
            "difficulty": ["easy", "hard"],
        },
        label_attribute="label",
        invalid_combinations=[{"label": "a", "difficulty": "hard"}],
    )


def test_schema_validates_membership():
    s = _schema()
    assert s.is_valid({"label": "a", "difficulty": "easy"})
    assert not s.is_valid({"label": "z", "difficulty": "easy"})
    assert not s.is_valid({"label": "a", "difficulty": "hard"})  # invalid combo


def test_schema_rejects_empty_attributes():
    with pytest.raises(ValueError):
        AttributeSchema(attributes={}, label_attribute="label")


def test_schema_rejects_duplicate_values():
    with pytest.raises(ValueError):
        AttributeSchema(
            attributes={"label": ["a", "a"]},
            label_attribute="label",
        )


def test_names_and_values():
    s = _schema()
    assert "label" in s.names()
    assert s.values("difficulty") == ["easy", "hard"]
