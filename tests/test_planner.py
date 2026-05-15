"""Attribute planner contract tests."""
from __future__ import annotations

from attrforge.planner import AttributePlanner, PlannerConfig
from attrforge.schema import AttributeSchema


def _schema() -> AttributeSchema:
    return AttributeSchema(
        attributes={
            "label": ["a", "b", "c"],
            "difficulty": ["easy", "medium", "hard"],
            "style": ["formal", "informal"],
        },
        label_attribute="label",
        invalid_combinations=[],
    )


def test_stratified_produces_valid_vectors():
    s = _schema()
    p = AttributePlanner(s, PlannerConfig(strategy="stratified", batch_size=12, seed=0))
    out = p.plan(12)
    assert len(out) == 12
    for vec in out:
        assert s.is_valid(vec.values)


def test_targeted_combinations_are_respected():
    s = _schema()
    p = AttributePlanner(s, PlannerConfig(strategy="stratified", batch_size=4, seed=0))
    out = p.plan(
        4,
        targeted_combinations=[{"label": "c", "difficulty": "hard"}],
    )
    assert out[0].values["label"] == "c"
    assert out[0].values["difficulty"] == "hard"


def test_coverage_gap_distributes_attention():
    s = _schema()
    p = AttributePlanner(s, PlannerConfig(strategy="coverage_gap", batch_size=20, seed=0))
    out = p.plan(20)
    labels = [v.values["label"] for v in out]
    # All three labels appear at least once.
    assert set(labels) == {"a", "b", "c"}
