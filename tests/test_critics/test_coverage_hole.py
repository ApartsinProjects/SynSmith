"""Unit tests for the Coverage Hole Finder.

The Coverage Hole Finder fits a TF-IDF + logistic regression density-ratio
classifier on a balanced real/synthetic set, then returns the top-K real
exemplars the classifier is most confident about (the modes the synthetic
distribution has not covered).

These tests assert:

1. With disjoint vocabularies, the classifier discriminates well; AUROC > 0.5.
2. With too few samples, the finder returns an empty result with AUROC = 0.5.
3. The result includes a configurable number of holes.
"""
from __future__ import annotations

from attrforge.critics.coverage_hole import CoverageHoleConfig, CoverageHoleFinder
from attrforge.schema import RealExample, SyntheticSample


def _real(text: str) -> RealExample:
    return RealExample(text=text, label="general_question")


def _synth(text: str, idx: int = 0) -> SyntheticSample:
    return SyntheticSample(
        sample_id=f"s{idx}",
        text=text,
        requested_attributes={"intent": "general_question"},
        generated_attributes={"intent": "general_question"},
        prompt_version=1,
        iteration=0,
    )


def test_disjoint_vocabularies_produce_holes():
    """Disjoint vocabularies -> classifier confident -> holes surface."""
    real_samples = [
        _real("apple orange banana grape kiwi"),
        _real("strawberry peach mango papaya guava"),
        _real("blueberry raspberry blackberry currant"),
        _real("watermelon cantaloupe honeydew melon"),
        _real("plum cherry apricot nectarine pear"),
    ]
    synth_samples = [
        _synth("wrench hammer screwdriver pliers", 0),
        _synth("drill saw chisel mallet level", 1),
        _synth("nail screw bolt nut washer", 2),
        _synth("paint brush roller tape primer", 3),
        _synth("ladder shelf bracket clamp anchor", 4),
    ]
    finder = CoverageHoleFinder(CoverageHoleConfig(top_k=3))
    result = finder.find(real_samples, synth_samples)
    assert 0.5 < result.classifier_auroc <= 1.0
    assert len(result.holes) == 3


def test_too_few_samples_returns_empty():
    """Below min_real / min_synth -> no holes, AUROC = 0.5."""
    finder = CoverageHoleFinder(CoverageHoleConfig(min_real=5, min_synth=5))
    result = finder.find([_real("only one real")], [_synth("only one synth")])
    assert len(result.holes) == 0
    assert result.classifier_auroc == 0.5
    assert "not enough" in result.notes.lower()


def test_top_k_limits_hole_count():
    """The result.holes length never exceeds top_k."""
    real_samples = [_real(f"real text number {i} apple orange banana") for i in range(20)]
    synth_samples = [_synth(f"synthetic widget {i} hammer wrench drill", i) for i in range(20)]
    finder = CoverageHoleFinder(CoverageHoleConfig(top_k=5))
    result = finder.find(real_samples, synth_samples)
    assert len(result.holes) <= 5
