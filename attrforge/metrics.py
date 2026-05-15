"""Lightweight metrics that summarize a single iteration.

The loop computes these every round and stores them in the run manifest
so a researcher can plot fidelity/realism/diversity curves over iterations
without re-reading the raw samples.
"""
from __future__ import annotations

import math
from collections import Counter

from attrforge.schema import (
    AttributeSchema,
    AttributeVerdict,
    DiversityReport,
    SyntheticSample,
)


def attribute_match_rate(verdicts: list[AttributeVerdict]) -> float:
    """Fraction of samples where every requested attribute matched."""
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.attribute_match) / len(verdicts)


def per_attribute_failure_rate(
    schema: AttributeSchema, verdicts: list[AttributeVerdict]
) -> dict[str, float]:
    """For each attribute, fraction of verdicts where it appeared in ``failed_attributes``."""
    n = max(1, len(verdicts))
    counts = Counter()
    for v in verdicts:
        for a in v.failed_attributes:
            counts[a] += 1
    return {name: counts[name] / n for name in schema.names()}


def attribute_entropy(
    schema: AttributeSchema, samples: list[SyntheticSample]
) -> dict[str, float]:
    """Per-attribute Shannon entropy of observed values.

    Normalized to ``log(|allowed values|)`` so 1.0 = perfectly uniform
    across the schema's allowed values and 0.0 = degenerate.
    """
    out: dict[str, float] = {}
    for name, allowed in schema.attributes.items():
        observed = [s.requested_attributes.get(name) for s in samples]
        observed = [v for v in observed if v is not None]
        if not observed:
            out[name] = 0.0
            continue
        n = len(observed)
        counts = Counter(observed)
        h = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
        denom = math.log(len(allowed)) if len(allowed) > 1 else 1.0
        out[name] = h / denom
    return out


def combination_coverage(
    schema: AttributeSchema, samples: list[SyntheticSample]
) -> float:
    """Fraction of (label, difficulty) cells that have at least one sample.

    Uses the first two schema attributes when ``label_attribute`` and a
    secondary attribute exist, otherwise returns the marginal coverage of
    the label attribute.
    """
    names = schema.names()
    if len(names) < 2:
        return 0.0
    a, b = schema.label_attribute, next(n for n in names if n != schema.label_attribute)
    grid = {(va, vb) for va in schema.values(a) for vb in schema.values(b)}
    seen = {
        (s.requested_attributes.get(a), s.requested_attributes.get(b))
        for s in samples
    }
    seen = {pair for pair in seen if all(pair)}
    return len(seen & grid) / max(1, len(grid))


def iteration_metrics(
    schema: AttributeSchema,
    samples: list[SyntheticSample],
    attribute_verdicts: list[AttributeVerdict],
    discriminator_accuracy: float,
    synthetic_detection_rate: float,
    diversity: DiversityReport,
) -> dict[str, float]:
    """Roll up every per-iteration scalar metric into one dict for the manifest."""
    metrics = {
        "attribute_match_rate": attribute_match_rate(attribute_verdicts),
        "discriminator_accuracy": discriminator_accuracy,
        "synthetic_detection_rate": synthetic_detection_rate,
        "near_duplicate_rate": diversity.near_duplicate_rate,
        "combination_coverage": combination_coverage(schema, samples),
    }
    for name, h in attribute_entropy(schema, samples).items():
        metrics[f"entropy/{name}"] = h
    for name, frac in diversity.coverage.items():
        metrics[f"coverage/{name}"] = frac
    return metrics
