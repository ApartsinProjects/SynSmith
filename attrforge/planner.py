"""Attribute Planner.

Decides which attribute vectors to ask the generator to produce next.

Two strategies are provided:

* ``stratified``: marginal balance across each attribute, sampled jointly.
  Cheap, no LLM call, useful as a default.

* ``coverage_gap``: targets attribute combinations that have low coverage
  in the dataset so far. The auditor's missing-mode hints can be folded
  in directly via ``targeted_combinations``.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from attrforge.schema import AttributeSchema, AttributeVector, SyntheticSample


@dataclass
class PlannerConfig:
    strategy: str = "stratified"
    batch_size: int = 16
    seed: int | None = None


class AttributePlanner:
    """Produce target attribute vectors for the next generation batch."""

    def __init__(self, schema: AttributeSchema, config: PlannerConfig | None = None) -> None:
        self.schema = schema
        self.config = config or PlannerConfig()
        self._rng = random.Random(self.config.seed)
        self._counter = 0

    def plan(
        self,
        n: int | None = None,
        *,
        existing: list[SyntheticSample] | None = None,
        targeted_combinations: list[dict[str, str]] | None = None,
    ) -> list[AttributeVector]:
        """Return ``n`` target vectors using the configured strategy.

        ``targeted_combinations`` lets the diversity auditor seed the
        planner with under-represented modes. These vectors are emitted
        first; the remainder is filled by the configured strategy.
        """
        n = n if n is not None else self.config.batch_size
        out: list[AttributeVector] = []

        if targeted_combinations:
            for partial in targeted_combinations[:n]:
                full = self._fill_partial(partial)
                if full is not None:
                    out.append(self._wrap(full))

        if self.config.strategy == "stratified":
            out.extend(self._stratified(n - len(out)))
        elif self.config.strategy == "coverage_gap":
            out.extend(self._coverage_gap(n - len(out), existing or []))
        else:
            raise ValueError(f"Unknown planner strategy: {self.config.strategy!r}")

        return out[:n]

    def _wrap(self, values: dict[str, str]) -> AttributeVector:
        self._counter += 1
        return AttributeVector(
            sample_id=f"target_{self._counter:05d}", values=values
        )

    def _fill_partial(self, partial: dict[str, str]) -> dict[str, str] | None:
        """Fill missing attributes randomly while respecting constraints.

        Tries up to ``max_tries`` random completions; returns None if it
        cannot find a valid one (e.g. partial conflicts with every fill).
        """
        max_tries = 32
        for _ in range(max_tries):
            full = dict(partial)
            for name in self.schema.names():
                if name not in full:
                    full[name] = self._rng.choice(self.schema.values(name))
            if self.schema.is_valid(full):
                return full
        return None

    def _stratified(self, n: int) -> list[AttributeVector]:
        if n <= 0:
            return []
        out: list[AttributeVector] = []
        attempts = 0
        max_attempts = n * 10
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            values = {
                name: self._rng.choice(vals)
                for name, vals in self.schema.attributes.items()
            }
            if self.schema.is_valid(values):
                out.append(self._wrap(values))
        return out

    def _coverage_gap(
        self, n: int, existing: list[SyntheticSample]
    ) -> list[AttributeVector]:
        """Score every two-attribute combination by inverse coverage, then sample."""
        if n <= 0:
            return []
        pair_counts: dict[tuple[str, str, str, str], int] = {}
        names = self.schema.names()
        for sample in existing:
            attrs = sample.requested_attributes
            for a, b in itertools.combinations(names, 2):
                if a in attrs and b in attrs:
                    key = (a, attrs[a], b, attrs[b])
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        # Inverse-count weights, normalized; never zero so every pair stays possible.
        weights: list[tuple[tuple[str, str, str, str], float]] = []
        for a, b in itertools.combinations(names, 2):
            for va in self.schema.values(a):
                for vb in self.schema.values(b):
                    c = pair_counts.get((a, va, b, vb), 0)
                    weights.append(((a, va, b, vb), 1.0 / (1 + c)))

        out: list[AttributeVector] = []
        attempts = 0
        max_attempts = n * 20
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            (a, va, b, vb), _ = self._weighted_pick(weights)
            values = self._fill_partial({a: va, b: vb})
            if values is not None:
                out.append(self._wrap(values))
        return out

    def _weighted_pick(
        self, weights: list[tuple[tuple[str, str, str, str], float]]
    ) -> tuple[tuple[str, str, str, str], float]:
        total = sum(w for _, w in weights)
        r = self._rng.random() * total
        acc = 0.0
        for key, w in weights:
            acc += w
            if acc >= r:
                return key, w
        return weights[-1]
