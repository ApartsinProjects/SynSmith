"""Typed data models that flow through the AttrForge pipeline.

Everything that crosses a component boundary, target attribute vectors,
synthetic samples, critic verdicts, and feedback bundles, is a Pydantic
model so we get free validation, serialization, and JSON Schema generation
for structured LLM outputs.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AttributeSchema(BaseModel):
    """Defines the controllable attribute space for a dataset.

    The schema is a mapping from attribute name (e.g. "difficulty") to the
    allowed discrete values (e.g. ["easy", "medium", "hard"]). It also
    optionally encodes:

    * ``label_attribute``: which attribute is the supervised target.
    * ``invalid_combinations``: attribute value pairs that should never
      be sampled together (e.g. ``difficulty=easy`` with ``ambiguity=high``).
    """

    model_config = ConfigDict(extra="forbid")

    attributes: dict[str, list[str]] = Field(
        ..., description="Attribute name to list of allowed values."
    )
    label_attribute: str = Field(
        ..., description="Name of the attribute that acts as the supervised label."
    )
    invalid_combinations: list[dict[str, str]] = Field(
        default_factory=list,
        description="Each entry is a partial assignment that must not appear.",
    )
    task_description: str = Field(
        default="",
        description="Short natural language description of the supervised task.",
    )
    domain: str = Field(
        default="generic",
        description="Domain tag used in prompts and logs (e.g. 'customer_support').",
    )

    @field_validator("attributes")
    @classmethod
    def _attributes_not_empty(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        if not v:
            raise ValueError("attributes must not be empty")
        for name, values in v.items():
            if not values:
                raise ValueError(f"attribute '{name}' must define at least one value")
            if len(set(values)) != len(values):
                raise ValueError(f"attribute '{name}' has duplicate values")
        return v

    def names(self) -> list[str]:
        return list(self.attributes.keys())

    def values(self, name: str) -> list[str]:
        return list(self.attributes[name])

    def is_valid(self, vector: dict[str, str]) -> bool:
        """Return True if ``vector`` respects schema membership and constraints."""
        for k, v in vector.items():
            if k not in self.attributes:
                return False
            if v not in self.attributes[k]:
                return False
        for bad in self.invalid_combinations:
            if all(vector.get(k) == v for k, v in bad.items()):
                return False
        return True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AttributeSchema":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


class AttributeVector(BaseModel):
    """A concrete target attribute assignment for one sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    values: dict[str, str]

    def get(self, attr: str) -> str | None:
        return self.values.get(attr)

    def as_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, **self.values}


class RealExample(BaseModel):
    """A real example used for few-shot prompting or discriminator comparison."""

    model_config = ConfigDict(extra="forbid")

    text: str
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyntheticSample(BaseModel):
    """A sample produced by the generator, with its requested attributes."""

    model_config = ConfigDict(extra="allow")

    sample_id: str
    text: str
    requested_attributes: dict[str, str]
    generated_attributes: dict[str, str] = Field(default_factory=dict)
    prompt_version: int = 0
    iteration: int = 0

    @property
    def label(self) -> str | None:
        return self.generated_attributes.get(
            "label"
        ) or self.requested_attributes.get("label")


class AttributeVerdict(BaseModel):
    """Output of the attribute verifier for a single sample."""

    sample_id: str
    attribute_match: bool
    failed_attributes: list[str] = Field(default_factory=list)
    reason: str = ""


class RealismVerdict(BaseModel):
    """Output of the realism discriminator for a single sample."""

    sample_id: str
    prediction: str = Field(..., description="'real' or 'synthetic'.")
    confidence: float = 0.0
    reason: str = ""


class DiversityReport(BaseModel):
    """Output of the diversity auditor over a full batch."""

    summary: str = ""
    missing_modes: list[str] = Field(default_factory=list)
    overrepresented_modes: list[str] = Field(default_factory=list)
    near_duplicate_rate: float = 0.0
    recommendations: list[str] = Field(default_factory=list)
    coverage: dict[str, float] = Field(
        default_factory=dict,
        description="Per attribute coverage as fraction of allowed values observed.",
    )


class IterationFeedback(BaseModel):
    """Aggregated feedback consumed by the prompt updater."""

    iteration: int
    attribute_failures: list[AttributeVerdict] = Field(default_factory=list)
    realism_artifacts: list[RealismVerdict] = Field(default_factory=list)
    diversity: DiversityReport = Field(default_factory=DiversityReport)
    metrics: dict[str, float] = Field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    """Write an iterable of dicts or Pydantic models to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                f.write(row.model_dump_json())
            else:
                f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
