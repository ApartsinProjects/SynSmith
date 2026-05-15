"""Generator: turns target attribute vectors into synthetic samples."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import yaml

from attrforge.llm import LLMClient, json_chat
from attrforge.prompts import GENERATOR_SYSTEM, GENERATOR_USER_TEMPLATE
from attrforge.schema import (
    AttributeSchema,
    AttributeVector,
    RealExample,
    SyntheticSample,
)


@dataclass
class GeneratorConfig:
    num_few_shot: int = 3
    temperature: float = 0.9
    max_tokens: int = 800
    seed: int | None = None


class Generator:
    """Run the current generator prompt against each target attribute vector."""

    def __init__(
        self,
        client: LLMClient,
        schema: AttributeSchema,
        real_examples: list[RealExample],
        config: GeneratorConfig | None = None,
    ) -> None:
        self.client = client
        self.schema = schema
        self.real_examples = real_examples
        self.config = config or GeneratorConfig()
        self._rng = random.Random(self.config.seed)

    def generate(
        self,
        targets: list[AttributeVector],
        *,
        prompt: str,
        prompt_version: int,
        iteration: int,
    ) -> list[SyntheticSample]:
        """Run the generator for every target vector. One LLM call per sample.

        We keep it sample-by-sample rather than batched because critics need
        per-sample IDs and because per-sample temperature variance gives us
        more diverse outputs than asking for ``k`` examples in one shot.
        """
        schema_str = yaml.safe_dump(self.schema.attributes, sort_keys=False)
        out: list[SyntheticSample] = []
        for target in targets:
            few_shot = self._format_few_shot()
            user_msg = GENERATOR_USER_TEMPLATE.format(
                generator_prompt=prompt,
                task_description=self.schema.task_description or "(none)",
                domain=self.schema.domain,
                attribute_schema=schema_str,
                few_shot_real_examples=few_shot,
                target_attribute_vector=json.dumps(target.values, indent=2),
                sample_id=target.sample_id,
            )
            obj = json_chat(
                self.client,
                GENERATOR_SYSTEM,
                [{"role": "user", "content": user_msg}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                retries=1,
            )
            sample = self._coerce(obj, target, prompt_version, iteration)
            out.append(sample)
        return out

    def _format_few_shot(self) -> str:
        if not self.real_examples:
            return "(no real examples provided)"
        n = min(self.config.num_few_shot, len(self.real_examples))
        sample = self._rng.sample(self.real_examples, n)
        return "\n".join(
            f"- (label={ex.label or '?'}) {ex.text}" for ex in sample
        )

    def _coerce(
        self,
        obj: dict,
        target: AttributeVector,
        prompt_version: int,
        iteration: int,
    ) -> SyntheticSample:
        """Defensively turn the model's JSON into a SyntheticSample."""
        text = obj.get("text") or obj.get("example") or ""
        attrs = obj.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        attrs = {k: str(v) for k, v in attrs.items()}
        return SyntheticSample(
            sample_id=str(obj.get("sample_id") or target.sample_id),
            text=str(text).strip(),
            requested_attributes=target.values,
            generated_attributes=attrs,
            prompt_version=prompt_version,
            iteration=iteration,
        )
