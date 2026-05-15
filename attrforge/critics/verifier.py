"""Attribute Verifier.

For each synthetic sample, asks an LLM judge whether the text actually
reflects every requested attribute. Returns per-sample verdicts that the
prompt updater consumes verbatim.
"""
from __future__ import annotations

import json

import yaml

from attrforge.llm import LLMClient, json_chat
from attrforge.prompts import VERIFIER_SYSTEM, VERIFIER_USER_TEMPLATE
from attrforge.schema import AttributeSchema, AttributeVerdict, SyntheticSample


class AttributeVerifier:
    def __init__(self, client: LLMClient, schema: AttributeSchema) -> None:
        self.client = client
        self.schema = schema

    def verify(self, samples: list[SyntheticSample]) -> list[AttributeVerdict]:
        schema_str = yaml.safe_dump(self.schema.attributes, sort_keys=False)
        verdicts: list[AttributeVerdict] = []
        for sample in samples:
            user_msg = VERIFIER_USER_TEMPLATE.format(
                attribute_schema=schema_str,
                sample_id=sample.sample_id,
                requested_attributes=json.dumps(sample.requested_attributes),
                text=sample.text,
            )
            obj = json_chat(
                self.client,
                VERIFIER_SYSTEM,
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=400,
                retries=1,
            )
            verdicts.append(self._coerce(obj, sample))
        return verdicts

    def _coerce(self, obj: dict, sample: SyntheticSample) -> AttributeVerdict:
        return AttributeVerdict(
            sample_id=str(obj.get("sample_id") or sample.sample_id),
            attribute_match=bool(obj.get("attribute_match", False)),
            failed_attributes=[str(a) for a in obj.get("failed_attributes", [])],
            reason=str(obj.get("reason", "")),
        )
