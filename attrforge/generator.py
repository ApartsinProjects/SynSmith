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
    # Verbalized Sampling: ask the model for k candidates with self-reported
    # probabilities, then sample one by that distribution (scout D1.1,
    # arXiv:2510.01171). Increases lexical+semantic diversity without
    # changing the critic stack.
    verbalized_sampling: bool = False
    vs_n_candidates: int = 5
    vs_sample_strategy: str = "weighted"  # "weighted" | "top1" | "uniform"


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

        Verbalized Sampling (config.verbalized_sampling=True) asks the
        generator for k candidates with self-reported probabilities per call
        and samples one by that distribution. The total number of synthesis
        LLM calls is unchanged (still one call per target); the model emits
        a richer per-call output that the post-call sampler decodes.
        """
        schema_str = yaml.safe_dump(self.schema.attributes, sort_keys=False)
        out: list[SyntheticSample] = []
        for target in targets:
            few_shot = self._format_few_shot()
            if self.config.verbalized_sampling:
                obj = self._verbalized_sampling_call(
                    prompt=prompt,
                    schema_str=schema_str,
                    few_shot=few_shot,
                    target=target,
                )
            else:
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

    def _verbalized_sampling_call(
        self,
        *,
        prompt: str,
        schema_str: str,
        few_shot: str,
        target: AttributeVector,
    ) -> dict:
        """One Verbalized-Sampling LLM call -> one chosen candidate.

        Asks the model for ``vs_n_candidates`` candidate utterances with
        self-reported probabilities, then samples one by the configured
        strategy. The selected candidate is returned in the same JSON shape
        the standard generator expects (text + attributes), so the rest
        of the pipeline is unchanged.
        """
        vs_user_msg = (
            "You are generating ONE synthetic example. To break the typicality "
            "bias that produces near-identical outputs across calls, first "
            f"propose {self.config.vs_n_candidates} CANDIDATE utterances and "
            "score each with the probability you would have assigned it under "
            "free generation. Return a JSON object:\n"
            '{\n'
            '  "sample_id": "<the requested id>",\n'
            '  "candidates": [\n'
            '    {"text": "<utterance 1>", "probability": <0-1>, "attributes": {...}},\n'
            '    {"text": "<utterance 2>", "probability": <0-1>, "attributes": {...}},\n'
            '    ...\n'
            '  ]\n'
            '}\n\n'
            "Probabilities should sum to roughly 1.0. The candidates should "
            "differ in surface form, structure, and length, not just paraphrase "
            "one underlying sentence. Each candidate must satisfy the "
            "requested attribute vector.\n\n"
            "Now apply the standard generator instruction below.\n\n"
            "===== generator prompt =====\n"
            f"{prompt}\n"
            "===== task =====\n"
            f"{self.schema.task_description or '(none)'}\n"
            f"domain: {self.schema.domain}\n"
            "===== attribute schema =====\n"
            f"{schema_str}\n"
            "===== few-shot real exemplars =====\n"
            f"{few_shot}\n"
            "===== target attribute vector =====\n"
            f"{json.dumps(target.values, indent=2)}\n"
            f"sample_id = {target.sample_id}\n"
        )
        obj = json_chat(
            self.client,
            GENERATOR_SYSTEM,
            [{"role": "user", "content": vs_user_msg}],
            temperature=self.config.temperature,
            max_tokens=max(self.config.max_tokens, 1500),
            retries=1,
        )
        cands = obj.get("candidates")
        if not isinstance(cands, list) or not cands:
            # Fallback: model ignored the verbalized-sampling instruction and
            # returned a plain {text, attributes}. Treat it as a 1-candidate
            # call.
            return obj
        # Select one candidate by configured strategy.
        weights: list[float] = []
        for c in cands:
            try:
                w = float(c.get("probability", 0.0))
            except (TypeError, ValueError):
                w = 0.0
            weights.append(max(w, 0.0))
        s = sum(weights)
        if s <= 0:
            weights = [1.0 / len(cands)] * len(cands)
        else:
            weights = [w / s for w in weights]
        if self.config.vs_sample_strategy == "top1":
            best = int(max(range(len(weights)), key=lambda i: weights[i]))
            chosen = cands[best]
        elif self.config.vs_sample_strategy == "uniform":
            chosen = cands[self._rng.randrange(len(cands))]
        else:  # weighted
            u = self._rng.random()
            cum = 0.0
            chosen = cands[-1]
            for c, w in zip(cands, weights):
                cum += w
                if u <= cum:
                    chosen = c
                    break
        return {
            "sample_id": obj.get("sample_id") or target.sample_id,
            "text": chosen.get("text", ""),
            "attributes": chosen.get("attributes") or {},
        }

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
