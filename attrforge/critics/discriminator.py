"""Realism Discriminator.

Implements the GAN-style adversary. Given a shuffled batch of real and
synthetic samples, the judge classifies each as real or synthetic and
reports the cues it used. The discriminator's accuracy is the primary
realism signal: a healthy AttrForge run drives it toward 0.5.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from attrforge.llm import LLMClient, json_chat
from attrforge.prompts import DISCRIMINATOR_SYSTEM, DISCRIMINATOR_USER_TEMPLATE
from attrforge.schema import RealExample, RealismVerdict, SyntheticSample


@dataclass
class DiscriminatorConfig:
    """Knobs for the realism discriminator.

    ``max_samples`` caps how many items go into a single judging call so the
    prompt stays inside model context. Larger batches give better calibrated
    confidence but cost more tokens per round.
    """

    max_samples: int = 24
    temperature: float = 0.0
    seed: int | None = None


@dataclass
class DiscriminationResult:
    verdicts: list[RealismVerdict]
    labels: dict[str, str]
    accuracy: float
    synthetic_detection_rate: float


class RealismDiscriminator:
    def __init__(
        self,
        client: LLMClient,
        config: DiscriminatorConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or DiscriminatorConfig()
        self._rng = random.Random(self.config.seed)

    def judge(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
    ) -> DiscriminationResult:
        """Mix, shuffle, judge, and score."""
        labels: dict[str, str] = {}
        mixed: list[tuple[str, str]] = []
        for i, ex in enumerate(real):
            sid = f"R{i:03d}"
            labels[sid] = "real"
            mixed.append((sid, ex.text))
        for s in synthetic:
            labels[s.sample_id] = "synthetic"
            mixed.append((s.sample_id, s.text))

        self._rng.shuffle(mixed)
        mixed = mixed[: self.config.max_samples]

        samples_block = "\n\n".join(
            f"[sample_id: {sid}]\n{text}" for sid, text in mixed
        )
        user_msg = DISCRIMINATOR_USER_TEMPLATE.format(
            n=len(mixed),
            samples_block=samples_block,
        )
        arr = json_chat(
            self.client,
            DISCRIMINATOR_SYSTEM,
            [{"role": "user", "content": user_msg}],
            temperature=self.config.temperature,
            max_tokens=1200,
            retries=1,
        )
        if not isinstance(arr, list):
            arr = []
        verdicts = [self._coerce(v) for v in arr if isinstance(v, dict)]

        correct = 0
        synth_total = 0
        synth_caught = 0
        for v in verdicts:
            gold = labels.get(v.sample_id)
            if gold is None:
                continue
            if v.prediction == gold:
                correct += 1
            if gold == "synthetic":
                synth_total += 1
                if v.prediction == "synthetic":
                    synth_caught += 1
        acc = correct / max(1, len([v for v in verdicts if v.sample_id in labels]))
        sdr = synth_caught / max(1, synth_total)
        return DiscriminationResult(
            verdicts=verdicts,
            labels=labels,
            accuracy=acc,
            synthetic_detection_rate=sdr,
        )

    def _coerce(self, obj: dict) -> RealismVerdict:
        return RealismVerdict(
            sample_id=str(obj.get("sample_id", "")),
            prediction=str(obj.get("prediction", "synthetic")).lower(),
            confidence=float(obj.get("confidence", 0.0) or 0.0),
            reason=str(obj.get("reason", "")),
        )
