"""Pack Discriminator: a PacGAN-style adversary for mode collapse.

The standard realism discriminator judges samples one at a time. Mode
collapse is invisible to it: every collapsed sample can individually look
real. PacGAN's insight is to show the judge ``k`` samples concatenated and
ask whether the pack reads like a slice of a human corpus or a fan of
LLM regenerations.

We construct M pairs of packs. Each pair contains ``k`` real samples and
``k`` synthetic samples (separately, randomized order across pairs), and
the judge is asked which pack is human-written. The judge sees pack-level
patterns (phrase repetition, structural repetition, length and rhythm
sameness) that no per-sample audit reveals.

The single scalar ``pack_accuracy`` is the diversity-collapse signal:
0.5 means the packs are indistinguishable, 1.0 means the synthetic pack
is trivially detectable as homogeneous.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from pydantic import BaseModel, Field

from synsmith.llm import LLMClient, json_chat
from synsmith.schema import RealExample, SyntheticSample


PACK_SYSTEM = (
    "You compare two packs of short texts. Pack A and Pack B each contain "
    "several samples. One pack comes from a human-written corpus; the other "
    "pack comes from a single language model that produced variations on a "
    "theme. Decide which pack is the LLM pack. Look for repeated openers, "
    "repeated structures, repeated rhythms, vocabulary plateaus, and "
    "stylistic tics that appear across multiple samples in the same pack.\n\n"
    "CRITICAL (Fix F2 / real-distribution gate): the shared_patterns field "
    "must list ONLY patterns that distinguish the LLM pack FROM the real "
    "pack. If a pattern appears in BOTH packs (e.g. 'short, direct sentences' "
    "when both packs contain short direct sentences; 'informal tone with "
    "minimal punctuation' when both packs are informal customer-support "
    "texts; 'use of question marks' when both packs are inquiries), DO NOT "
    "report it as a shared_pattern. That pattern characterises the target "
    "distribution, not a synthesis artefact, and the updater would interpret "
    "it as 'avoid the target register'.\n\n"
    "Before listing each candidate pattern, check it against the real pack: "
    "if the real pack samples ALSO exhibit the pattern, drop it. Only report "
    "patterns that are systematically present in the LLM pack and "
    "systematically absent (or rare) in the real pack. Return JSON only."
)

PACK_USER_TEMPLATE = """Pack A ({n_a} samples):
{pack_a}

Pack B ({n_b} samples):
{pack_b}

Output JSON:
{{
  "llm_pack": "A" or "B",
  "confidence": 0.0 to 1.0,
  "shared_patterns": ["<concrete phrase or structural pattern that appears in the LLM pack but NOT in the real pack>", ...]
}}

Reminder: shared_patterns are LLM-pack-EXCLUSIVE patterns. If a pattern is
present in both the LLM pack and the real pack (i.e. it characterises the
target distribution rather than an LLM artefact), do NOT list it. The
updater will treat any listed pattern as something to suppress, so listing
target-distribution patterns would corrupt the loop.
"""


@dataclass
class PackDiscriminatorConfig:
    pack_size: int = 5
    n_comparisons: int = 6
    temperature: float = 0.0
    seed: int | None = None


class PackArtifact(BaseModel):
    pattern: str
    n_pairs_observed: int = 1


class PackResult(BaseModel):
    pack_accuracy: float = Field(
        ..., description="Fraction of pair comparisons where the LLM pack was correctly identified."
    )
    confidence_mean: float = 0.0
    n_comparisons: int = 0
    shared_patterns: list[PackArtifact] = Field(default_factory=list)


class PackDiscriminator:
    """PacGAN-style adversary for SynSmith.

    The pack-accuracy scalar is independent of any per-sample realism
    judgment, so it can be plotted as a separate curve alongside
    ``discriminator_accuracy``. A healthy diversifying loop drives BOTH
    toward 0.5; a loop where individual realism keeps improving but pack
    accuracy stays at 1.0 has silently mode-collapsed.
    """

    def __init__(
        self,
        client: LLMClient,
        config: PackDiscriminatorConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or PackDiscriminatorConfig()
        self._rng = random.Random(self.config.seed)

    def attack(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
    ) -> PackResult:
        k = self.config.pack_size
        if len(real) < k or len(synthetic) < k:
            return PackResult(pack_accuracy=0.5, n_comparisons=0)

        comparisons: list[tuple[bool, float, list[str]]] = []
        for _ in range(self.config.n_comparisons):
            real_pack = self._rng.sample(real, k)
            synth_pack = self._rng.sample(synthetic, k)
            llm_is_a = self._rng.random() < 0.5
            pack_a = synth_pack if llm_is_a else real_pack
            pack_b = real_pack if llm_is_a else synth_pack
            verdict = self._judge_pair(pack_a, pack_b)
            if verdict is None:
                continue
            chose, conf, patterns = verdict
            correct = (chose == "A" and llm_is_a) or (chose == "B" and not llm_is_a)
            comparisons.append((correct, conf, patterns))

        if not comparisons:
            return PackResult(pack_accuracy=0.5, n_comparisons=0)

        acc = sum(1 for c, _, _ in comparisons if c) / len(comparisons)
        conf_mean = sum(conf for _, conf, _ in comparisons) / len(comparisons)
        merged = self._merge_patterns([ps for _, _, ps in comparisons])
        return PackResult(
            pack_accuracy=acc,
            confidence_mean=conf_mean,
            n_comparisons=len(comparisons),
            shared_patterns=merged,
        )

    def _judge_pair(
        self,
        pack_a: list,
        pack_b: list,
    ) -> tuple[str, float, list[str]] | None:
        def fmt(pack: list) -> str:
            return "\n".join(
                f"  {i + 1}. {self._text(item)}" for i, item in enumerate(pack)
            )

        user_msg = PACK_USER_TEMPLATE.format(
            n_a=len(pack_a),
            n_b=len(pack_b),
            pack_a=fmt(pack_a),
            pack_b=fmt(pack_b),
        )
        try:
            obj = json_chat(
                self.client,
                PACK_SYSTEM,
                [{"role": "user", "content": user_msg}],
                temperature=self.config.temperature,
                max_tokens=400,
                retries=1,
            )
        except Exception:
            return None
        chose = str(obj.get("llm_pack", "")).strip().upper()
        if chose not in ("A", "B"):
            return None
        conf = float(obj.get("confidence", 0.5) or 0.5)
        patterns = [str(p) for p in obj.get("shared_patterns", [])]
        return chose, conf, patterns

    @staticmethod
    def _text(item) -> str:
        if hasattr(item, "text"):
            return item.text[:240]
        return str(item)[:240]

    @staticmethod
    def _merge_patterns(pattern_lists: list[list[str]]) -> list[PackArtifact]:
        """Count how many pair comparisons surfaced each pattern (case-folded)."""
        counts: dict[str, int] = {}
        for plist in pattern_lists:
            for p in plist:
                key = p.strip().lower()
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
        return [
            PackArtifact(pattern=k, n_pairs_observed=v)
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ]
