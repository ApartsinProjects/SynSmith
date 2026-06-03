"""Realism Discriminator.

Implements the GAN-style adversary. Given a shuffled batch of real and
synthetic samples, the judge classifies each as real or synthetic and
reports the cues it used. The discriminator's accuracy is the primary
realism signal: a healthy SynSmith run drives it toward 0.5.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from synsmith.llm import LLMClient, json_chat
from synsmith.prompts import DISCRIMINATOR_SYSTEM, DISCRIMINATOR_USER_TEMPLATE
from synsmith.schema import RealExample, RealismVerdict, SyntheticSample


@dataclass
class DiscriminatorConfig:
    """Knobs for the realism discriminator.

    ``max_samples`` caps how many items go into a single judging call so the
    prompt stays inside model context. Larger batches give better calibrated
    confidence but cost more tokens per round.

    Fix F1 (v2.10.1): ``use_fixed_real_anchor`` switches the per-iteration
    real-sample subset from random-per-call to a FIXED stratified anchor
    cached at first call. The default-on behaviour eliminates a calibration
    bug where the discriminator's per-iteration random subsample of real
    examples drifted the perceived real distribution (e.g. one iter sees
    a more-formal subset and reports 'real = formal', the next iter sees
    a more-colloquial subset and reports 'real = direct'), feeding the
    updater contradictory register signals that pushed the generator off
    the actual target register.
    """

    max_samples: int = 24
    temperature: float = 0.0
    seed: int | None = None
    use_fixed_real_anchor: bool = True
    """When True, pick a FIXED stratified real-anchor subset once and reuse
    it for every judging call. When False, restore the v2.9.x behaviour of
    random per-call sampling (kept for backward-compat reproducibility)."""
    real_anchor_size: int = 12
    """How many real samples to include in the fixed anchor. The remaining
    ``max_samples - real_anchor_size`` slots are filled with synthetics."""


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
        self._fixed_real_anchor: list[tuple[str, str]] | None = None
        # F7: hybrid anchor = F1 train-stratified + Coverage Hole top-K. Populated
        # by ingest_coverage_hole_anchor() after iter_0. When set, takes
        # precedence over _fixed_real_anchor.
        self._hybrid_anchor: list[tuple[str, str]] | None = None

    def ingest_coverage_hole_anchor(
        self,
        coverage_hole_texts: list[str],
        frac: float = 0.5,
    ) -> None:
        """Fix F7: augment the Discriminator's anchor with Coverage Hole exemplars.

        Called by the loop after iter_0 with the Coverage Hole Finder's top-K
        real-sample texts (the framework's own signal of which real samples
        are MOST DISTINCT from the current synth distribution). Replaces
        ``frac`` of the F1 anchor with these exemplars; the rest stays from
        the F1 class-stratified anchor.

        Rationale: F1 stabilises the Discriminator across iterations but
        anchors on the train distribution, which may be slightly off the
        test register. Coverage Hole adapts to the synth-vs-real gap and
        surfaces the real samples whose register is the LEAST captured by
        synth. Mixing the two anchors gives the Discriminator both a stable
        view of the train distribution AND a focused view of the under-
        covered modes.

        No-op when:
        - F1 anchor not yet populated (judge() not yet called)
        - coverage_hole_texts is empty
        - frac is 0 (effectively disables F7)
        """
        if not self.config.use_fixed_real_anchor:
            return  # F1 disabled; F7 has nothing to augment
        if self._fixed_real_anchor is None or not coverage_hole_texts or frac <= 0:
            return
        n_total = len(self._fixed_real_anchor)
        n_replace = max(1, int(n_total * min(1.0, frac)))
        # Take the first n_replace coverage-hole exemplars, tag them so
        # debug output distinguishes them from F1 anchor IDs.
        ch_anchor: list[tuple[str, str]] = [
            (f"H{i:03d}", text) for i, text in enumerate(coverage_hole_texts[:n_replace])
        ]
        # Keep the LAST (n_total - n_replace) F1 anchor entries so the F1
        # stratification across classes is preserved as much as possible.
        kept_f1 = self._fixed_real_anchor[n_replace:]
        self._hybrid_anchor = ch_anchor + kept_f1

    def _pick_real_anchor(
        self, real: list[RealExample]
    ) -> list[tuple[str, str]]:
        """Pick a FIXED stratified real-anchor subset once.

        Fix F1: cached at first call and reused for every subsequent judge()
        call. Stratified by ``RealExample.label`` when labels are present,
        so the perceived real distribution does not drift iter-to-iter
        when the per-call random subset happens to over-represent one
        class. When no labels are present (or fewer real examples than
        ``real_anchor_size``), falls back to a deterministic shuffle.
        """
        n_target = max(1, self.config.real_anchor_size)
        if not real or len(real) <= n_target:
            return [(f"R{i:03d}", ex.text) for i, ex in enumerate(real)]
        # Stratify by label when available.
        by_label: dict[str, list[tuple[int, RealExample]]] = {}
        for idx, ex in enumerate(real):
            by_label.setdefault(ex.label or "_", []).append((idx, ex))
        labels = sorted(by_label.keys())
        rng = random.Random(self.config.seed)
        # Compute per-label quota (ceil-distribute) so every class gets at
        # least one anchor when possible.
        per_label = max(1, -(-n_target // len(labels)))
        picks: list[tuple[int, RealExample]] = []
        for lbl in labels:
            pool = by_label[lbl][:]
            rng.shuffle(pool)
            picks.extend(pool[:per_label])
        # If overshoot (rare), trim deterministically.
        picks.sort(key=lambda kv: kv[0])
        picks = picks[:n_target]
        return [(f"R{idx:03d}", ex.text) for idx, ex in picks]

    def judge(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
    ) -> DiscriminationResult:
        """Mix, shuffle, judge, and score.

        Fix F1: when ``config.use_fixed_real_anchor`` (default True), the
        real-sample subset is the FIXED stratified anchor picked once at
        first call; only the synthetic samples vary iteration-to-iteration.
        Eliminates the per-iter "real-distribution drift" bug that pushed
        the updater toward whichever register the iter's random real
        subsample happened to over-represent.
        """
        labels: dict[str, str] = {}
        mixed: list[tuple[str, str]] = []
        if self.config.use_fixed_real_anchor:
            if self._fixed_real_anchor is None:
                self._fixed_real_anchor = self._pick_real_anchor(real)
            # F7: prefer the hybrid anchor (F1 + Coverage Hole top-K) when set.
            anchor = self._hybrid_anchor or self._fixed_real_anchor
            for sid, text in anchor:
                labels[sid] = "real"
                mixed.append((sid, text))
        else:
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
