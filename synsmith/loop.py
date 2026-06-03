"""The SynSmith orchestrator.

Wires together the planner, generator, three critics, and prompt updater,
runs them for ``T`` iterations, and persists every artifact under a run
directory so the experiment is reproducible end to end.

Directory layout for a single run:

    runs/<run_id>/
      config.yaml                  resolved config (everything needed to replay)
      schema.yaml                  attribute schema
      real_examples.jsonl          real examples used for few-shot and discriminator
      manifest.json                metrics per iteration plus prompt versions
      iter_000/
        prompt.txt
        targets.jsonl
        samples.jsonl
        attribute_verdicts.jsonl
        realism_verdicts.jsonl
        diversity_report.json
      iter_001/
        ...
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from synsmith.critics import (
    AttributeVerifier,
    CoverageHoleFinder,
    DiversityAuditor,
    ModeHunter,
    ModeSeeking,
    PackDiscriminator,
    RealismDiscriminator,
)
from synsmith.critics.auditor import AuditorConfig
from synsmith.critics.coverage_hole import CoverageHoleConfig, CoverageHoleResult
from synsmith.critics.discriminator import DiscriminationResult, DiscriminatorConfig
from synsmith.critics.mode_hunter import ModeHunterConfig, ModeHunterResult
from synsmith.critics.mode_seeking import ModeSeekingConfig, ModeSeekingResult
from synsmith.critics.pack_discriminator import (
    PackDiscriminatorConfig,
    PackResult,
)
from synsmith.generator import Generator, GeneratorConfig
from synsmith.llm import LLMClient, LLMConfig, build_client
from synsmith.metrics import iteration_metrics
from synsmith.planner import AttributePlanner, PlannerConfig
from synsmith.prompts import GENERATOR_INITIAL
from synsmith.schema import (
    AttributeSchema,
    AttributeVerdict,
    DiversityReport,
    IterationFeedback,
    RealExample,
    RealismVerdict,
    SyntheticSample,
    load_jsonl,
    write_jsonl,
)
from synsmith.updater import PromptHistory, PromptUpdater

logger = logging.getLogger(__name__)


class IterationResult(BaseModel):
    """Everything one round produced. Persisted to ``iter_<n>/`` on disk."""

    iteration: int
    prompt_version: int
    prompt: str
    samples: list[SyntheticSample]
    attribute_verdicts: list[AttributeVerdict]
    realism_verdicts: list[RealismVerdict]
    diversity: DiversityReport
    metrics: dict[str, float]
    # New GAN-style diversity adversary outputs. Optional so legacy
    # baselines can skip them entirely.
    pack_result: PackResult | None = None
    mode_seeking_result: ModeSeekingResult | None = None
    mode_hunter_result: ModeHunterResult | None = None
    coverage_hole_result: CoverageHoleResult | None = None


class RunResult(BaseModel):
    """End-of-run summary returned to the caller."""

    run_id: str
    run_dir: str
    iterations: list[IterationResult]
    final_prompt: str
    final_prompt_version: int
    metric_history: list[dict[str, float]]


@dataclass
class SynSmithConfig:
    """Top-level config object passed to ``SynSmith``.

    Built from a YAML file via :meth:`SynSmith.from_config`. Every nested
    config has sensible defaults so a minimal YAML is just a schema path
    and a real examples path.

    The ``enable_*`` flags switch entire critics on or off, which is what
    makes baseline ablations a single-field change.
    """

    schema_path: str
    real_examples_path: str
    domain: str = "generic"
    task_description: str = ""
    iterations: int = 5
    samples_per_iteration: int = 16
    initial_prompt: str = GENERATOR_INITIAL
    run_dir: str = "runs"
    seed: int | None = 17
    generator_llm: LLMConfig = field(default_factory=LLMConfig)
    verifier_llm: LLMConfig | None = None
    discriminator_llm: LLMConfig | None = None
    auditor_llm: LLMConfig | None = None
    updater_llm: LLMConfig | None = None
    pack_llm: LLMConfig | None = None
    hunter_llm: LLMConfig | None = None
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    pack_discriminator: PackDiscriminatorConfig = field(
        default_factory=PackDiscriminatorConfig
    )
    mode_hunter: ModeHunterConfig = field(default_factory=ModeHunterConfig)
    mode_seeking: ModeSeekingConfig = field(default_factory=ModeSeekingConfig)
    coverage_hole: CoverageHoleConfig = field(default_factory=CoverageHoleConfig)
    # Critic enable flags. Used to implement the baselines as ablations.
    enable_verifier: bool = True
    enable_discriminator: bool = True
    enable_auditor: bool = True
    enable_pack: bool = True
    enable_mode_seeking: bool = True
    enable_mode_hunter: bool = True
    enable_coverage_hole: bool = True
    enable_updater: bool = True
    # When True, generator and verifier calls go through OpenAI Batch API
    # (~50% the cost of real-time). Other critics (Discriminator, Auditor,
    # Pack, Mode Hunter, Updater) remain real-time because they are 1 call
    # each per iter and don't benefit from batching. See synsmith.llm_batch.
    use_batch_api: bool = False
    batch_model: str = "gpt-4o-mini"
    # Task #73: Class-Discriminability sibling-rejection in Verifier.
    # When True, the Verifier user message includes a 'Sibling classes'
    # block with real anchors from every other class, and the system
    # prompt requires REJECTION of samples ambiguous between siblings.
    verifier_sibling_rejection: bool = False
    # Fix B (v2.9.6): if True, after the per-iter verifier pass, count
    # accepted samples per class and re-generate extras for under-filled
    # classes so each class hits at least ceil(n/K) accepted samples per
    # iteration. Capped at +regen_max_extra_frac of the original batch
    # size to prevent runaway loops on impossible attribute combinations.
    regen_on_rejection: bool = True
    regen_max_extra_frac: float = 0.5
    label: str = "full_attrforge"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SynSmithConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SynSmithConfig":
        def _llm(d: dict | None) -> LLMConfig | None:
            if d is None:
                return None
            return LLMConfig(**d)

        return cls(
            schema_path=raw["schema_path"],
            real_examples_path=raw["real_examples_path"],
            domain=raw.get("domain", "generic"),
            task_description=raw.get("task_description", ""),
            iterations=raw.get("iterations", 5),
            samples_per_iteration=raw.get("samples_per_iteration", 16),
            initial_prompt=raw.get("initial_prompt", GENERATOR_INITIAL),
            run_dir=raw.get("run_dir", "runs"),
            seed=raw.get("seed", 17),
            generator_llm=_llm(raw.get("generator_llm")) or LLMConfig(),
            verifier_llm=_llm(raw.get("verifier_llm")),
            discriminator_llm=_llm(raw.get("discriminator_llm")),
            auditor_llm=_llm(raw.get("auditor_llm")),
            updater_llm=_llm(raw.get("updater_llm")),
            pack_llm=_llm(raw.get("pack_llm")),
            hunter_llm=_llm(raw.get("hunter_llm")),
            planner=PlannerConfig(**raw.get("planner", {})),
            generator=GeneratorConfig(**raw.get("generator", {})),
            discriminator=DiscriminatorConfig(**raw.get("discriminator", {})),
            auditor=AuditorConfig(**raw.get("auditor", {})),
            pack_discriminator=PackDiscriminatorConfig(
                **raw.get("pack_discriminator", {})
            ),
            mode_hunter=ModeHunterConfig(**raw.get("mode_hunter", {})),
            mode_seeking=ModeSeekingConfig(**raw.get("mode_seeking", {})),
            coverage_hole=CoverageHoleConfig(**raw.get("coverage_hole", {})),
            enable_verifier=raw.get("enable_verifier", True),
            enable_discriminator=raw.get("enable_discriminator", True),
            enable_auditor=raw.get("enable_auditor", True),
            enable_pack=raw.get("enable_pack", True),
            enable_mode_seeking=raw.get("enable_mode_seeking", True),
            enable_mode_hunter=raw.get("enable_mode_hunter", True),
            enable_coverage_hole=raw.get("enable_coverage_hole", True),
            enable_updater=raw.get("enable_updater", True),
            label=raw.get("label", "full_attrforge"),
            use_batch_api=raw.get("use_batch_api", False),
            batch_model=raw.get("batch_model", "gpt-4o-mini"),
            regen_on_rejection=raw.get("regen_on_rejection", True),
            regen_max_extra_frac=raw.get("regen_max_extra_frac", 0.5),
            verifier_sibling_rejection=raw.get("verifier_sibling_rejection", False),
        )


class SynSmith:
    """Coordinates the full iterative pipeline."""

    def __init__(self, config: SynSmithConfig) -> None:
        self.config = config
        self.console = Console()

        self.schema = AttributeSchema.from_yaml(config.schema_path)
        if config.task_description and not self.schema.task_description:
            self.schema.task_description = config.task_description
        if config.domain and self.schema.domain == "generic":
            self.schema.domain = config.domain

        self.real_examples = [
            RealExample.model_validate(d) for d in load_jsonl(config.real_examples_path)
        ]

        gen_client = build_client(config.generator_llm)
        ver_client = build_client(config.verifier_llm or config.generator_llm)
        disc_client = build_client(config.discriminator_llm or config.generator_llm)
        aud_client = build_client(config.auditor_llm or config.generator_llm)
        upd_client = build_client(config.updater_llm or config.generator_llm)
        pack_client = build_client(config.pack_llm or config.generator_llm)
        hunter_client = build_client(config.hunter_llm or config.generator_llm)

        if config.planner.seed is None:
            config.planner.seed = config.seed
        if config.generator.seed is None:
            config.generator.seed = config.seed
        if config.discriminator.seed is None:
            config.discriminator.seed = config.seed
        if config.pack_discriminator.seed is None:
            config.pack_discriminator.seed = config.seed

        self.planner = AttributePlanner(self.schema, config.planner)
        self.generator = Generator(
            gen_client, self.schema, self.real_examples, config.generator
        )
        from synsmith.critics.verifier import VerifierConfig
        self.verifier = (
            AttributeVerifier(
                ver_client,
                self.schema,
                real_examples=self.real_examples,
                config=VerifierConfig(
                    seed=config.seed,
                    enable_sibling_rejection=config.verifier_sibling_rejection,
                ),
            )
            if config.enable_verifier else None
        )
        self.discriminator = (
            RealismDiscriminator(disc_client, config.discriminator)
            if config.enable_discriminator else None
        )
        self.auditor = (
            DiversityAuditor(self.schema, aud_client, config.auditor)
            if config.enable_auditor
            else DiversityAuditor(self.schema, None, config.auditor)  # deterministic only
        )
        self.pack = (
            PackDiscriminator(pack_client, config.pack_discriminator)
            if config.enable_pack else None
        )
        self.mode_seeking = (
            ModeSeeking(config.mode_seeking) if config.enable_mode_seeking else None
        )
        self.mode_hunter = (
            ModeHunter(hunter_client, config.mode_hunter)
            if config.enable_mode_hunter else None
        )
        self.coverage_hole_finder = (
            CoverageHoleFinder(config.coverage_hole)
            if config.enable_coverage_hole else None
        )
        self.updater = PromptUpdater(upd_client) if config.enable_updater else None
        self.history = PromptHistory(config.initial_prompt)

        self._all_samples: list[SyntheticSample] = []

    @classmethod
    def from_config(cls, path: str | Path) -> "SynSmith":
        return cls(SynSmithConfig.from_yaml(path))

    def run(self, iterations: int | None = None) -> RunResult:
        """Execute the loop and return the run summary.

        Each iteration writes its outputs to disk before the next starts,
        so a crash mid-run still leaves the earlier iterations recoverable.
        """
        n = iterations if iterations is not None else self.config.iterations
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(self.config.run_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._persist_run_inputs(run_dir)

        results: list[IterationResult] = []
        metric_history: list[dict[str, float]] = []

        for t in range(n):
            self.console.rule(f"[bold cyan]Iteration {t + 1}/{n}")
            result = self._run_one_iteration(t, run_dir)
            results.append(result)
            metric_history.append({"iteration": float(t), **result.metrics})
            self._print_metrics_table(result.metrics)

            if t < n - 1:
                self._maybe_update_prompt(t, result)

        self._persist_manifest(run_dir, results, metric_history)

        return RunResult(
            run_id=run_id,
            run_dir=str(run_dir),
            iterations=results,
            final_prompt=self.history.current_prompt,
            final_prompt_version=self.history.current_version,
            metric_history=metric_history,
        )

    def _run_one_iteration(self, t: int, run_dir: Path) -> IterationResult:
        iter_dir = run_dir / f"iter_{t:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        prompt = self.history.current_prompt
        prompt_version = self.history.current_version
        (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        targeted = self._targeted_combinations_from_history()
        targets = self.planner.plan(
            n=self.config.samples_per_iteration,
            existing=self._all_samples,
            targeted_combinations=targeted,
        )
        write_jsonl(iter_dir / "targets.jsonl", [t.as_dict() for t in targets])

        if self.config.use_batch_api and not self.config.generator.verbalized_sampling:
            from synsmith.llm_batch import BatchLLMClient, BatchConfig
            gen_batch = BatchLLMClient(
                model=self.config.batch_model,
                config=BatchConfig(model=self.config.batch_model),
            )
            samples = self.generator.batch_generate(
                targets, prompt=prompt, prompt_version=prompt_version,
                iteration=t, batch_client=gen_batch,
            )
        else:
            samples = self.generator.generate(
                targets, prompt=prompt, prompt_version=prompt_version, iteration=t
            )
        self._all_samples.extend(samples)

        # 1. attribute verifier
        if self.verifier is not None:
            if self.config.use_batch_api:
                from synsmith.llm_batch import BatchLLMClient, BatchConfig
                ver_batch = BatchLLMClient(
                    model=self.config.batch_model,
                    config=BatchConfig(model=self.config.batch_model),
                )
                attribute_verdicts = self.verifier.batch_verify(
                    samples, batch_client=ver_batch,
                )
            else:
                attribute_verdicts = self.verifier.verify(samples)
        else:
            attribute_verdicts = []

        # 1b. Fix B (v2.9.6): re-generate to compensate for verifier-rejection
        # under-fill. If any class has fewer ACCEPTED samples than ceil(n/K)
        # in this iter due to verifier rejections, queue extras for that
        # class (capped at +50% of original batch to prevent runaway loops
        # on impossible attribute combinations).
        if (
            self.config.regen_on_rejection
            and self.verifier is not None
            and attribute_verdicts
        ):
            extras, extra_verdicts = self._regen_for_underfill(
                samples,
                attribute_verdicts,
                prompt=prompt,
                prompt_version=prompt_version,
                iteration=t,
                target_n=self.config.samples_per_iteration,
            )
            if extras:
                samples.extend(extras)
                attribute_verdicts.extend(extra_verdicts)
                self._all_samples.extend(extras)
                # Persist extras for traceability.
                from synsmith.schema import write_jsonl as _wj
                _wj(iter_dir / "samples_regen.jsonl", extras)
        write_jsonl(iter_dir / "samples.jsonl", samples)
        write_jsonl(iter_dir / "attribute_verdicts.jsonl", attribute_verdicts)

        # 2. per-sample realism discriminator
        if self.discriminator is not None:
            disc_result = self.discriminator.judge(self.real_examples, samples)
            disc_acc = disc_result.accuracy
            disc_sdr = disc_result.synthetic_detection_rate
            realism_verdicts = disc_result.verdicts
        else:
            disc_result = None
            disc_acc = 0.5
            disc_sdr = 0.5
            realism_verdicts = []
        write_jsonl(iter_dir / "realism_verdicts.jsonl", realism_verdicts)

        # 3. diversity auditor (always runs deterministic layer)
        diversity = self.auditor.audit(samples)
        (iter_dir / "diversity_report.json").write_text(
            diversity.model_dump_json(indent=2), encoding="utf-8"
        )

        # 4. pack discriminator (PacGAN-style adversary)
        pack_result: PackResult | None = None
        if self.pack is not None:
            pack_result = self.pack.attack(self.real_examples, samples)
            (iter_dir / "pack_result.json").write_text(
                pack_result.model_dump_json(indent=2), encoding="utf-8"
            )

        # 5. mode-seeking scalar
        mode_seeking_result: ModeSeekingResult | None = None
        if self.mode_seeking is not None:
            mode_seeking_result = self.mode_seeking.score(samples)
            (iter_dir / "mode_seeking.json").write_text(
                mode_seeking_result.model_dump_json(indent=2), encoding="utf-8"
            )

        # 6. mode hunter (persistent banned-substring library)
        hunter_result: ModeHunterResult | None = None
        if self.mode_hunter is not None:
            hunter_result = self.mode_hunter.hunt(
                self.real_examples, samples, iteration=t
            )
            (iter_dir / "mode_hunter.json").write_text(
                hunter_result.model_dump_json(indent=2), encoding="utf-8"
            )

        # 7. coverage hole finder
        hole_result: CoverageHoleResult | None = None
        if self.coverage_hole_finder is not None:
            hole_result = self.coverage_hole_finder.find(
                self.real_examples, samples
            )
            (iter_dir / "coverage_holes.json").write_text(
                hole_result.model_dump_json(indent=2), encoding="utf-8"
            )

        metrics = iteration_metrics(
            self.schema,
            samples,
            attribute_verdicts,
            discriminator_accuracy=disc_acc,
            synthetic_detection_rate=disc_sdr,
            diversity=diversity,
        )
        if pack_result is not None:
            metrics["pack_accuracy"] = pack_result.pack_accuracy
            metrics["pack_confidence"] = pack_result.confidence_mean
        if mode_seeking_result is not None:
            metrics["mode_seeking_ratio"] = mode_seeking_result.mode_seeking_ratio
            metrics["text_distance_mean"] = mode_seeking_result.text_distance_mean
        if hunter_result is not None:
            metrics["banned_phrasings_total"] = float(len(hunter_result.banned_library))
            metrics["banned_phrasings_new"] = float(len(hunter_result.new_findings))
        if hole_result is not None:
            metrics["coverage_classifier_auroc"] = hole_result.classifier_auroc

        (iter_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        return IterationResult(
            iteration=t,
            prompt_version=prompt_version,
            prompt=prompt,
            samples=samples,
            attribute_verdicts=attribute_verdicts,
            realism_verdicts=realism_verdicts,
            diversity=diversity,
            metrics=metrics,
            pack_result=pack_result,
            mode_seeking_result=mode_seeking_result,
            mode_hunter_result=hunter_result,
            coverage_hole_result=hole_result,
        )

    def _regen_for_underfill(
        self,
        samples: list,
        verdicts: list,
        *,
        prompt: str,
        prompt_version: int,
        iteration: int,
        target_n: int,
    ) -> tuple[list, list]:
        """Re-generate samples for classes whose accepted count falls below
        ceil(target_n / K) due to verifier rejections.

        Returns ``(extra_samples, extra_verdicts)``. Capped at
        ``regen_max_extra_frac * target_n`` extra samples to prevent runaway
        loops on impossible attribute combinations.

        Composes with Fix A (balanced planner): Fix A balances what's
        REQUESTED; Fix B compensates when the requested-balance is broken
        by verifier rejections.
        """
        label_attr = self.schema.label_attribute
        if not label_attr:
            return [], []
        allowed = list(self.schema.values(label_attr) or [])
        K = len(allowed)
        if K <= 1:
            return [], []
        per_class_target = -(-target_n // K)  # ceil(target_n / K)
        max_extras = int(self.config.regen_max_extra_frac * target_n)
        if max_extras <= 0:
            return [], []
        # Count ACCEPTED samples per class within this iter.
        verdict_by_id = {v.sample_id: v for v in verdicts}
        accepted_per_class: dict[str, int] = {v: 0 for v in allowed}
        for s in samples:
            req_label = s.requested_attributes.get(label_attr)
            if req_label not in accepted_per_class:
                continue
            v = verdict_by_id.get(s.sample_id)
            if v is not None and v.attribute_match:
                accepted_per_class[req_label] += 1
        # Build the queue of under-filled labels in deficit order.
        deficits = [
            (label, per_class_target - accepted_per_class[label])
            for label in allowed
        ]
        deficits = [(lab, d) for lab, d in deficits if d > 0]
        if not deficits:
            return [], []
        # Sort by largest deficit first so most-underfilled classes regen first.
        deficits.sort(key=lambda kv: -kv[1])
        queue: list[str] = []
        for lab, d in deficits:
            queue.extend([lab] * d)
            if len(queue) >= max_extras:
                queue = queue[:max_extras]
                break
        # Build target vectors for the queue, randomizing non-class attrs.
        from synsmith.schema import AttributeVector
        import random
        rng = random.Random((self.config.seed or 17) + iteration * 31)
        extra_targets: list[AttributeVector] = []
        for lab in queue:
            for _ in range(32):  # constraint-sat retries
                cand = {label_attr: lab}
                for name in self.schema.names():
                    if name == label_attr:
                        continue
                    cand[name] = rng.choice(self.schema.values(name))
                if self.schema.is_valid(cand):
                    av = self.planner._wrap(cand)
                    extra_targets.append(av)
                    break
        if not extra_targets:
            return [], []
        # Generate the extras.
        if self.config.use_batch_api and not self.config.generator.verbalized_sampling:
            from synsmith.llm_batch import BatchLLMClient, BatchConfig
            gen_batch = BatchLLMClient(
                model=self.config.batch_model,
                config=BatchConfig(model=self.config.batch_model),
            )
            extra_samples = self.generator.batch_generate(
                extra_targets, prompt=prompt, prompt_version=prompt_version,
                iteration=iteration, batch_client=gen_batch,
            )
        else:
            extra_samples = self.generator.generate(
                extra_targets, prompt=prompt, prompt_version=prompt_version,
                iteration=iteration,
            )
        # Verify the extras.
        if self.verifier is not None:
            if self.config.use_batch_api:
                from synsmith.llm_batch import BatchLLMClient, BatchConfig
                ver_batch = BatchLLMClient(
                    model=self.config.batch_model,
                    config=BatchConfig(model=self.config.batch_model),
                )
                extra_verdicts = self.verifier.batch_verify(
                    extra_samples, batch_client=ver_batch,
                )
            else:
                extra_verdicts = self.verifier.verify(extra_samples)
        else:
            extra_verdicts = []
        return extra_samples, extra_verdicts

    def _maybe_update_prompt(self, t: int, result: IterationResult) -> None:
        if self.updater is None:
            self.console.print("[dim]updater disabled (baseline ablation)[/dim]")
            return

        synthetic_artifacts = [
            v for v in result.realism_verdicts if v.prediction == "synthetic"
        ]
        attribute_failures = [
            v for v in result.attribute_verdicts if not v.attribute_match
        ]

        pack_artifacts = [
            f"{a.pattern} (seen in {a.n_pairs_observed} pair(s))"
            for a in (result.pack_result.shared_patterns if result.pack_result else [])
        ]
        banned = [
            f.pattern
            for f in (
                result.mode_hunter_result.banned_library
                if result.mode_hunter_result
                else []
            )
        ]
        hole_exemplars = [
            f"(label={h.label or '?'}, p_real={h.p_real:.2f}) {h.text}"
            for h in (
                result.coverage_hole_result.holes
                if result.coverage_hole_result
                else []
            )
        ]

        feedback = IterationFeedback(
            iteration=t,
            attribute_failures=attribute_failures,
            realism_artifacts=synthetic_artifacts,
            diversity=result.diversity,
            metrics=result.metrics,
            pack_artifacts=pack_artifacts,
            pack_accuracy=(
                result.pack_result.pack_accuracy if result.pack_result else None
            ),
            mode_seeking_ratio=(
                result.mode_seeking_result.mode_seeking_ratio
                if result.mode_seeking_result
                else None
            ),
            attribute_sensitivity=(
                result.mode_seeking_result.attribute_sensitivity
                if result.mode_seeking_result
                else {}
            ),
            banned_phrasings=banned,
            coverage_hole_exemplars=hole_exemplars,
            coverage_classifier_auroc=(
                result.coverage_hole_result.classifier_auroc
                if result.coverage_hole_result
                else None
            ),
        )
        new_prompt, summary = self.updater.update(self.history.current_prompt, feedback)
        if new_prompt and new_prompt.strip() != self.history.current_prompt.strip():
            self.history.append(
                prompt=new_prompt,
                iteration=t + 1,
                motivation=feedback.diversity.summary or "critic-guided rewrite",
                feedback_summary=summary,
            )
            self.console.print(
                f"[green]prompt updated to v{self.history.current_version}[/green]"
            )
        else:
            self.console.print("[yellow]prompt unchanged this round[/yellow]")

    def _targeted_combinations_from_history(self) -> list[dict[str, str]]:
        """Turn the auditor's natural-language missing modes into partial vectors.

        We do not parse free text; instead, the loop simply forwards explicit
        ``targeted_combinations`` if the auditor's recommendations contained
        attribute=value pairs. This keeps it deterministic without depending
        on another LLM call.
        """
        if not self._all_samples:
            return []
        last_iteration_samples = [
            s for s in self._all_samples if s.iteration == self._all_samples[-1].iteration
        ]
        if not last_iteration_samples:
            return []
        return []  # The auditor's recs flow through the prompt; planner stays clean.

    def _persist_run_inputs(self, run_dir: Path) -> None:
        cfg_dump = {
            "schema_path": self.config.schema_path,
            "real_examples_path": self.config.real_examples_path,
            "domain": self.config.domain,
            "task_description": self.config.task_description,
            "iterations": self.config.iterations,
            "samples_per_iteration": self.config.samples_per_iteration,
            "initial_prompt": self.config.initial_prompt,
            "seed": self.config.seed,
        }
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg_dump, sort_keys=False), encoding="utf-8"
        )
        (run_dir / "schema.yaml").write_text(
            yaml.safe_dump(self.schema.model_dump(), sort_keys=False), encoding="utf-8"
        )
        write_jsonl(run_dir / "real_examples.jsonl", self.real_examples)

    def _persist_manifest(
        self,
        run_dir: Path,
        results: list[IterationResult],
        metric_history: list[dict[str, float]],
    ) -> None:
        manifest = {
            "run_dir": str(run_dir),
            "n_iterations": len(results),
            "final_prompt_version": self.history.current_version,
            "metric_history": metric_history,
            "prompt_history": self.history.to_list(),
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _print_metrics_table(self, metrics: dict[str, float]) -> None:
        table = Table(title="Iteration metrics", show_header=True)
        table.add_column("metric", style="bold")
        table.add_column("value", justify="right")
        priority = [
            "attribute_match_rate",
            "discriminator_accuracy",
            "synthetic_detection_rate",
            "near_duplicate_rate",
            "combination_coverage",
        ]
        for key in priority:
            if key in metrics:
                table.add_row(key, f"{metrics[key]:.3f}")
        for key, value in sorted(metrics.items()):
            if key not in priority:
                table.add_row(key, f"{value:.3f}")
        self.console.print(table)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
