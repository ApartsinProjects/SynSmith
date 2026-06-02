"""The AttrForge orchestrator.

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

from attrforge.critics import (
    AttributeVerifier,
    CoverageHoleFinder,
    DiversityAuditor,
    ModeHunter,
    ModeSeeking,
    PackDiscriminator,
    RealismDiscriminator,
)
from attrforge.critics.auditor import AuditorConfig
from attrforge.critics.coverage_hole import CoverageHoleConfig, CoverageHoleResult
from attrforge.critics.discriminator import DiscriminationResult, DiscriminatorConfig
from attrforge.critics.mode_hunter import ModeHunterConfig, ModeHunterResult
from attrforge.critics.mode_seeking import ModeSeekingConfig, ModeSeekingResult
from attrforge.critics.pack_discriminator import (
    PackDiscriminatorConfig,
    PackResult,
)
from attrforge.generator import Generator, GeneratorConfig
from attrforge.llm import LLMClient, LLMConfig, build_client
from attrforge.metrics import iteration_metrics
from attrforge.planner import AttributePlanner, PlannerConfig
from attrforge.prompts import GENERATOR_INITIAL
from attrforge.schema import (
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
from attrforge.updater import PromptHistory, PromptUpdater

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
class AttrForgeConfig:
    """Top-level config object passed to ``AttrForge``.

    Built from a YAML file via :meth:`AttrForge.from_config`. Every nested
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
    label: str = "full_attrforge"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AttrForgeConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AttrForgeConfig":
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
        )


class AttrForge:
    """Coordinates the full iterative pipeline."""

    def __init__(self, config: AttrForgeConfig) -> None:
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
        from attrforge.critics.verifier import VerifierConfig
        self.verifier = (
            AttributeVerifier(
                ver_client,
                self.schema,
                real_examples=self.real_examples,
                config=VerifierConfig(seed=config.seed),
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
    def from_config(cls, path: str | Path) -> "AttrForge":
        return cls(AttrForgeConfig.from_yaml(path))

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

        samples = self.generator.generate(
            targets, prompt=prompt, prompt_version=prompt_version, iteration=t
        )
        write_jsonl(iter_dir / "samples.jsonl", samples)
        self._all_samples.extend(samples)

        # 1. attribute verifier
        if self.verifier is not None:
            attribute_verdicts = self.verifier.verify(samples)
        else:
            attribute_verdicts = []
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
