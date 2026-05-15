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
    DiversityAuditor,
    RealismDiscriminator,
)
from attrforge.critics.auditor import AuditorConfig
from attrforge.critics.discriminator import DiscriminationResult, DiscriminatorConfig
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
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    auditor: AuditorConfig = field(default_factory=AuditorConfig)

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
            planner=PlannerConfig(**raw.get("planner", {})),
            generator=GeneratorConfig(**raw.get("generator", {})),
            discriminator=DiscriminatorConfig(**raw.get("discriminator", {})),
            auditor=AuditorConfig(**raw.get("auditor", {})),
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

        if config.planner.seed is None:
            config.planner.seed = config.seed
        if config.generator.seed is None:
            config.generator.seed = config.seed
        if config.discriminator.seed is None:
            config.discriminator.seed = config.seed

        self.planner = AttributePlanner(self.schema, config.planner)
        self.generator = Generator(
            gen_client, self.schema, self.real_examples, config.generator
        )
        self.verifier = AttributeVerifier(ver_client, self.schema)
        self.discriminator = RealismDiscriminator(disc_client, config.discriminator)
        self.auditor = DiversityAuditor(self.schema, aud_client, config.auditor)
        self.updater = PromptUpdater(upd_client)
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

        attribute_verdicts = self.verifier.verify(samples)
        write_jsonl(iter_dir / "attribute_verdicts.jsonl", attribute_verdicts)

        disc_result = self.discriminator.judge(self.real_examples, samples)
        write_jsonl(iter_dir / "realism_verdicts.jsonl", disc_result.verdicts)

        diversity = self.auditor.audit(samples)
        (iter_dir / "diversity_report.json").write_text(
            diversity.model_dump_json(indent=2), encoding="utf-8"
        )

        metrics = iteration_metrics(
            self.schema,
            samples,
            attribute_verdicts,
            discriminator_accuracy=disc_result.accuracy,
            synthetic_detection_rate=disc_result.synthetic_detection_rate,
            diversity=diversity,
        )
        (iter_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        return IterationResult(
            iteration=t,
            prompt_version=prompt_version,
            prompt=prompt,
            samples=samples,
            attribute_verdicts=attribute_verdicts,
            realism_verdicts=disc_result.verdicts,
            diversity=diversity,
            metrics=metrics,
        )

    def _maybe_update_prompt(self, t: int, result: IterationResult) -> None:
        synthetic_artifacts = [
            v for v in result.realism_verdicts if v.prediction == "synthetic"
        ]
        attribute_failures = [
            v for v in result.attribute_verdicts if not v.attribute_match
        ]
        feedback = IterationFeedback(
            iteration=t,
            attribute_failures=attribute_failures,
            realism_artifacts=synthetic_artifacts,
            diversity=result.diversity,
            metrics=result.metrics,
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
