"""End-to-end experiment runner for the AttrForge paper.

For each named baseline, run the loop with the same config (only critic
stack varies), then evaluate every iteration's synthetic batch on the
held-out real test set with the downstream classifier.

Outputs (per run):

    experiments/<run_id>/
      manifest.json                # raw run manifest from loop.py
      summary.json                 # condition-level rollup
      downstream/iter_NNN.json     # per-iter classifier metrics
      downstream/final.json        # final iteration only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attrforge.baselines import BASELINES, build  # noqa: E402
from attrforge.eval.downstream import DownstreamConfig, DownstreamEvaluator  # noqa: E402
from attrforge.loop import AttrForge, AttrForgeConfig, configure_logging  # noqa: E402
from attrforge.schema import RealExample, load_jsonl  # noqa: E402


def split_real(
    path: Path, n_test: int, seed: int, splits_prefix: str = ""
) -> tuple[list[RealExample], list[RealExample], Path, Path]:
    """Stratified split: hold out n_test/labels per class for the downstream test set.

    Persists both splits to ``experiments/_splits/`` so every condition
    sees the *same* test set. The output filenames are derived from the
    INPUT real_examples_path's stem so that running on multiple datasets
    (e.g. customer_support + banking77) does not silently overwrite each
    other's split files. Concretely, if input is
    ``experiments/_splits/banking77_real_train.jsonl``, the output is
    ``experiments/_splits/banking77_real_train.split_train.jsonl`` and
    ``...split_test.jsonl``. If input is the legacy
    ``examples/customer_support/real_examples.jsonl``, the output retains
    the v1 names ``real_train.jsonl`` / ``real_test.jsonl`` so existing
    analyses keep working.
    """
    rows = [RealExample.model_validate(r) for r in load_jsonl(path)]
    by_label: dict[str, list[RealExample]] = {}
    for r in rows:
        by_label.setdefault(r.label or "_", []).append(r)
    rng = random.Random(seed)
    train: list[RealExample] = []
    test: list[RealExample] = []
    per_class_test = max(1, n_test // max(1, len(by_label)))
    for lbl, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        test.extend(items[:per_class_test])
        train.extend(items[per_class_test:])

    out_dir = REPO / "experiments" / "_splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve v1 names for the original customer-support layout. For any
    # other input file, derive split filenames from the input stem so the
    # canonical real_train.jsonl/real_test.jsonl never get clobbered.
    input_stem = Path(path).stem
    if input_stem == "real_examples":
        train_path = out_dir / "real_train.jsonl"
        test_path = out_dir / "real_test.jsonl"
    else:
        train_path = out_dir / f"{input_stem}.split_train.jsonl"
        test_path = out_dir / f"{input_stem}.split_test.jsonl"
    train_path.write_text(
        "\n".join(r.model_dump_json() for r in train) + "\n", encoding="utf-8"
    )
    test_path.write_text(
        "\n".join(r.model_dump_json() for r in test) + "\n", encoding="utf-8"
    )
    return train, test, train_path, test_path


def run_one(
    name: str,
    base_cfg: AttrForgeConfig,
    test_real: list[RealExample],
    run_root: Path,
    iterations: int,
    samples_per_iteration: int,
) -> dict:
    cfg = build(name, base_cfg)
    cfg.iterations = iterations if name not in ("naive", "few_shot") else 1
    cfg.samples_per_iteration = samples_per_iteration
    run_dir = run_root / name
    cfg.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running condition: {name} (iters={cfg.iterations}, n/iter={cfg.samples_per_iteration}) ===", flush=True)
    t0 = time.time()
    forge = AttrForge(cfg)
    result = forge.run()
    elapsed = time.time() - t0

    eval_dir = Path(result.run_dir) / "downstream"
    eval_dir.mkdir(parents=True, exist_ok=True)
    evaluator = DownstreamEvaluator(DownstreamConfig(seed=base_cfg.seed or 17))

    per_iter_downstream: list[dict] = []
    cumulative: list = []
    for it in result.iterations:
        cumulative.extend(it.samples)
        # Per-iteration downstream: train on samples produced *in this iteration only*.
        ds = evaluator.evaluate(it.samples, test_real, label_attribute="intent")
        ds_iter_dict = ds.model_dump()
        ds_iter_dict["iteration"] = it.iteration
        per_iter_downstream.append(ds_iter_dict)
        (eval_dir / f"iter_{it.iteration:03d}.json").write_text(
            json.dumps(ds_iter_dict, indent=2), encoding="utf-8"
        )

    # Final cumulative downstream: train on ALL iterations of synthetic data.
    ds_cumulative = evaluator.evaluate(
        cumulative, test_real, label_attribute="intent"
    )
    (eval_dir / "final.json").write_text(
        json.dumps(ds_cumulative.model_dump(), indent=2), encoding="utf-8"
    )

    last_metrics = result.metric_history[-1] if result.metric_history else {}
    summary = {
        "condition": name,
        "label": cfg.label,
        "iterations": cfg.iterations,
        "samples_per_iteration": cfg.samples_per_iteration,
        "total_samples": len(cumulative),
        "wall_time_seconds": elapsed,
        "final_prompt_version": result.final_prompt_version,
        "final_metrics": last_metrics,
        "per_iter_downstream": per_iter_downstream,
        "final_downstream": ds_cumulative.model_dump(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"  done in {elapsed:.1f}s. final macro_f1 (cumulative) = "
        f"{ds_cumulative.macro_f1:.3f}, accuracy = {ds_cumulative.accuracy:.3f}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=Path, default=REPO / "examples/customer_support/config.yaml"
    )
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--samples-per-iteration", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=[
            "naive",
            "few_shot",
            "self_critique",
            "realism_only",
            "diversity_only",
            "full_classic",
            "full_attrforge",
        ],
    )
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--backend", type=str, default=None,
                    help="Override LLM backend for all critics (echo|openai|anthropic|sim).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="One or more seeds. When >1 seed is supplied, the run-id is "
                         "suffixed with _seed<n> and each condition runs once per seed.")
    args = ap.parse_args()

    configure_logging("WARNING")
    base_cfg = AttrForgeConfig.from_yaml(args.config)

    # Override backend everywhere if requested.
    if args.backend:
        for attr in (
            "generator_llm", "verifier_llm", "discriminator_llm",
            "auditor_llm", "updater_llm", "pack_llm", "hunter_llm",
        ):
            existing = getattr(base_cfg, attr, None)
            if existing is not None:
                existing.backend = args.backend

    real_train, real_test, train_path, test_path = split_real(
        Path(base_cfg.real_examples_path), n_test=args.n_test, seed=base_cfg.seed or 17
    )
    # Point the loop's real-examples path at the train split, so the
    # discriminator and few-shot pool never see the held-out test set.
    base_cfg.real_examples_path = str(train_path)

    print(
        f"Real split: {len(real_train)} train (for prompts + critics), "
        f"{len(real_test)} held-out test for downstream classifier.",
        flush=True,
    )
    print(f"Conditions: {args.conditions}", flush=True)

    base_run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    seeds = args.seeds or [base_cfg.seed or 17]

    for seed in seeds:
        seed_cfg = AttrForgeConfig(**{**base_cfg.__dict__, "seed": seed})
        # Propagate seed to nested configs.
        seed_cfg.planner.seed = seed
        seed_cfg.generator.seed = seed
        seed_cfg.discriminator.seed = seed
        seed_cfg.pack_discriminator.seed = seed
        run_id = base_run_id if len(seeds) == 1 else f"{base_run_id}_seed{seed}"
        run_root = REPO / "experiments" / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        print(f"\n### Run: {run_id} (seed={seed}) ###", flush=True)
        summaries: list[dict] = []
        for name in args.conditions:
            try:
                s = run_one(
                    name,
                    seed_cfg,
                    real_test,
                    run_root,
                    iterations=args.iterations,
                    samples_per_iteration=args.samples_per_iteration,
                )
                summaries.append(s)
            except Exception as exc:
                print(f"!! condition {name} failed: {exc}", flush=True)
                summaries.append({"condition": name, "error": str(exc)})

        (run_root / "all_summaries.json").write_text(
            json.dumps(summaries, indent=2, default=str), encoding="utf-8"
        )
        print(f"All results written to {run_root}", flush=True)


if __name__ == "__main__":
    main()
