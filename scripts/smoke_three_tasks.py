"""Smoke test for tasks #69, #73, #74 (Hypothesis-First gate).

Runs each new condition for ONE seed, ONE iter, 8 samples per iter, no
real downstream evaluation. The goal is to catch setup bugs before the
full sweep:

- #69 no_pack_vs on customer-support: validates that no_pack + VS runs
  end-to-end without breaking the generator's VS code path or critic
  feedback shape.
- #73 full_attrforge_sibling on Banking77: validates that the sibling
  anchor block + system-prompt rejection clause flow through the
  Verifier and produce verdicts whose `reason` field references sibling
  classes when rejecting.
- #74 topic-axis schema on TREC: validates that the planner / generator
  handle a schema with an extra `topic` axis without breaking, and that
  the per-class + per-topic balance is enforceable.

For each, we dump:
- Final attribute_match_rate (sanity: should be in [0.3, 1.0])
- For #73: count of rejections whose reason mentions a sibling class
- For #74: per-topic coverage (sanity: each topic represented)

Total budget: ~30 LLM calls per condition (8 generations + 8 verifies +
small critic calls) = ~$0.05 across three conditions.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.baselines import build  # noqa: E402
from synsmith.loop import SynSmith, SynSmithConfig, configure_logging  # noqa: E402


CASES = [
    ("69_no_pack_vs", "examples/customer_support/config.yaml", "no_pack_vs"),
    ("73_full_attrforge_sibling", "examples/banking77/config.yaml", "full_attrforge_sibling"),
    ("74_topic_full", "examples/trec/config_topic.yaml", "full_attrforge"),
]


def _force_sim_backend(cfg: SynSmithConfig) -> None:
    """Rewrite every LLM config to use the sim backend in-place.

    The sim backend's heuristic responses are NOT a meaningful test of
    the sibling-anchor rejection content or topic-coverage quality, but
    they DO exercise template formatting, prompt construction, the new
    keyword-arg flow, and overall pipeline reachability. Use when the
    real API is unavailable (quota / network) to at least validate the
    code path before the next API-budget cycle.
    """
    for attr in (
        "generator_llm", "verifier_llm", "discriminator_llm",
        "auditor_llm", "updater_llm", "pack_llm", "hunter_llm",
    ):
        existing = getattr(cfg, attr, None)
        if existing is not None:
            existing.backend = "sim"


def main() -> None:
    configure_logging("WARNING")
    smoke_root = REPO / "experiments" / "_diagnostics" / "smoke_tasks_69_73_74"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    use_sim = os.environ.get("SMOKE_USE_SIM", "0") in ("1", "true", "yes")
    if use_sim:
        print("[smoke] using sim backend (no real LLM calls)", flush=True)

    results = {}
    for case_id, config_path, condition in CASES:
        case_dir = smoke_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {case_id} ({condition}) ===", flush=True)

        base = SynSmithConfig.from_yaml(REPO / config_path)
        base.iterations = 1
        base.samples_per_iteration = 8
        base.seed = 17
        base.run_dir = str(case_dir / "runs")

        cfg = build(condition, base)
        cfg.iterations = 1
        cfg.samples_per_iteration = 8
        cfg.seed = 17
        if use_sim:
            _force_sim_backend(cfg)

        t0 = time.time()
        try:
            forge = SynSmith(cfg)
            result = forge.run(iterations=1)
            elapsed = time.time() - t0
            it = result.iterations[0]
            metrics = it.metrics
            attribute_match_rate = float(metrics.get("attribute_match_rate", -1.0))

            # Sibling-rejection-specific check: how many verdicts have a reason
            # mentioning a sibling-class string?
            sibling_rejection_hits = 0
            sample_reasons = []
            if condition == "full_attrforge_sibling":
                from synsmith.schema import AttributeSchema
                schema = AttributeSchema.from_yaml(cfg.schema_path)
                class_values = set(schema.values(schema.label_attribute))
                for v in it.attribute_verdicts:
                    if not v.attribute_match:
                        for cv in class_values:
                            if cv in v.reason:
                                sibling_rejection_hits += 1
                                sample_reasons.append(v.reason[:200])
                                break

            # Topic-coverage check
            topic_coverage = {}
            if "topic_full" in case_id:
                from collections import Counter
                topic_counts = Counter()
                for s in it.samples:
                    topic_counts[s.requested_attributes.get("topic", "?")] += 1
                topic_coverage = dict(topic_counts)

            results[case_id] = {
                "condition": condition,
                "elapsed_s": round(elapsed, 1),
                "n_samples": len(it.samples),
                "attribute_match_rate": attribute_match_rate,
                "sibling_rejection_hits": sibling_rejection_hits,
                "sample_rejection_reasons": sample_reasons[:3],
                "topic_coverage": topic_coverage,
                "metrics": {k: round(float(v), 3) for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            print(json.dumps(results[case_id], indent=2), flush=True)

        except Exception as e:
            results[case_id] = {"condition": condition, "ERROR": str(e)[:500]}
            print(f"!!! {case_id} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    (smoke_root / "smoke_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nSmoke results written to {smoke_root / 'smoke_results.json'}", flush=True)


if __name__ == "__main__":
    main()
