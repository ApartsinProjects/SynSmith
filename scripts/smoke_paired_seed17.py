"""Paired smoke: full_attrforge vs no_pack at seed 17 with F1-F6 fixes.

Question: does the F1-F6 fix close the no_pack gap? Or does Pack still
add net-negative value on customer-support even after F2?
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.baselines import build
from synsmith.eval.downstream import DownstreamConfig, DownstreamEvaluator
from synsmith.loop import SynSmith, SynSmithConfig, configure_logging
from synsmith.schema import RealExample, SyntheticSample, load_jsonl


def run_one(condition: str) -> dict:
    cfg = SynSmithConfig.from_yaml(REPO / "examples/customer_support/config.yaml")
    cfg.iterations = 3
    cfg.samples_per_iteration = 16
    cfg.seed = 17
    out_root = REPO / "experiments" / "_diagnostics" / f"smoke_paired_f1_f6_{condition}"
    if out_root.exists():
        import shutil; shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(out_root)
    cfg = build(condition, cfg)

    t0 = time.time()
    forge = SynSmith(cfg)
    result = forge.run()
    elapsed = (time.time() - t0) / 60.0

    # Pool synth from the run
    synth: list[SyntheticSample] = []
    for it in range(cfg.iterations):
        for path in (Path(result.run_dir) / f"iter_{it:03d}" / "samples.jsonl",
                     Path(result.run_dir) / f"iter_{it:03d}" / "samples_regen.jsonl"):
            if path.exists():
                for row in load_jsonl(path):
                    synth.append(SyntheticSample.model_validate(row))

    test_real = [
        RealExample.model_validate(r)
        for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")
    ]
    evaluator = DownstreamEvaluator(DownstreamConfig(seed=17))
    ds = evaluator.evaluate(synth, test_real, label_attribute="intent")
    return {
        "condition": condition,
        "elapsed_min": elapsed,
        "n_synth": len(synth),
        "macro_f1": ds.macro_f1,
        "accuracy": ds.accuracy,
        "per_class_f1": ds.per_class_f1,
    }


def main() -> None:
    configure_logging("WARNING")
    results = {}
    for cond in ["full_attrforge", "no_pack"]:
        print(f"\n##### Running {cond} #####", flush=True)
        results[cond] = run_one(cond)
        print(f"  done: F1={results[cond]['macro_f1']:.3f}  elapsed={results[cond]['elapsed_min']:.1f} min", flush=True)

    print("\n##### F1-F6 paired smoke summary (seed 17) #####")
    for cond, r in results.items():
        print(f"  {cond}: macro F1 = {r['macro_f1']:.3f}, per-class = {r['per_class_f1']}")

    delta = results["no_pack"]["macro_f1"] - results["full_attrforge"]["macro_f1"]
    print()
    print(f"  no_pack - full_attrforge delta (post F1-F6): {delta:+.3f}")
    print(f"  pre-fix delta at seed 17 was: +0.487")

    out = REPO / "experiments" / "_diagnostics" / "smoke_paired_f1_f6_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
