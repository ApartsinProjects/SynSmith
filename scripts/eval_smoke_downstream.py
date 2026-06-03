"""Evaluate downstream F1 on the F1-F6 smoke synthetic batch and compare
against the pre-fix seed-17 result (0.313).

Quick paired comparison: same seed, same test set, just fixed critics.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.eval.downstream import DownstreamConfig, DownstreamEvaluator
from synsmith.schema import RealExample, SyntheticSample, load_jsonl


def main() -> None:
    # Find the most recent F1-F6 smoke run
    runs = sorted(glob.glob(str(REPO / "experiments/_diagnostics/smoke_fixes_f1_f6/runs/2026*")))
    if not runs:
        print("No F1-F6 smoke run found.")
        return
    rd = Path(runs[-1])
    print(f"Evaluating: {rd}")

    # Pool all iters' synth
    synth: list[SyntheticSample] = []
    for it in range(3):
        sf = rd / f"iter_{it:03d}" / "samples.jsonl"
        regen = rd / f"iter_{it:03d}" / "samples_regen.jsonl"
        for path in (sf, regen):
            if path.exists():
                for row in load_jsonl(path):
                    synth.append(SyntheticSample.model_validate(row))
    print(f"  synth pooled: {len(synth)} samples")

    # Load real test
    test_real = [
        RealExample.model_validate(r)
        for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")
    ]
    print(f"  test real: {len(test_real)} items")

    evaluator = DownstreamEvaluator(DownstreamConfig(seed=17))
    result = evaluator.evaluate(synth, test_real, label_attribute="intent")
    print()
    print("=== F1-F6 fixed full_attrforge seed 17 ===")
    print(f"  macro F1: {result.macro_f1:.3f}")
    print(f"  accuracy: {result.accuracy:.3f}")
    print(f"  per-class F1: {result.per_class_f1}")
    print()
    print("=== Comparison to pre-fix seed 17 ===")
    print(f"  pre-fix full_attrforge: 0.313 macro F1")
    print(f"  pre-fix no_pack:        0.800 macro F1 (the 'accidentally-wins' result)")
    print(f"  post-fix full_attrforge: {result.macro_f1:.3f}")
    delta_vs_prefix = result.macro_f1 - 0.313
    print(f"  delta vs pre-fix full_attrforge: {delta_vs_prefix:+.3f}")


if __name__ == "__main__":
    main()
