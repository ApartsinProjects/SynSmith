"""Full sweep launcher for tasks #69 (no_pack+VS), #73 (sibling), #74 (topic).

Each task is launched as a separate run via scripts/run_experiments.py with
the relevant conditions and a 5-seed schedule. Tasks are launched in
SERIES (not parallel) to avoid OpenAI RPM contention, but each task is
self-contained and writes to its own experiments/ directory tag.

Output (one tagged directory per task):
  experiments/task69_no_pack_vs_seedXX/<condition>/
  experiments/task73_sibling_seedXX/<condition>/
  experiments/task74_topic_seedXX/<condition>/

Compares against the v297 full_attrforge baseline already in the repo:
  experiments/customer_support v297 ... (existing)
  experiments/banking77_v297_seedXX/full_attrforge/
  experiments/trec_v297_seed{17,23,41,53,89}/full_attrforge/

Run separately when ready:
    /c/Python314/python scripts/sweep_three_tasks.py --task 69
    /c/Python314/python scripts/sweep_three_tasks.py --task 73
    /c/Python314/python scripts/sweep_three_tasks.py --task 74
    /c/Python314/python scripts/sweep_three_tasks.py --task all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SEEDS = [17, 23, 41, 53, 89]

TASKS = {
    "69": {
        "config": "examples/customer_support/config.yaml",
        # Compare no_pack_vs against its direct neighbors (no_pack and full_attrforge).
        "conditions": ["full_attrforge", "no_pack", "no_pack_vs"],
        "run_id": "task69_no_pack_vs",
        "n_test": 10,
        "iterations": 3,
        "samples_per_iteration": 16,
    },
    "73": {
        "config": "examples/banking77/config.yaml",
        # Compare sibling-rejection against the existing full_attrforge headline.
        "conditions": ["full_attrforge", "full_attrforge_sibling"],
        "run_id": "task73_sibling",
        "n_test": 400,
        "iterations": 3,
        "samples_per_iteration": 16,
    },
    "74": {
        # Topic schema is loaded by config_topic.yaml.
        "config": "examples/trec/config_topic.yaml",
        # Compare full_attrforge under topic-aware schema vs the existing
        # plain-schema headline (in experiments/trec_v297_seedXX).
        "conditions": ["full_attrforge"],
        "run_id": "task74_topic",
        "n_test": 89,
        "iterations": 3,
        "samples_per_iteration": 16,
    },
}


def run_task(task_id: str) -> int:
    spec = TASKS[task_id]
    print(f"\n##### Task #{task_id}: {spec['run_id']} #####", flush=True)
    print(f"  config: {spec['config']}", flush=True)
    print(f"  conditions: {spec['conditions']}", flush=True)
    print(f"  seeds: {SEEDS}", flush=True)
    cmd = [
        r"C:\Python314\python", "scripts/run_experiments.py",
        "--config", spec["config"],
        "--iterations", str(spec["iterations"]),
        "--samples-per-iteration", str(spec["samples_per_iteration"]),
        "--n-test", str(spec["n_test"]),
        "--conditions", *spec["conditions"],
        "--seeds", *[str(s) for s in SEEDS],
        "--run-id", spec["run_id"],
    ]
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(REPO))
    elapsed = (time.time() - t0) / 60.0
    print(f"  rc={rc} elapsed_min={elapsed:.1f}", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS) + ["all"])
    args = ap.parse_args()
    if args.task == "all":
        tasks = list(TASKS)
    else:
        tasks = [args.task]
    rcs = {}
    for t in tasks:
        rcs[t] = run_task(t)
    print("\n##### Summary #####")
    for t, rc in rcs.items():
        print(f"  Task #{t}: rc={rc}")


if __name__ == "__main__":
    main()
