"""Mid-sweep partial-completion analyzer for tasks #69 / #73 / #74.

Per CLAUDE.md Productive Wait rule: while the full sweep is running, scan
the per-seed completed runs and report early signal. Run as often as
desired during a long-running sweep; it just summarizes what's already
on disk.

Output: prints a per-task table of completed seeds + early F1 directional
signal. Does NOT modify any files.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [17, 23, 41, 53, 89]

TASKS = {
    "69": ("task69_no_pack_vs", ["full_attrforge", "no_pack", "no_pack_vs"], "customer_support"),
    "73": ("task73_sibling", ["full_attrforge", "full_attrforge_sibling"], "banking77"),
    "74": ("task74_topic", ["full_attrforge"], "TREC (topic schema)"),
}


def main() -> None:
    for task_id, (run_id, conditions, dataset) in TASKS.items():
        print(f"\n##### Task #{task_id}: {dataset} ({run_id}) #####")
        for cond in conditions:
            f1s: list[float] = []
            completed_seeds: list[int] = []
            for seed in SEEDS:
                p = REPO / "experiments" / f"{run_id}_seed{seed}" / cond / "summary.json"
                if not p.exists():
                    continue
                try:
                    s = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                ds = s.get("final_downstream") or {}
                f1 = ds.get("macro_f1")
                if f1 is not None:
                    f1s.append(float(f1))
                    completed_seeds.append(seed)
            if not f1s:
                print(f"  {cond}: 0/5 seeds done")
                continue
            mean = statistics.mean(f1s)
            std = statistics.stdev(f1s) if len(f1s) >= 2 else 0.0
            print(f"  {cond}: {len(f1s)}/5 seeds done; macro F1 = {mean:.3f} +/- {std:.3f}  (seeds: {completed_seeds})")


if __name__ == "__main__":
    main()
