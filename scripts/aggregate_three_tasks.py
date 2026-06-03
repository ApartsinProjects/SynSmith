"""Aggregate results for tasks #69, #73, #74 after sweep_three_tasks.py.

For each task, scan experiments/<task>_seedXX/<condition>/ summary.json and
compute per-condition mean ± std macro F1 across seeds. Compare against the
existing v297 baselines already in the repo where applicable.

Output: experiments/_diagnostics/tasks_69_73_74_results.md

Run after sweep_three_tasks.py finishes (or while it is still running for
partial-data analysis per the Productive Wait rule).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEEDS = [17, 23, 41, 53, 89]


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_summary(run_id: str, seed: int, condition: str) -> Path | None:
    """The run_experiments.py harness writes per-condition summaries at
    experiments/<run_id>_seed<seed>/<condition>/summary.json.
    """
    candidate = REPO / "experiments" / f"{run_id}_seed{seed}" / condition / "summary.json"
    if candidate.exists():
        return candidate
    return None


def aggregate_condition(run_id: str, condition: str) -> dict:
    f1s: list[float] = []
    accs: list[float] = []
    for seed in SEEDS:
        s_path = find_summary(run_id, seed, condition)
        if s_path is None:
            continue
        s = load_summary(s_path)
        if s is None:
            continue
        ds = s.get("final_downstream") or {}
        f1 = ds.get("macro_f1")
        acc = ds.get("accuracy")
        if f1 is not None:
            f1s.append(float(f1))
        if acc is not None:
            accs.append(float(acc))
    return {
        "n_seeds": len(f1s),
        "macro_f1_mean": statistics.mean(f1s) if f1s else None,
        "macro_f1_std": statistics.stdev(f1s) if len(f1s) >= 2 else None,
        "acc_mean": statistics.mean(accs) if accs else None,
        "acc_std": statistics.stdev(accs) if len(accs) >= 2 else None,
        "f1s": f1s,
    }


def main() -> None:
    out_lines: list[str] = ["# Tasks #69 / #73 / #74 results aggregation", ""]
    out_lines.append("Per-condition mean ± std macro F1 across 5 seeds (17, 23, 41, 53, 89).")
    out_lines.append("")

    # Task #69: no_pack + VS on customer-support
    out_lines.append("## Task #69 - no_pack_vs on customer-support")
    out_lines.append("")
    out_lines.append("| Condition | n_seeds | macro F1 (mean) | macro F1 (std) | accuracy (mean) | per-seed F1 |")
    out_lines.append("|---|---|---|---|---|---|")
    for cond in ["full_attrforge", "no_pack", "no_pack_vs"]:
        a = aggregate_condition("task69_no_pack_vs", cond)
        f1m = f"{a['macro_f1_mean']:.3f}" if a['macro_f1_mean'] is not None else "n/a"
        f1s = f"{a['macro_f1_std']:.3f}" if a['macro_f1_std'] is not None else "n/a"
        accm = f"{a['acc_mean']:.3f}" if a['acc_mean'] is not None else "n/a"
        per_seed = ", ".join(f"{x:.3f}" for x in a["f1s"]) or "n/a"
        out_lines.append(f"| {cond} | {a['n_seeds']} | {f1m} | {f1s} | {accm} | {per_seed} |")
    out_lines.append("")

    # Task #73: sibling rejection on Banking77
    out_lines.append("## Task #73 - full_attrforge_sibling on Banking77")
    out_lines.append("")
    out_lines.append("| Condition | n_seeds | macro F1 (mean) | macro F1 (std) | accuracy (mean) | per-seed F1 |")
    out_lines.append("|---|---|---|---|---|---|")
    for cond in ["full_attrforge", "full_attrforge_sibling"]:
        a = aggregate_condition("task73_sibling", cond)
        f1m = f"{a['macro_f1_mean']:.3f}" if a['macro_f1_mean'] is not None else "n/a"
        f1s = f"{a['macro_f1_std']:.3f}" if a['macro_f1_std'] is not None else "n/a"
        accm = f"{a['acc_mean']:.3f}" if a['acc_mean'] is not None else "n/a"
        per_seed = ", ".join(f"{x:.3f}" for x in a["f1s"]) or "n/a"
        out_lines.append(f"| {cond} | {a['n_seeds']} | {f1m} | {f1s} | {accm} | {per_seed} |")
    out_lines.append("")

    # Task #74: topic schema on TREC
    out_lines.append("## Task #74 - topic-axis schema on TREC")
    out_lines.append("")
    out_lines.append("| Condition | n_seeds | macro F1 (mean) | macro F1 (std) | accuracy (mean) | per-seed F1 |")
    out_lines.append("|---|---|---|---|---|---|")
    a = aggregate_condition("task74_topic", "full_attrforge")
    f1m = f"{a['macro_f1_mean']:.3f}" if a['macro_f1_mean'] is not None else "n/a"
    f1s = f"{a['macro_f1_std']:.3f}" if a['macro_f1_std'] is not None else "n/a"
    accm = f"{a['acc_mean']:.3f}" if a['acc_mean'] is not None else "n/a"
    per_seed = ", ".join(f"{x:.3f}" for x in a["f1s"]) or "n/a"
    out_lines.append(f"| full_attrforge (topic schema) | {a['n_seeds']} | {f1m} | {f1s} | {accm} | {per_seed} |")
    out_lines.append("")
    out_lines.append("Compare against the v297 plain-schema TREC headline (0.609 ± 0.056) already in `experiments/_diagnostics/v297_canonical_headlines.md`.")
    out_lines.append("")

    out_path = REPO / "experiments" / "_diagnostics" / "tasks_69_73_74_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
