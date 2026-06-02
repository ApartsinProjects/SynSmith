"""Per-critic attribution: combine full_attrforge baseline + 4 leave-one-out
conditions into one attribution table.

The full_attrforge condition lives in main_run_002 (10 seeds); the 4
leave-one-out conditions (no_pack, no_mode_seeking, no_mode_hunter,
no_coverage_hole) live in loo_run_002 (10 seeds). This script joins them
by seed and reports, for each adversary:

    Vendi(full)        - Vendi(no_X)        -> the diversity drop from
                                                removing critic X.
    distinct-n / self-BLEU drops.
    Macro F1 augmentation drop (ensemble + solo).
    Worst-class F1 drop.

A negative drop means removing the critic IMPROVES that metric (i.e., the
critic is hurting on that axis). A positive drop means removing the critic
HURTS (i.e., the critic is contributing positively on that axis).

Outputs:
    experiments/loo_run_002_aggregated/per_critic_attribution.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_per_seed(base: str, condition: str, key: str) -> list[float]:
    """Read per-seed values from an aggregated JSON."""
    # Try the reaudit_fixed.json (Vendi, MS, AUROC) first.
    p = REPO / "experiments" / f"{base}_aggregated" / "reaudit_fixed.json"
    if p.exists():
        d = json.load(open(p, encoding="utf-8"))
        rows = d.get("augmented", {}).get(condition, [])
        if rows and key in rows[0]:
            return [r[key] for r in rows]
    return []


def load_ensemble_solo(base: str, condition: str) -> tuple[list[float], list[float]]:
    """Read per-seed macro / worst-class F1 from ensemble_deep.json."""
    p = REPO / "experiments" / f"{base}_aggregated" / "ensemble_deep.json"
    if not p.exists():
        return [], []
    d = json.load(open(p, encoding="utf-8"))
    rec = d.get("solo", {}).get(condition, {})
    return rec.get("macros", []), rec.get("worsts", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-base",
        default="main_run_002",
        help="Aggregated run that contains the full_attrforge condition.",
    )
    ap.add_argument(
        "--loo-base",
        default="loo_run_002",
        help="Aggregated run that contains the no_X leave-one-out conditions.",
    )
    args = ap.parse_args()

    out_dir = REPO / "experiments" / f"{args.loo_base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    LOO = ["no_pack", "no_mode_seeking", "no_mode_hunter", "no_coverage_hole"]
    BASELINE = "full_attrforge"

    # Vendi: from reaudit_fixed
    print("=== Per-critic attribution on Vendi (semantic diversity) ===")
    base_vendi = load_per_seed(args.baseline_base, BASELINE, "vendi")
    base_vendi_mean = statistics.mean(base_vendi) if base_vendi else float("nan")
    print(
        f"  baseline (full_attrforge in {args.baseline_base}): "
        f"{base_vendi_mean:.3f} +- "
        f"{statistics.stdev(base_vendi) if len(base_vendi) > 1 else 0:.3f}  "
        f"N={len(base_vendi)}"
    )
    attribution = {"baseline_vendi": base_vendi}
    for cond in LOO:
        v = load_per_seed(args.loo_base, cond, "vendi")
        if not v:
            print(f"  {cond}: NO DATA (run reaudit_fixed --base {args.loo_base})")
            continue
        m = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0
        drop = base_vendi_mean - m
        print(
            f"  {cond:<22}: {m:.3f} +- {sd:.3f}  drop from baseline = {drop:+.3f}  N={len(v)}"
        )
        attribution[f"{cond}_vendi"] = v

    # Coverage AUROC (CV)
    print()
    print("=== Per-critic attribution on Coverage AUROC (5-fold CV) ===")
    base_auroc = load_per_seed(args.baseline_base, BASELINE, "coverage_auroc_cv")
    if base_auroc:
        bm = statistics.mean(base_auroc)
        print(f"  baseline auroc: {bm:.3f} +- {statistics.stdev(base_auroc) if len(base_auroc) > 1 else 0:.3f}")
        attribution["baseline_auroc"] = base_auroc
        for cond in LOO:
            v = load_per_seed(args.loo_base, cond, "coverage_auroc_cv")
            if not v:
                continue
            m = statistics.mean(v)
            drop = bm - m
            print(f"  {cond:<22}: {m:.3f}  drop = {drop:+.3f}")
            attribution[f"{cond}_auroc"] = v

    # Macro / worst-class F1 (downstream solo)
    print()
    print("=== Per-critic attribution on macro / worst F1 (solo classifier) ===")
    base_macro, base_worst = load_ensemble_solo(args.baseline_base, BASELINE)
    if base_macro:
        bmm = statistics.mean(base_macro)
        bww = statistics.mean(base_worst) if base_worst else float("nan")
        print(f"  baseline macro: {bmm:.3f}, worst-class: {bww:.3f}")
        attribution["baseline_macro"] = base_macro
        attribution["baseline_worst"] = base_worst
        for cond in LOO:
            m, w = load_ensemble_solo(args.loo_base, cond)
            if not m:
                print(f"  {cond}: NO DATA (run ensemble_deep --base {args.loo_base})")
                continue
            mm = statistics.mean(m)
            ww = statistics.mean(w) if w else float("nan")
            print(
                f"  {cond:<22}: macro={mm:.3f} (drop {bmm-mm:+.3f}), "
                f"worst={ww:.3f} (drop {bww-ww:+.3f})"
            )
            attribution[f"{cond}_macro"] = m
            attribution[f"{cond}_worst"] = w

    # Save
    out = out_dir / "per_critic_attribution.json"
    out.write_text(json.dumps(attribution, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    print(
        "\nReading the table: a positive drop means the critic CONTRIBUTES "
        "to that metric; a negative drop means removing the critic IMPROVES "
        "the metric (i.e., the critic is hurting on that axis)."
    )


if __name__ == "__main__":
    main()
