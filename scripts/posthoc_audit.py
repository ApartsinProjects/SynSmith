"""Post-hoc audit: run all four GAN-style adversaries on every condition's final batch.

Closes review weakness M6: the experiment runner only computes a critic
metric when the critic was enabled in that condition. So the "differential
firing" claim in the paper is structurally tautological. This script reads
the final batch from each condition and runs Pack, Mode-Seeking, Mode
Hunter, and Coverage Hole Finder as POST-HOC auditors. Now every condition
has every adversary metric, and differences attribute to the data, not the
config.

Outputs:
  experiments/<run_id>/audit/audit_summary.csv
  experiments/<run_id>/audit/audit_summary.json
  paper/figures/<run_id>_audit_differential.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import attrforge  # noqa: E402  (triggers .env load)
from attrforge.critics.coverage_hole import (  # noqa: E402
    CoverageHoleConfig,
    CoverageHoleFinder,
)
from attrforge.critics.mode_hunter import ModeHunter, ModeHunterConfig  # noqa: E402
from attrforge.critics.mode_seeking import ModeSeeking, ModeSeekingConfig  # noqa: E402
from attrforge.critics.pack_discriminator import (  # noqa: E402
    PackDiscriminator,
    PackDiscriminatorConfig,
)
from attrforge.llm import LLMConfig, build_client  # noqa: E402
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_final_samples(condition_dir: Path) -> list[SyntheticSample]:
    """Return all synthetic samples produced by a condition (all iterations pooled)."""
    samples: list[SyntheticSample] = []
    for iter_dir in sorted(condition_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for row in load_jsonl(sj):
                samples.append(SyntheticSample.model_validate(row))
    return samples


def load_real(path: Path) -> list[RealExample]:
    return [RealExample.model_validate(r) for r in load_jsonl(path)]


def pack_null_real_vs_real(
    real: list[RealExample], pack: PackDiscriminator, n: int = 6
) -> float:
    """Null baseline: pack accuracy when the 'synthetic' pack is also drawn from real.

    Equivalent to pack(real_split_A, real_split_B). Any deviation from 0.5
    is the discriminator's inherent bias on the real set; it lower-bounds
    what 'collapsed synth' means in this simulator.
    """
    if len(real) < 2 * pack.config.pack_size:
        return 0.5
    half = len(real) // 2
    fake_synth = [SyntheticSample(
        sample_id=f"R{i:03d}",
        text=r.text,
        requested_attributes={"intent": r.label or "x"},
    ) for i, r in enumerate(real[half:])]
    result = pack.attack(real[:half], fake_synth)
    return result.pack_accuracy


def run_audit(
    run_id: str,
    backend: str = "sim",
) -> dict:
    run_dir = REPO / "experiments" / run_id
    if not run_dir.exists():
        sys.exit(f"no such run dir: {run_dir}")

    real_path = REPO / "experiments" / "_splits" / "real_train.jsonl"
    real_test_path = REPO / "experiments" / "_splits" / "real_test.jsonl"
    if not real_path.exists():
        sys.exit("no real split found; run scripts/run_experiments.py first")
    real_train = load_real(real_path)
    real_test = load_real(real_test_path) if real_test_path.exists() else []
    real_for_audit = real_train + real_test

    # Build auditors. Use the same backend for the LLM-needing ones.
    # IMPORTANT: do NOT instantiate the pack discriminator once and reuse it
    # across conditions, because its RNG state would evolve with each call,
    # making pack accuracies for later conditions a function of processing
    # order rather than data. We re-instantiate per condition below.
    pack_cfg = PackDiscriminatorConfig(pack_size=4, n_comparisons=16, seed=99)
    pack_client = build_client(LLMConfig(backend=backend, model=f"{backend}-audit"))

    mode_seeking = ModeSeeking(ModeSeekingConfig(use_embeddings=False))
    hunter_client = build_client(LLMConfig(backend=backend, model=f"{backend}-audit"))
    coverage_hole = CoverageHoleFinder(CoverageHoleConfig(top_k=5))

    # Compute the null reference once. real-vs-real-split pack accuracy.
    null_pack = PackDiscriminator(pack_client, pack_cfg)
    null_pack_acc = pack_null_real_vs_real(real_for_audit, null_pack)

    # Also compute mode-seeking on the real set, for the M2 review point.
    real_as_synth = [
        SyntheticSample(
            sample_id=f"R{i:03d}",
            text=r.text,
            requested_attributes={"intent": r.label or "x"},
        )
        for i, r in enumerate(real_for_audit)
    ]
    real_ms = mode_seeking.score(real_as_synth)

    audit_rows: list[dict] = []

    condition_dirs = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name not in {"audit", "aggregated"} and not p.name.startswith("_")
    )

    for cond_dir in condition_dirs:
        cond = cond_dir.name
        samples = load_final_samples(cond_dir)
        if not samples:
            print(f"[skip] {cond}: no samples found", flush=True)
            continue

        # Fresh pack discriminator and mode hunter per condition: identical
        # RNG state, identical inputs, identical output for identical data.
        pack = PackDiscriminator(pack_client, pack_cfg)
        mode_hunter = ModeHunter(
            hunter_client,
            ModeHunterConfig(max_findings_per_iter=4, min_repeats=1),
        )
        pack_res = pack.attack(real_for_audit, samples)
        ms_res = mode_seeking.score(samples)
        hunt_res = mode_hunter.hunt(real_for_audit, samples, iteration=999)
        hole_res = coverage_hole.find(real_for_audit, samples)

        # ms ratio relative to real
        ms_real_ref = real_ms.mode_seeking_ratio if real_ms.mode_seeking_ratio else 1e-6
        ms_relative = ms_res.mode_seeking_ratio / ms_real_ref if ms_real_ref else 0.0

        row = {
            "condition": cond,
            "n_samples": len(samples),
            "pack_accuracy": pack_res.pack_accuracy,
            "pack_accuracy_above_null": pack_res.pack_accuracy - null_pack_acc,
            "pack_confidence": pack_res.confidence_mean,
            "mode_seeking_ratio": ms_res.mode_seeking_ratio,
            "mode_seeking_relative_to_real": ms_relative,
            "n_banned_phrasings_found": len(hunt_res.new_findings),
            "coverage_auroc": hole_res.classifier_auroc,
            "n_coverage_holes": len(hole_res.holes),
        }
        audit_rows.append(row)
        print(
            f"  {cond:<18} pack_acc={row['pack_accuracy']:.3f} "
            f"ms={row['mode_seeking_ratio']:.3f} ({ms_relative:.2f}x real) "
            f"auroc={row['coverage_auroc']:.3f} hunter_new={row['n_banned_phrasings_found']}",
            flush=True,
        )

    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "null_pack_accuracy_real_vs_real": null_pack_acc,
        "real_mode_seeking_ratio": real_ms.mode_seeking_ratio,
        "real_text_distance_mean": real_ms.text_distance_mean,
        "conditions": audit_rows,
    }
    (audit_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with (audit_dir / "audit_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(audit_rows)

    # Differential plot.
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_audit(audit_rows, null_pack_acc, real_ms.mode_seeking_ratio,
                fig_dir / f"{run_id}_audit_differential.png")

    print(f"\nAudit written to {audit_dir}")
    return summary


def _plot_audit(rows: list[dict], null_pack: float, real_ms: float, out_path: Path) -> None:
    conds = [r["condition"] for r in rows]
    pack_vals = [r["pack_accuracy"] for r in rows]
    ms_vals = [r["mode_seeking_ratio"] for r in rows]
    ms_rel = [r["mode_seeking_relative_to_real"] for r in rows]
    auroc_vals = [r["coverage_auroc"] for r in rows]
    hunter = [r["n_banned_phrasings_found"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Pack accuracy with null reference.
    ax = axes[0, 0]
    ax.bar(range(len(conds)), pack_vals, color="#c0392b")
    ax.axhline(0.5, color="#888", linestyle=":", label="chance")
    ax.axhline(null_pack, color="#3a6ea5", linestyle="--", label=f"null real-vs-real ({null_pack:.2f})")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("pack accuracy")
    ax.set_title("Pack accuracy (audited on every condition; lower = more diverse)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    for i, v in enumerate(pack_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Mode-seeking, relative to real.
    ax = axes[0, 1]
    ax.bar(range(len(conds)), ms_rel, color="#3a6ea5")
    ax.axhline(1.0, color="#888", linestyle="--", label=f"real ms = {real_ms:.3f}")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, max(1.2, max(ms_rel) * 1.15))
    ax.set_ylabel("synth-ms / real-ms")
    ax.set_title("Mode-seeking ratio relative to real (higher = more attribute-responsive)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    for i, v in enumerate(ms_rel):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # AUROC.
    ax = axes[1, 0]
    ax.bar(range(len(conds)), auroc_vals, color="#7c4dff")
    ax.axhline(0.5, color="#888", linestyle=":", label="indistinguishable (0.5)")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("classifier AUROC")
    ax.set_title("Coverage-hole AUROC (lower = more real-like coverage)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    for i, v in enumerate(auroc_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Mode hunter findings.
    ax = axes[1, 1]
    ax.bar(range(len(conds)), hunter, color="#2e715a")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("# LLM tics detected in final batch")
    ax.set_title("Mode Hunter findings on final batch (lower = fewer surviving tics)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, v in enumerate(hunter):
        ax.text(i, v + 0.05, f"{int(v)}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Post-hoc adversary audit: all four GAN-style metrics on every condition's final batch", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--backend", default="sim")
    args = ap.parse_args()
    run_audit(args.run_id, backend=args.backend)


if __name__ == "__main__":
    main()
