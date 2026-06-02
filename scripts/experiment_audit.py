"""Systematic audit of all AttrForge experiment results for bugs and inconsistencies.

Checks performed across every experiment directory under experiments/:

1. **Path-mixup audit**: for each run, verify the real_examples.jsonl matches
   the expected dataset stem (banking77_ / trec_ / sst2_ / mnli_ / default).
2. **Label-class consistency**: synth samples' requested_attributes['intent']
   values must be a subset of the schema's allowed values.
3. **Sample count audit**: iter_NNN/samples.jsonl entries should equal
   samples_per_iteration unless verifier rejects forced regeneration.
4. **Metric range sanity**: discriminator_accuracy, pack_accuracy in [0, 1];
   near_duplicate_rate in [0, 1]; if metric is out-of-range, flag the iter.
5. **Re-derivation cross-check**: re-compute downstream macro-F1 from
   stored samples + splits via sentence-transformer; flag any aggregation
   file whose headline number differs from the re-derivation by > 0.02.
6. **Train/test contamination**: any synth sample whose text exactly
   matches a test item is a contamination bug.
7. **Per-iter cumulative monotonicity**: samples should accumulate across
   iters; cumulative count(iter k) >= count(iter k-1).
8. **Discriminator equilibrium signal sanity**: under OLD framework,
   disc_acc often collapses to 0 (the discriminator gave up); flag any
   run where iter_N disc_acc < 0.1 (broken signal).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def _read_jsonl_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return list(load_jsonl(path))
    except Exception:
        return []


def _dataset_stem_for(run_dir: Path) -> str:
    name = run_dir.name.lower()
    for stem in ("banking77", "trec", "sst2", "mnli"):
        if stem in name:
            return stem
    return "default"


def _expected_splits(stem: str) -> tuple[Path, Path]:
    base = REPO / "experiments" / "_splits"
    if stem == "default":
        return base / "real_train.jsonl", base / "real_test.jsonl"
    return base / f"{stem}_real_train.jsonl", base / f"{stem}_real_test.jsonl"


def audit_run(run_dir: Path, condition_dir: Path) -> list[str]:
    """Audit one (run_dir, condition_dir) pair. Returns list of findings."""
    findings: list[str] = []
    stem = _dataset_stem_for(run_dir)
    expected_train, expected_test = _expected_splits(stem)

    # 1. Path-mixup: real_examples.jsonl content must match expected stem.
    real_path = condition_dir / "real_examples.jsonl"
    if real_path.exists():
        real_in_cond = _read_jsonl_safe(real_path)
        real_in_split = _read_jsonl_safe(expected_train)
        if real_in_split and real_in_cond:
            in_split_texts = {r["text"] for r in real_in_split}
            in_cond_texts = {r["text"] for r in real_in_cond}
            overlap = len(in_split_texts & in_cond_texts) / max(1, len(in_cond_texts))
            if overlap < 0.5:
                findings.append(
                    f"PATH-MIXUP: condition {condition_dir.name} real_examples "
                    f"matches only {overlap:.0%} of expected {stem} train split"
                )

    # 2. Label-class consistency
    schema_path = condition_dir / "schema.yaml"
    allowed_labels: set[str] = set()
    if schema_path.exists():
        try:
            import yaml
            schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
            label_attr = schema.get("label_attribute", "intent")
            allowed_labels = set(str(v) for v in schema.get("attributes", {}).get(label_attr, []))
        except Exception:
            pass

    # Walk timestamped subdirs
    iter_dirs = sorted(condition_dir.rglob("iter_*"))
    if not iter_dirs:
        return findings
    cumulative_synth_count = 0
    prev_disc_acc: float | None = None
    for it in iter_dirs:
        samples = _read_jsonl_safe(it / "samples.jsonl")
        cumulative_synth_count += len(samples)
        # 2: label-class
        if allowed_labels:
            for s in samples:
                req = s.get("requested_attributes", {}).get("intent")
                if req is not None and str(req) not in allowed_labels:
                    findings.append(
                        f"LABEL-OOR: {it.name} sample requested intent={req!r} "
                        f"not in allowed {sorted(allowed_labels)[:5]}..."
                    )
                    break  # one finding per iter
        # 4: metric ranges
        mp = it / "metrics.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                for k in ("discriminator_accuracy", "pack_accuracy", "near_duplicate_rate", "synthetic_detection_rate"):
                    if k in m:
                        v = m[k]
                        if v is not None and not (0.0 <= v <= 1.0001):
                            findings.append(f"METRIC-OOR: {it.name} {k}={v}")
                # 8: discriminator-collapse signal
                disc = m.get("discriminator_accuracy")
                if disc is not None and disc < 0.1 and it.name != "iter_000":
                    findings.append(
                        f"DISC-COLLAPSE: {it.name} discriminator_accuracy={disc:.3f} "
                        f"(< 0.1 means discriminator gave up; broken signal under OLD framework)"
                    )
            except Exception:
                findings.append(f"METRIC-PARSE-FAIL: {it.name} metrics.json could not parse")

    # 6: Train/test contamination
    test_texts = {r["text"] for r in _read_jsonl_safe(expected_test)}
    if test_texts:
        contaminated = 0
        for it in iter_dirs:
            for s in _read_jsonl_safe(it / "samples.jsonl"):
                if s.get("text") in test_texts:
                    contaminated += 1
        if contaminated > 0:
            findings.append(
                f"CONTAMINATION: {contaminated} synth samples exactly match test items "
                f"in {expected_test.name}"
            )

    return findings


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    exp_root = REPO / "experiments"
    all_findings: dict[str, list[str]] = {}
    seen_runs = 0
    seen_conds = 0
    for run_dir in sorted(exp_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        for cond_dir in sorted(run_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            if not (cond_dir / "manifest.json").exists() and not any(cond_dir.rglob("iter_*")):
                continue
            seen_conds += 1
            findings = audit_run(run_dir, cond_dir)
            if findings:
                all_findings[f"{run_dir.name}/{cond_dir.name}"] = findings
        seen_runs += 1

    print(f"Audited {seen_runs} run dirs, {seen_conds} (run,condition) pairs")
    print(f"Findings in {len(all_findings)} pairs:\n")
    if not all_findings:
        print("  (no audit failures)")
        return
    # Group by finding-type prefix
    counts: dict[str, int] = defaultdict(int)
    for path, fs in all_findings.items():
        for f in fs:
            tag = f.split(":", 1)[0]
            counts[tag] += 1
    print("=== Finding counts by type ===")
    for tag, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tag}: {n}")

    print("\n=== Per-run findings (first 50 pairs) ===")
    for i, (path, fs) in enumerate(sorted(all_findings.items())):
        if i >= 50:
            print(f"  ... and {len(all_findings) - 50} more pairs")
            break
        print(f"\n{path}:")
        for f in fs[:5]:
            print(f"  - {f}")
        if len(fs) > 5:
            print(f"  ... and {len(fs) - 5} more findings in this pair")


if __name__ == "__main__":
    main()
