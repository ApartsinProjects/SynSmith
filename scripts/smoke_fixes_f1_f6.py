"""Smoke test for the F1-F6 fix bundle on customer-support seed 17.

Runs ONE full SynSmith iteration with the fixed critics enabled and dumps
the key intermediates needed for deep analysis:

- F1: Realism Discriminator anchor stability (which real samples appear across iters)
- F2: Pack Discriminator shared_patterns (should no longer include register-describing phrases)
- F4: Updater prompt (does it now lead with a Preferred Phrasings block?)
- F5: Pack-vs-real filter language in the Updater's instructions
- F6: Coverage Hole exemplars (stratified across classes)

Cost: 1 seed × 3 iters × 16 samples × ~7 critic calls = ~$0.10
Wall-clock: ~5 minutes
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.baselines import build  # noqa: E402
from synsmith.loop import SynSmith, SynSmithConfig, configure_logging  # noqa: E402


def main() -> None:
    configure_logging("WARNING")
    smoke_dir = REPO / "experiments" / "_diagnostics" / "smoke_fixes_f1_f6"
    if smoke_dir.exists():
        import shutil
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    cfg = SynSmithConfig.from_yaml(REPO / "examples/customer_support/config.yaml")
    cfg.iterations = 3
    cfg.samples_per_iteration = 16
    cfg.seed = 17
    cfg.run_dir = str(smoke_dir / "runs")
    cfg = build("full_attrforge", cfg)

    print(f"=== F1-F6 smoke on customer-support seed 17 ===", flush=True)
    t0 = time.time()
    forge = SynSmith(cfg)
    result = forge.run()
    elapsed = (time.time() - t0) / 60.0
    print(f"\n=== smoke run finished in {elapsed:.1f} min ===", flush=True)

    # Per-iter summary of F1-F6 evidence
    rd = Path(result.run_dir)
    report: list[str] = ["# F1-F6 smoke evidence report", ""]
    report.append(f"Run dir: `{rd}`")
    report.append(f"Wall-clock: {elapsed:.1f} min")
    report.append("")

    for it in range(cfg.iterations):
        iter_dir = rd / f"iter_{it:03d}"
        report.append(f"## iter {it}")
        report.append("")

        # F2: Pack shared_patterns
        pf = iter_dir / "pack_result.json"
        if pf.exists():
            p = json.loads(pf.read_text())
            report.append(f"### F2 - Pack Discriminator shared_patterns (should EXCLUDE target-distribution patterns)")
            report.append(f"- pack_accuracy: {p.get('pack_accuracy')}")
            patterns = p.get("shared_patterns", [])
            report.append(f"- {len(patterns)} patterns reported:")
            for sp in patterns[:8]:
                report.append(f"  - `{sp.get('pattern')!r}` (in {sp.get('n_pairs_observed')} pairs)")
            report.append("")

        # F1: Realism Discriminator verdict reasons
        rf = iter_dir / "realism_verdicts.jsonl"
        if rf.exists():
            verdicts = [json.loads(line) for line in rf.read_text().splitlines() if line.strip()]
            real_anchors_seen = sorted({v["sample_id"] for v in verdicts if v["sample_id"].startswith("R")})
            report.append(f"### F1 - Realism real-anchor IDs this iter (should be SAME across iters)")
            report.append(f"- {len(real_anchors_seen)} unique real anchor IDs: {real_anchors_seen}")
            report.append("")
            # Reasons sample
            synth_verdicts = [v for v in verdicts if v.get("prediction") == "synthetic"]
            report.append(f"### F1 - Discriminator reasons (sample of 3)")
            for v in synth_verdicts[:3]:
                report.append(f"  - {v.get('reason','')[:200]}")
            report.append("")

        # F6: Coverage Hole exemplars (should be stratified by class)
        cf = iter_dir / "coverage_holes.json"
        if cf.exists():
            c = json.loads(cf.read_text())
            holes = c.get("holes", [])
            from collections import Counter
            labels = Counter(h.get("label") for h in holes)
            report.append(f"### F6 - Coverage Hole exemplars (should be stratified across classes)")
            report.append(f"- classifier_auroc: {c.get('classifier_auroc')}")
            report.append(f"- {len(holes)} exemplars, label distribution: {dict(labels)}")
            for h in holes[:6]:
                report.append(f"  - [{h.get('label')}] {h.get('text','')[:120]}")
            report.append("")

    # F4/F5: rewritten prompt at iter_1 and iter_2 (lead with Preferred Phrasings?)
    for it in (1, 2):
        pp = rd / f"iter_{it:03d}" / "prompt.txt"
        if pp.exists():
            prompt = pp.read_text()
            report.append(f"### F4/F5 - rewritten generator prompt at iter {it} (look for 'Preferred phrasings' lead)")
            report.append("```")
            report.append(prompt[:2000])
            report.append("```")
            report.append("")

    # Final downstream macro F1 (for sanity vs n_test=10 baseline ~0.4)
    final_summary = rd / "iter_002" / "metrics.json"
    if final_summary.exists():
        m = json.loads(final_summary.read_text())
        report.append(f"### Sanity downstream macro F1 (iter_2 attribute-match-rate)")
        report.append(f"- attribute_match_rate: {m.get('attribute_match_rate')}")
        report.append(f"- combination_coverage: {m.get('combination_coverage')}")
        report.append("")

    out_md = smoke_dir / "evidence.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    print(f"\nEvidence report written to: {out_md}", flush=True)
    print(f"Inspect with: cat {out_md}", flush=True)


if __name__ == "__main__":
    main()
