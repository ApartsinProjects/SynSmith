"""End-to-end loop test using the echo backend.

Confirms that all components wire together, produce the expected run
directory layout, and don't crash on the bookkeeping logic.
"""
from __future__ import annotations

from pathlib import Path

from attrforge.loop import AttrForge, AttrForgeConfig

REPO = Path(__file__).resolve().parents[1]


def test_offline_loop_smoke(tmp_path):
    cfg = AttrForgeConfig.from_yaml(REPO / "examples/customer_support/config.echo.yaml")
    cfg.run_dir = str(tmp_path)
    cfg.iterations = 2
    cfg.samples_per_iteration = 4

    forge = AttrForge(cfg)
    result = forge.run()

    assert len(result.iterations) == 2
    run_dir = Path(result.run_dir)
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "iter_000" / "samples.jsonl").exists()
    assert (run_dir / "iter_001" / "samples.jsonl").exists()
    assert (run_dir / "iter_000" / "metrics.json").exists()

    # Every iteration must have written samples_per_iteration synthetic samples.
    for it in result.iterations:
        assert len(it.samples) == cfg.samples_per_iteration
        assert all(s.text for s in it.samples)


def test_metrics_present():
    cfg = AttrForgeConfig.from_yaml(REPO / "examples/customer_support/config.echo.yaml")
    forge = AttrForge(cfg)
    result = forge.run(iterations=1)
    m = result.iterations[0].metrics
    assert "attribute_match_rate" in m
    assert "discriminator_accuracy" in m
    assert "near_duplicate_rate" in m
    assert "combination_coverage" in m
