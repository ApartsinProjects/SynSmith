"""Baseline configurations for the AttrForge ablation study.

Each baseline is a stock ``AttrForgeConfig`` with a different combination
of ``enable_*`` flags. Same harness, same dataset, same seed, same
metrics: only the critic stack changes.

This is exactly the protocol the project description's Section 10 calls
for, made executable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from attrforge.loop import AttrForgeConfig


def naive(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """One static prompt, one round of generation, no critics, no updater."""
    out = deepcopy(cfg)
    out.label = "naive"
    out.iterations = 1
    out.enable_verifier = False
    out.enable_discriminator = False
    out.enable_auditor = False
    out.enable_pack = False
    out.enable_mode_seeking = False
    out.enable_mode_hunter = False
    out.enable_coverage_hole = False
    out.enable_updater = False
    return out


def few_shot(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """Naive + a larger few-shot pool, still no critics or updater."""
    out = naive(cfg)
    out.label = "few_shot"
    out.generator.num_few_shot = max(out.generator.num_few_shot, 8)
    return out


def self_critique(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """No external critics, but the updater runs each round with empty critic feedback.

    This is the "generator critiques and improves its own prompt" baseline:
    no attribute, realism, or diversity audit; only the updater's own
    rewrite signal. Simulates the common "ask the LLM to make the prompt
    better" approach.
    """
    out = deepcopy(cfg)
    out.label = "self_critique"
    out.enable_verifier = False
    out.enable_discriminator = False
    out.enable_auditor = True  # deterministic auditor only (cheap, no LLM call)
    out.enable_pack = False
    out.enable_mode_seeking = False
    out.enable_mode_hunter = False
    out.enable_coverage_hole = False
    out.enable_updater = True
    return out


def realism_only(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """Realism discriminator + updater. No attribute or diversity feedback."""
    out = deepcopy(cfg)
    out.label = "realism_only"
    out.enable_verifier = False
    out.enable_discriminator = True
    out.enable_auditor = True  # deterministic only (no LLM call)
    out.enable_pack = False
    out.enable_mode_seeking = False
    out.enable_mode_hunter = False
    out.enable_coverage_hole = False
    out.enable_updater = True
    return out


def diversity_only(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """Auditor + (optionally) GAN-style diversity adversaries, no verifier or realism critic."""
    out = deepcopy(cfg)
    out.label = "diversity_only"
    out.enable_verifier = False
    out.enable_discriminator = False
    out.enable_auditor = True
    out.enable_pack = False
    out.enable_mode_seeking = True
    out.enable_mode_hunter = False
    out.enable_coverage_hole = True
    out.enable_updater = True
    return out


def attribute_only(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """Verifier + updater. Tests attribute fidelity in isolation."""
    out = deepcopy(cfg)
    out.label = "attribute_only"
    out.enable_verifier = True
    out.enable_discriminator = False
    out.enable_auditor = True  # deterministic
    out.enable_pack = False
    out.enable_mode_seeking = False
    out.enable_mode_hunter = False
    out.enable_coverage_hole = False
    out.enable_updater = True
    return out


def full_classic(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """The 3-critic version (verifier + discriminator + auditor), no GAN extras."""
    out = deepcopy(cfg)
    out.label = "full_classic"
    out.enable_verifier = True
    out.enable_discriminator = True
    out.enable_auditor = True
    out.enable_pack = False
    out.enable_mode_seeking = False
    out.enable_mode_hunter = False
    out.enable_coverage_hole = False
    out.enable_updater = True
    return out


def full_attrforge(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """All 7 critics + updater. The full proposed system."""
    out = deepcopy(cfg)
    out.label = "full_attrforge"
    out.enable_verifier = True
    out.enable_discriminator = True
    out.enable_auditor = True
    out.enable_pack = True
    out.enable_mode_seeking = True
    out.enable_mode_hunter = True
    out.enable_coverage_hole = True
    out.enable_updater = True
    return out


# Leave-one-out ablations: full_attrforge minus one GAN-style adversary.
# Used in the per-critic attribution analysis (paper §7.3 / leave-one-out).

def no_pack(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """full_attrforge minus Pack Discriminator."""
    out = full_attrforge(cfg)
    out.label = "no_pack"
    out.enable_pack = False
    return out


def no_mode_seeking(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """full_attrforge minus Mode-Seeking critic."""
    out = full_attrforge(cfg)
    out.label = "no_mode_seeking"
    out.enable_mode_seeking = False
    return out


def no_mode_hunter(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """full_attrforge minus Mode Hunter (no persistent banned-phrasings memory)."""
    out = full_attrforge(cfg)
    out.label = "no_mode_hunter"
    out.enable_mode_hunter = False
    return out


def no_coverage_hole(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """full_attrforge minus Coverage Hole Finder."""
    out = full_attrforge(cfg)
    out.label = "no_coverage_hole"
    out.enable_coverage_hole = False
    return out


def full_attrforge_vs(cfg: AttrForgeConfig) -> AttrForgeConfig:
    """full_attrforge + Verbalized Sampling generator (scout D1.1).

    Asks the generator to verbalize 5 candidates with self-reported
    probabilities per call, then samples one. Single-prompt fix demonstrated
    to yield 1.6-2.1x diversity gain on creative-writing in arXiv:2510.01171.
    """
    out = full_attrforge(cfg)
    out.label = "full_attrforge_vs"
    out.generator.verbalized_sampling = True
    out.generator.vs_n_candidates = 5
    out.generator.vs_sample_strategy = "weighted"
    return out


BASELINES = {
    "naive": naive,
    "few_shot": few_shot,
    "self_critique": self_critique,
    "attribute_only": attribute_only,
    "realism_only": realism_only,
    "diversity_only": diversity_only,
    "full_classic": full_classic,
    "full_attrforge": full_attrforge,
    # Leave-one-out ablations.
    "no_pack": no_pack,
    "no_mode_seeking": no_mode_seeking,
    "no_mode_hunter": no_mode_hunter,
    "no_coverage_hole": no_coverage_hole,
    # Verbalized-Sampling variant (scout D1.1).
    "full_attrforge_vs": full_attrforge_vs,
}


def build(name: str, cfg: AttrForgeConfig) -> AttrForgeConfig:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name!r}; choices: {sorted(BASELINES)}")
    return BASELINES[name](cfg)
