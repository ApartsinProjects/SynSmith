"""Baseline configurations for the SynSmith ablation study.

Each baseline is a stock ``SynSmithConfig`` with a different combination
of ``enable_*`` flags. Same harness, same dataset, same seed, same
metrics: only the critic stack changes.

This is exactly the protocol the project description's Section 10 calls
for, made executable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from synsmith.loop import SynSmithConfig


def naive(cfg: SynSmithConfig) -> SynSmithConfig:
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


def few_shot(cfg: SynSmithConfig) -> SynSmithConfig:
    """Naive + a larger few-shot pool, still no critics or updater."""
    out = naive(cfg)
    out.label = "few_shot"
    out.generator.num_few_shot = max(out.generator.num_few_shot, 8)
    return out


def self_critique(cfg: SynSmithConfig) -> SynSmithConfig:
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


def realism_only(cfg: SynSmithConfig) -> SynSmithConfig:
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


def diversity_only(cfg: SynSmithConfig) -> SynSmithConfig:
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


def attribute_only(cfg: SynSmithConfig) -> SynSmithConfig:
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


def full_classic(cfg: SynSmithConfig) -> SynSmithConfig:
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


def full_attrforge(cfg: SynSmithConfig) -> SynSmithConfig:
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

def no_pack(cfg: SynSmithConfig) -> SynSmithConfig:
    """full_attrforge minus Pack Discriminator."""
    out = full_attrforge(cfg)
    out.label = "no_pack"
    out.enable_pack = False
    return out


def no_mode_seeking(cfg: SynSmithConfig) -> SynSmithConfig:
    """full_attrforge minus Mode-Seeking critic."""
    out = full_attrforge(cfg)
    out.label = "no_mode_seeking"
    out.enable_mode_seeking = False
    return out


def no_mode_hunter(cfg: SynSmithConfig) -> SynSmithConfig:
    """full_attrforge minus Mode Hunter (no persistent banned-phrasings memory)."""
    out = full_attrforge(cfg)
    out.label = "no_mode_hunter"
    out.enable_mode_hunter = False
    return out


def no_coverage_hole(cfg: SynSmithConfig) -> SynSmithConfig:
    """full_attrforge minus Coverage Hole Finder."""
    out = full_attrforge(cfg)
    out.label = "no_coverage_hole"
    out.enable_coverage_hole = False
    return out


def full_attrforge_3judge(cfg: SynSmithConfig) -> SynSmithConfig:
    """full_attrforge + 3-judge debate Realism Critic via OpenRouter.

    Same seven critics, but the standard single-judge Realism Discriminator
    is wrapped in a 3-judge debate (gpt-4o-mini + claude-3-haiku +
    gemini-flash-1.5) with Kolmogorov-Smirnov adaptive stopping (scout
    D3.1, arXiv:2510.12697). All three judges are served through
    OpenRouter (https://openrouter.ai/api/v1) with one OPENROUTER_API_KEY.

    The debate-critic baseline is opt-in because it requires
    OPENROUTER_API_KEY to be set, and because it costs ~3x the realism-
    critic API budget vs the single-judge default. Use it for the
    realism-bias-control comparison in the paper's §10 boundary discussion.
    """
    out = full_attrforge(cfg)
    out.label = "full_attrforge_3judge"
    # The actual debate-critic wiring lives in synsmith.critics.debate_discriminator
    # and is invoked by the post-hoc audit script
    # scripts/debate_realism_audit.py rather than by the live loop, so
    # the existing run_experiments.py harness does not need changes.
    return out


def full_attrforge_vs(cfg: SynSmithConfig) -> SynSmithConfig:
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


def no_pack_vs(cfg: SynSmithConfig) -> SynSmithConfig:
    """no_pack + Verbalized Sampling. Task #69 candidate clear-winner config.

    Per-datapoint root-cause analysis on customer-support item [7] showed
    the Pack Discriminator suppressed delivery-style complaint phrasings,
    turning 9/10 correct into 6/10 under the full stack. Removing Pack
    frees the generator for the structural variation VS was designed to
    add. Hypothesis: the combination beats both full_attrforge (with Pack)
    and no_pack (without VS) on worst-class F1.
    """
    out = no_pack(cfg)
    out.label = "no_pack_vs"
    out.generator.verbalized_sampling = True
    out.generator.vs_n_candidates = 5
    out.generator.vs_sample_strategy = "weighted"
    return out


def full_attrforge_sibling(cfg: SynSmithConfig) -> SynSmithConfig:
    """full_attrforge + Class-Discriminability check in Verifier. Task #73.

    On fine-grained classification (Banking77's 10 sibling card-intents),
    the standard Verifier confirms the requested attribute holds but does
    not check the sample is DISTINGUISHABLE from sibling classes. The
    sibling-rejection variant shows the Verifier real anchors from the
    nearest sibling classes and requires REJECTION when the synth is
    equally compatible with a sibling.
    """
    out = full_attrforge(cfg)
    out.label = "full_attrforge_sibling"
    # Wired via VerifierConfig.enable_sibling_rejection in loop.py builder.
    out.verifier_sibling_rejection = True
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
    # Task #69: no_pack + VS (candidate clear-winner).
    "no_pack_vs": no_pack_vs,
    # Task #73: Class-Discriminability sibling-rejection Verifier.
    "full_attrforge_sibling": full_attrforge_sibling,
    # 3-judge debate Realism Critic via OpenRouter (scout D3.1).
    "full_attrforge_3judge": full_attrforge_3judge,
}


def build(name: str, cfg: SynSmithConfig) -> SynSmithConfig:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline: {name!r}; choices: {sorted(BASELINES)}")
    return BASELINES[name](cfg)
