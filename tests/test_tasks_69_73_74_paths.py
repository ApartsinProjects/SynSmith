"""Code-path validation for tasks #69 / #73 / #74.

These tests exercise the new prompt-template formatting, baseline
wiring, and schema-loading paths WITHOUT making any LLM call. They are
the substitute for the live smoke when the OpenAI quota is depleted:
they catch format-string mismatches, missing placeholder dict entries,
schema parsing errors, and BASELINES registry misses.

They DO NOT validate the SUBSTANCE of any change (sibling rejection
actually working, topic coverage actually improving downstream F1, no_pack
+ VS actually winning on worst-class F1). That validation requires real
LLM calls; defer until API budget is restored.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from synsmith.baselines import BASELINES, build
from synsmith.critics.verifier import AttributeVerifier, VerifierConfig
from synsmith.loop import SynSmithConfig
from synsmith.prompts.templates import (
    GENERATOR_SYSTEM,
    VERIFIER_SYSTEM,
    VERIFIER_USER_TEMPLATE,
)
from synsmith.schema import AttributeSchema, RealExample, load_jsonl

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Task #69: no_pack + VS baseline wiring
# ---------------------------------------------------------------------------


def test_69_no_pack_vs_baseline_is_registered() -> None:
    assert "no_pack_vs" in BASELINES, (
        "Task #69: no_pack_vs must appear in the BASELINES registry so "
        "scripts/run_experiments.py can route the sweep."
    )


def test_69_no_pack_vs_disables_pack_and_enables_vs() -> None:
    base = SynSmithConfig(
        schema_path="examples/customer_support/schema.yaml",
        real_examples_path="experiments/_splits/real_train.jsonl",
    )
    out = build("no_pack_vs", base)
    assert out.label == "no_pack_vs"
    assert out.enable_pack is False, "no_pack_vs MUST disable the Pack Discriminator"
    assert out.generator.verbalized_sampling is True, "no_pack_vs MUST enable VS"
    assert out.generator.vs_n_candidates == 5
    assert out.generator.vs_sample_strategy == "weighted"
    # Sanity: other GAN-style critics stay ON (it's full_attrforge minus pack + VS).
    assert out.enable_verifier is True
    assert out.enable_discriminator is True
    assert out.enable_mode_seeking is True
    assert out.enable_mode_hunter is True
    assert out.enable_coverage_hole is True


# ---------------------------------------------------------------------------
# Task #73: Class-Discriminability sibling-rejection in Verifier
# ---------------------------------------------------------------------------


def test_73_baseline_registered_and_flag_propagates() -> None:
    assert "full_attrforge_sibling" in BASELINES
    base = SynSmithConfig(
        schema_path="examples/banking77/schema.yaml",
        real_examples_path="experiments/_splits/banking77_real_train.jsonl",
    )
    out = build("full_attrforge_sibling", base)
    assert out.label == "full_attrforge_sibling"
    assert out.verifier_sibling_rejection is True
    # All seven critics stay on; only Verifier config is altered.
    assert out.enable_pack is True
    assert out.enable_coverage_hole is True


def test_73_verifier_user_template_accepts_new_placeholder() -> None:
    """The user template must format with the new sibling_anchor_block kwarg.

    A KeyError here would be a missing placeholder; an extra-key TypeError
    would be a leftover {kwarg} in the template that we forgot to feed.
    """
    msg = VERIFIER_USER_TEMPLATE.format(
        attribute_schema="(yaml)",
        sample_id="S1",
        requested_attributes='{"intent": "card_not_working"}',
        text="my card has stopped working at every machine since yesterday",
        real_anchor_block="(some real anchors here)",
        sibling_anchor_block=(
            "Sibling classes of intent='card_not_working' (...):"
            "\n  intent='card_swallowed':\n    - ATM swallowed my card."
        ),
    )
    assert "card_not_working" in msg
    assert "Sibling classes" in msg
    assert "{sibling_anchor_block}" not in msg, "placeholder must be substituted"
    # Verifier system prompt must contain the sibling-rejection clause so
    # the LLM knows what to do with the block.
    assert "Sibling classes" in VERIFIER_SYSTEM


def test_73_sibling_anchors_render_when_enabled() -> None:
    """Banking77 sibling-anchor block contains real anchors per sibling class."""
    schema = AttributeSchema.from_yaml(REPO / "examples/banking77/schema.yaml")
    reals = [
        RealExample.model_validate(d)
        for d in load_jsonl(REPO / "experiments/_splits/banking77_real_train.jsonl")
    ]
    cfg_on = VerifierConfig(enable_sibling_rejection=True, k_sibling_anchors=1, seed=17)
    v_on = AttributeVerifier(client=None, schema=schema, real_examples=reals, config=cfg_on)
    block = v_on._format_sibling_anchors({"intent": "card_not_working"})
    # 9 sibling classes other than card_not_working should each appear.
    for sib in [
        "card_arrival",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "card_swallowed",
        "declined_card_payment",
        "lost_or_stolen_card",
        "pending_card_payment",
        "top_up_failed",
    ]:
        assert f"intent='{sib}'" in block, f"sibling class {sib} missing from block"
    assert "REJECT" in block, "rejection instruction must be in the block lead-in"


def test_73_sibling_anchors_disabled_returns_inert_string() -> None:
    schema = AttributeSchema.from_yaml(REPO / "examples/banking77/schema.yaml")
    reals = [
        RealExample.model_validate(d)
        for d in load_jsonl(REPO / "experiments/_splits/banking77_real_train.jsonl")
    ]
    cfg_off = VerifierConfig(enable_sibling_rejection=False, seed=17)
    v_off = AttributeVerifier(client=None, schema=schema, real_examples=reals, config=cfg_off)
    block = v_off._format_sibling_anchors({"intent": "card_not_working"})
    assert "disabled" in block
    assert "intent='card_swallowed'" not in block, (
        "with sibling rejection off, NO sibling anchors should leak into the prompt"
    )


# ---------------------------------------------------------------------------
# Task #74: Topic-axis schema for TREC
# ---------------------------------------------------------------------------


def test_74_topic_schema_parses_and_has_topic_axis() -> None:
    schema_path = REPO / "examples/trec/schema_topic.yaml"
    assert schema_path.exists(), "schema_topic.yaml must be derived first via scripts/derive_topic_axis.py"
    schema = AttributeSchema.from_yaml(schema_path)
    assert "topic" in schema.attributes
    topics = schema.values("topic")
    assert len(topics) >= 4, "expected at least a handful of topic clusters"
    # All topic labels should be schema-safe (no spaces or quotes).
    for t in topics:
        assert " " not in t and "'" not in t


def test_74_topic_config_points_at_topic_schema() -> None:
    cfg_path = REPO / "examples/trec/config_topic.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["schema_path"].endswith("schema_topic.yaml")


def test_74_loadable_via_synsmithconfig() -> None:
    cfg = SynSmithConfig.from_yaml(REPO / "examples/trec/config_topic.yaml")
    assert cfg.schema_path.endswith("schema_topic.yaml")
    # Sanity: it instantiates with the topic schema attached.
    schema = AttributeSchema.from_yaml(cfg.schema_path)
    assert "topic" in schema.attributes
