"""Prompt templates used by every component.

Templates are kept here, separate from logic, so a researcher can iterate
on wording without touching code. Each template is a plain f-string; the
calling component substitutes fields explicitly.
"""
from attrforge.prompts.templates import (
    AUDITOR_SYSTEM,
    AUDITOR_USER_TEMPLATE,
    DISCRIMINATOR_SYSTEM,
    DISCRIMINATOR_USER_TEMPLATE,
    GENERATOR_INITIAL,
    GENERATOR_SYSTEM,
    GENERATOR_USER_TEMPLATE,
    UPDATER_SYSTEM,
    UPDATER_USER_TEMPLATE,
    VERIFIER_SYSTEM,
    VERIFIER_USER_TEMPLATE,
)

__all__ = [
    "AUDITOR_SYSTEM",
    "AUDITOR_USER_TEMPLATE",
    "DISCRIMINATOR_SYSTEM",
    "DISCRIMINATOR_USER_TEMPLATE",
    "GENERATOR_INITIAL",
    "GENERATOR_SYSTEM",
    "GENERATOR_USER_TEMPLATE",
    "UPDATER_SYSTEM",
    "UPDATER_USER_TEMPLATE",
    "VERIFIER_SYSTEM",
    "VERIFIER_USER_TEMPLATE",
]
