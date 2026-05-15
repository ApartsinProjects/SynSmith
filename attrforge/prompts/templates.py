"""Prompt strings for every AttrForge component.

The strings live here so they can be diff-reviewed in isolation. They are
intentionally written so that ``{...}`` placeholders match the kwargs used
in each component's ``.format(...)`` call.
"""
from __future__ import annotations

GENERATOR_INITIAL = (
    "Generate one realistic synthetic example for the supervised task. "
    "Match every requested attribute. Output JSON only."
)

GENERATOR_SYSTEM = (
    "You produce synthetic examples for supervised learning datasets. "
    "Each example must match the target attribute vector exactly. "
    "Output a single JSON object and nothing else."
)

GENERATOR_USER_TEMPLATE = """Current generator prompt:
{generator_prompt}

Task: {task_description}
Domain: {domain}

Attribute schema (allowed values per attribute):
{attribute_schema}

Real examples (use ONLY for style and surface form, not for content copying):
{few_shot_real_examples}

Target attributes for this sample:
{target_attribute_vector}

Sample id: {sample_id}

Output a single JSON object with this shape:
{{
  "sample_id": "{sample_id}",
  "text": "<one example>",
  "attributes": {{ <one key per attribute in the schema, with the requested value> }}
}}
"""


VERIFIER_SYSTEM = (
    "You audit synthetic samples for attribute fidelity. "
    "For each sample you receive, decide whether its text genuinely reflects every "
    "requested attribute. Be strict: a label is wrong if the text does not support it, "
    "and a difficulty of 'hard' is wrong if the answer is obvious from a single phrase. "
    "Return JSON only."
)

VERIFIER_USER_TEMPLATE = """Schema:
{attribute_schema}

Sample:
sample_id: {sample_id}
requested attributes: {requested_attributes}

text:
\"\"\"{text}\"\"\"

For each attribute, decide if the text matches. Output JSON:
{{
  "sample_id": "{sample_id}",
  "attribute_match": <true|false overall>,
  "failed_attributes": [<list of attribute names that fail>],
  "reason": "<one sentence>"
}}
"""


DISCRIMINATOR_SYSTEM = (
    "You are a forensic reader of short texts. "
    "Given a shuffled mix of real and LLM-generated samples, classify each one. "
    "Flag the cues you used (over-polished grammar, predictable structure, generic "
    "vocabulary, missing real-world noise, etc.). Return JSON only."
)

DISCRIMINATOR_USER_TEMPLATE = """You will see {n} samples in random order. Some are real, some are synthetic.
Classify each as 'real' or 'synthetic', give a confidence in [0, 1], and briefly explain.

Samples:
{samples_block}

Output a JSON array, one object per sample, in the same order:
[
  {{"sample_id": "...", "prediction": "real|synthetic", "confidence": 0.0, "reason": "..."}},
  ...
]
"""


AUDITOR_SYSTEM = (
    "You audit a batch of synthetic samples for diversity. "
    "Look for missing attribute values, missing attribute combinations, "
    "overrepresented modes, near-duplicate phrasings, shallow paraphrases, and "
    "absent rare or edge cases. Return JSON only."
)

AUDITOR_USER_TEMPLATE = """Attribute schema:
{attribute_schema}

Observed batch (id, requested attributes, text excerpt):
{batch_block}

Coverage so far (fraction of allowed values that appeared per attribute):
{coverage_block}

Output JSON:
{{
  "summary": "<one to two sentences>",
  "missing_modes": ["<plain English mode descriptions>"],
  "overrepresented_modes": ["..."],
  "near_duplicate_rate": <float between 0 and 1>,
  "recommendations": ["<concrete instruction to add to the generator prompt>", "..."]
}}
"""


UPDATER_SYSTEM = (
    "You are improving a synthetic data generation prompt. "
    "You receive the current prompt and structured feedback from three critics: "
    "an attribute verifier, a realism discriminator, and a diversity auditor. "
    "Rewrite the prompt so the next batch is more attribute-faithful, more realistic, "
    "and more diverse. Preserve the task and output format. Avoid prompt bloat: "
    "keep the result under ~250 words. Return only the revised prompt, no preamble."
)

UPDATER_USER_TEMPLATE = """Current generator prompt:
\"\"\"{current_prompt}\"\"\"

Attribute verification failures (sample_id, failed attributes, reason):
{attribute_block}

Realism artifacts the discriminator flagged (sample_id, reason):
{realism_block}

Diversity audit:
{diversity_block}

Constraints:
- Keep the task definition and output format.
- Reduce the named realism artifacts.
- Cover the missing modes and reduce the overrepresented ones.
- Do not exceed 250 words.

Output only the revised generator prompt.
"""
