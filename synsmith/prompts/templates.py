"""Prompt strings for every SynSmith component.

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
    "You audit synthetic samples for attribute fidelity by reading the "
    "text FIRST, identifying what the text actually exhibits, and only THEN "
    "comparing to the requested attributes.\n\n"
    "CRITICAL: do NOT trust the requested attributes as a starting point. "
    "Synthetic samples are routinely mislabeled by the generator: ironic "
    "praise may be labeled 'positive', a contradiction may be labeled "
    "'neutral', a card-arrival query may be labeled 'card_swallowed'. "
    "Your job is to detect those mismatches, not to confirm them.\n\n"
    "Procedure for each sample:\n"
    "1. Read the text. State (silently, to yourself) what attributes a "
    "blind human annotator would assign to this text. Focus on the "
    "NUANCES: sentiment carried by surface words AND by structure/irony; "
    "register and tone; what the text is actually about vs what it claims "
    "to be about; whether the request's claimed attribute (e.g. 'hard') "
    "is supportable from the text or whether the text trivially gives it "
    "away.\n"
    "2. Compare your reading to the requested attributes. For each "
    "attribute, decide whether the text matches the request - or whether "
    "the request is wrong.\n"
    "3. If a 'Sibling classes' block is present in the user message, "
    "verify DISTINGUISHABILITY from every sibling class shown. A sample "
    "that satisfies the requested attribute but is equally compatible "
    "with one or more sibling classes (would plausibly be labeled as the "
    "sibling by a blind annotator who sees only the text) must be "
    "REJECTED by including the class attribute in failed_attributes, "
    "with the reason naming the sibling it is ambiguous with. "
    "Attribute fidelity alone is not sufficient when sibling-rejection "
    "is active; class-discriminability is required.\n"
    "4. Return a structured verdict. Return JSON only."
)

VERIFIER_USER_TEMPLATE = """Schema:
{attribute_schema}

Real-distribution empirical anchors (the attribute value labels MEAN what
the real examples show; do NOT use a generic-English interpretation):
{real_anchor_block}

{sibling_anchor_block}

Sample:
sample_id: {sample_id}

text:
\"\"\"{text}\"\"\"

Requested attributes (the generator's CLAIM about this sample; verify
against the text AND against the real-distribution anchors above; do
not assume the request is correct):
{requested_attributes}

Read the text first. Identify what attributes a blind annotator would
assign by COMPARING the text to the real-distribution anchors for each
attribute value. If a 'Sibling classes' block was shown above, also
verify the sample is DISTINGUISHABLE from every sibling class: a sample
equally compatible with a sibling must be REJECTED by including the
class attribute in failed_attributes. THEN compare to the requested
attributes. For each attribute, decide if the text matches. Output JSON:
{{
  "sample_id": "{sample_id}",
  "attribute_match": <true|false overall>,
  "failed_attributes": [<list of attribute names that fail>],
  "reason": "<one sentence; cite a specific feature of the text OR a
             specific contrast with the real-distribution anchors OR
             the sibling-class the sample is ambiguous with>"
}}
"""


DISCRIMINATOR_SYSTEM = (
    "You are a forensic reader of short texts. You receive a shuffled mix "
    "of real samples (drawn from a SPECIFIC TARGET DISTRIBUTION) and "
    "synthetic LLM-generated samples. Your job is to classify each sample "
    "against the IDENTIFIED REAL DISTRIBUTION, not against a generic "
    "'plausible English' baseline.\n\n"
    "Procedure:\n"
    "1. First, scan the samples to identify the target distribution's "
    "register, vocabulary, length, syntactic patterns, and stylistic tics. "
    "What kind of writing is the real seed? Formal review? Casual chat? "
    "Newswire? Critic prose? Domain-specific jargon?\n"
    "2. Then classify each sample as 'real' or 'synthetic' by comparing "
    "it to THAT identified distribution. A casually-written text is "
    "SYNTHETIC if the real samples are formal; a polished text is REAL "
    "if the real samples are professional writing. Register mismatch is "
    "as strong a cue as grammatical artifacts.\n"
    "3. Flag the cues you used in the reason field, naming the register "
    "mismatch when relevant (e.g., 'casual viewer voice; real samples "
    "are professional critic prose').\n"
    "Return JSON only."
)

DISCRIMINATOR_USER_TEMPLATE = """You will see {n} samples in random order. Some are real, some are synthetic.
Identify the real distribution's register and style FIRST by scanning all
samples; classify each against that identified register, not against a
generic 'plausible English' baseline. Register mismatch is itself a
synthetic cue.

Samples:
{samples_block}

Output a JSON array, one object per sample, in the same order:
[
  {{"sample_id": "...", "prediction": "real|synthetic", "confidence": 0.0, "reason": "..."}},
  ...
]
"""


AUDITOR_SYSTEM = (
    "You audit a batch of synthetic samples for diversity, reading the "
    "TEXTS first and judging diversity by what the texts actually say, "
    "not by what the requested-attribute labels claim.\n\n"
    "CRITICAL: a batch can be 'attribute-diverse' on paper (every label "
    "value represented) while being SURFACE-diverse on paper but "
    "NUANCE-redundant in practice (every 'positive' sample uses the same "
    "construction; every 'hard' sample telegraphs the answer in a stock "
    "phrase). The label-level coverage check is already computed "
    "deterministically; YOUR job is to find the nuance-level redundancy "
    "the deterministic check cannot see.\n\n"
    "Look for missing attribute values, missing attribute combinations, "
    "overrepresented surface patterns, near-duplicate phrasings, shallow "
    "paraphrases that reuse the same template with one word swapped, and "
    "absent rare or edge cases. Return JSON only."
)

AUDITOR_USER_TEMPLATE = """Attribute schema:
{attribute_schema}

Observed batch (id, text excerpt, requested attributes shown LAST so you
read the text first):
{batch_block}

Coverage so far (deterministic; fraction of allowed values that appeared
per attribute):
{coverage_block}

Read the texts first. Judge nuance-level diversity (surface patterns,
phrasings, sentence structures, vocabulary plateaus) rather than just
label-level coverage. Output JSON:
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

Pack-level patterns the pack discriminator detected across batches of samples
(these are visible only when looking at multiple samples together):
{pack_block}

Mode-seeking responsiveness: ratio of text distance to attribute distance.
{mode_seeking_block}

Persistent banned phrasings library (NEVER reuse these in any future prompt):
{banned_block}

Real exemplars the synthetic distribution is failing to cover (PREFERRED
phrasings the generator MUST steer toward in the next batch; these are
real surface patterns from the target distribution that the synth is
missing):
{coverage_hole_block}

Constraints:
- Keep the task definition and output format.
- Reduce the named realism artifacts.
- Cover the missing modes and reduce the overrepresented ones.
- Address the pack-level patterns by introducing structural variation.
- If mode-seeking ratio is low (< 0.5), explicitly amplify the surface
  effect of attribute changes.
- Do not reuse any banned phrasing. Add a "Forbidden phrasings" block to
  the generator prompt if helpful.
- Add a "Preferred phrasings" block (or analogous positive guidance) to
  the generator prompt that lists the real-exemplar fragments shown
  above. The generator MUST attempt to produce samples that USE those
  phrasings or close paraphrases of them. These are the target
  distribution's surface signal; the banned list alone cannot replace
  positive register guidance.
- Banned-phrasings guidance is NEGATIVE (what to avoid); preferred-
  phrasings guidance is POSITIVE (what to emulate). Both belong in the
  new prompt when their respective inputs are non-empty.
- Do not exceed 320 words.

Output only the revised generator prompt.
"""
