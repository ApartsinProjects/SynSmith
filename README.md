<div align="center">

<img src="assets/hero.png" alt="AttrForge: a forge anvil with a glowing crystalline prompt being shaped, surrounded by three critic spirits (cyan attribute verifier, amber diversity auditor, magenta realism discriminator) sending structured feedback back into the prompt." width="100%" />

<br />

# AttrForge

**Multi-Objective Prompt Debugging for Realistic, Diverse, and Attribute-Controlled Synthetic Data Generation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: research preview](https://img.shields.io/badge/status-research%20preview-orange.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

*An open-source framework that reframes synthetic data generation as iterative,*
*critic-guided prompt optimization.*

[Overview](#overview) · [Method](#method) · [Quickstart](#quickstart) · [Architecture](#architecture) · [Research questions](#research-questions) · [Citation](#citation)

</div>

---

## Overview

Large language models are increasingly used to bootstrap labeled data when real
examples are scarce, expensive, private, or hard to annotate. In practice,
naive prompting produces datasets that are repetitive, over-polished, and only
shallowly tied to the requested labels. The standard fix is to hand-tune the
prompt, which scales poorly and loses information about *why* a given sample
went wrong.

**AttrForge** treats synthetic data generation as a closed-loop optimization
problem. A generator LLM produces samples conditioned on explicit target
attribute vectors. Three LLM critics, an *attribute verifier*, a *realism
discriminator*, and a *diversity auditor*, score the batch along independent
axes. A prompt updater consumes their structured feedback and rewrites the
generator prompt for the next round.

The result is a GAN-style process in which the optimized variable is the
prompt rather than the weights, with three simultaneous objectives:

> **attribute fidelity · realism · diversity**

---

## Method

The system implements one of the few prompt-optimization loops with multiple,
*orthogonal* critic objectives. Each iteration runs the following procedure:

```text
P_t  : current generator prompt
A    : attribute schema  (e.g. label × difficulty × ambiguity × style × noise)
R    : small real dataset (50–200 examples)

# 1. plan
targets ← AttributePlanner(A, history) ........... target attribute vectors
# 2. generate
S_t ← Generator(P_t, R, targets) ................. synthetic samples
# 3. critique
V_t ← AttributeVerifier(S_t, A) .................. per-sample attribute audits
D_t ← RealismDiscriminator(R ∪ S_t) .............. real-vs-synthetic verdicts
C_t ← DiversityAuditor(S_t, A) ................... batch-level coverage report
# 4. update
P_{t+1} ← PromptUpdater(P_t, V_t, D_t, C_t)
```

The discriminator's accuracy on the mixed batch is the realism signal: a
healthy run drives it toward chance level. The auditor reports per-attribute
coverage, near-duplicate rate, and named missing modes. The verifier flags
specific failed attributes per sample. All three feedback signals are
serialized into the updater's prompt template, so the rewrite is grounded in
named failures rather than free-form self-critique.

### Why three critics

| Critic                  | Question it answers                                | Failure mode it prevents                                     |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Attribute Verifier      | Does the text actually reflect the requested vector? | Metadata-only labels: the right attribute string with mismatched text |
| Realism Discriminator   | Can a judge separate synthetic from real?          | Over-polished, template-y, telltale LLM phrasing             |
| Diversity Auditor       | Does the batch cover the attribute space?          | Mode collapse, shallow paraphrases, missing rare/edge cases  |

Removing any one critic should produce a measurably degraded distribution along
its axis; this is the central ablation the framework is designed to test.

---

## Quickstart

### Install

```bash
git clone https://github.com/yourusername/attrforge.git
cd attrforge
pip install -e ".[openai]"        # or .[anthropic], or .[all]
```

### Dry run (no API key required)

The `echo` backend exercises the full pipeline offline against a stubbed model.
Useful for CI, smoke tests, and reading the on-disk run layout.

```bash
attrforge run examples/customer_support/config.echo.yaml
```

### Real run with OpenAI

```bash
export OPENAI_API_KEY=sk-...
attrforge run examples/customer_support/config.yaml --iterations 5
attrforge inspect runs/<run_id>
```

### Programmatic API

```python
from attrforge import AttrForge

forge = AttrForge.from_config("examples/customer_support/config.yaml")
result = forge.run(iterations=5)

print(result.final_prompt)
print(result.metric_history[-1])
# {'attribute_match_rate': 0.92,
#  'discriminator_accuracy': 0.58,
#  'near_duplicate_rate': 0.04,
#  'combination_coverage': 0.83, ...}
```

---

## Architecture

```text
                       ┌─────────────────────────┐
                       │   Attribute Schema A    │
                       │  + small real set R     │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Attribute Planner     │      (stratified or
                       │                         │       coverage-gap)
                       └────────────┬────────────┘
                                    │  target attribute vectors
                                    ▼
                       ┌─────────────────────────┐
                       │      Generator G        │ ◄────────────────────────┐
                       │  (current prompt P_t)   │                          │
                       └────────────┬────────────┘                          │
                                    │  synthetic samples S_t                │
              ┌─────────────────────┼─────────────────────┐                 │
              ▼                     ▼                     ▼                 │
   ┌────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐    │
   │ Attribute Verifier │ │  Realism           │ │  Diversity Auditor  │    │
   │  per-sample        │ │  Discriminator     │ │  batch-level        │    │
   │  failed-attr list  │ │  acc, conf, reason │ │  coverage, modes    │    │
   └─────────┬──────────┘ └─────────┬──────────┘ └──────────┬──────────┘    │
             │                      │                       │                │
             └──────────────┬───────┴──────────────┬────────┘                │
                            ▼                      ▼                         │
                       ┌─────────────────────────────────┐                   │
                       │       Prompt Updater U          │                   │
                       │   P_{t+1} = U(P_t, feedback)    │ ──────────────────┘
                       └─────────────────────────────────┘
```

Every component lives behind a small typed interface (`attrforge/schema.py`).
Each can be swapped without touching the loop, which is what makes the
ablations cheap. The whole run, prompts, targets, samples, verdicts, reports,
and metrics, is persisted under `runs/<id>/` so experiments are reproducible.

```
runs/<id>/
  config.yaml
  schema.yaml
  real_examples.jsonl
  manifest.json                 ← metric_history, prompt_history
  iter_000/
    prompt.txt
    targets.jsonl
    samples.jsonl
    attribute_verdicts.jsonl
    realism_verdicts.jsonl
    diversity_report.json
    metrics.json
  iter_001/ ...
```

---

## Research questions

This codebase is designed to make each of the following questions answerable
with an ablation flag, not a re-implementation.

**RQ1. Attribute fidelity.**
Does iterative prompt debugging improve the rate at which generated samples
satisfy their requested attribute vector? Metrics: per-attribute precision,
recall, F1, and total attribute-match accuracy.

**RQ2. Realism.**
Does discriminator-guided prompt debugging drive an LLM judge's real-vs-
synthetic accuracy toward chance? Metrics: discriminator accuracy, synthetic
detection rate, calibration. A successful loop produces near-50% accuracy in
balanced settings, with named artifacts disappearing over iterations.

**RQ3. Diversity.**
Does coverage-guided debugging produce broader semantic coverage? Metrics:
per-attribute entropy (normalized), combination coverage across pairs of
attributes, embedding/TF-IDF near-duplicate rate, and a qualitative audit of
missing modes.

**RQ4. Downstream usefulness.**
Does AttrForge-generated data improve held-out real-test performance for a
downstream classifier, especially on rare, hard, or ambiguous slices? Baselines
include naive prompting, few-shot prompting, self-critique, diversity-only,
realism-only, and human prompt refinement.

---

## Baselines included

| Baseline                | Switch                            | Description                                              |
| ----------------------- | --------------------------------- | -------------------------------------------------------- |
| Naive prompting         | `iterations: 1`                   | One manually written prompt, no critic loop              |
| Few-shot prompting      | `generator.num_few_shot: 5`       | Larger few-shot, no iterative refinement                 |
| Self-critique           | disable verifier and auditor      | Only realism feedback, no attribute or diversity control |
| Diversity-only          | disable verifier and discriminator | Coverage-guided refinement only                          |
| Realism-only            | disable verifier and auditor      | Discriminator-guided refinement only                     |
| Full AttrForge          | default                           | All three critics, prompt updater on                     |

Every baseline runs through the same harness and writes the same artifacts, so
results are directly comparable.

---

## Repository layout

```
attrforge/
├── schema.py            typed data models (Pydantic)
├── llm.py               backend-agnostic LLM client (OpenAI, Anthropic, echo)
├── planner.py           attribute planner (stratified, coverage-gap)
├── generator.py         per-target synthetic sample generation
├── critics/
│   ├── verifier.py      per-sample attribute audit
│   ├── discriminator.py mixed-batch real-vs-synthetic judge
│   └── auditor.py       batch-level coverage and near-duplicate audit
├── updater.py           prompt rewriter and versioned history
├── loop.py              orchestrator, persistence, run manifests
├── metrics.py           per-iteration scalar metrics
├── prompts/templates.py canonical prompt strings for every component
└── cli.py               `attrforge run | inspect | schema`
examples/
└── customer_support/    schema, 15 real seed examples, two configs
tests/                   schema, planner, end-to-end offline loop
```

---

## Design notes

* **Backends are pluggable.** OpenAI and Anthropic ship in-tree. An offline
  `echo` backend lets the full pipeline run without any API key, which is what
  the test suite and the dry-run config use.
* **Critics never see the planner's intent directly.** They evaluate the
  *resulting samples*, then the planner uses their output indirectly through
  the updated prompt. This avoids the verifier becoming a noisy oracle.
* **The discriminator measures progress, not the loss.** Realism feedback is
  surfaced to the updater as named artifacts (`"too polished, follows a
  predictable structure"`), not as a scalar to minimize, which empirically
  reduces mode chasing.
* **Prompt history is first-class.** Every rewrite is appended to
  `PromptHistory` with the feedback bundle that motivated it. Comparing
  prompts side-by-side across iterations is the primary way to debug a run.
* **Determinism where it matters.** The planner and generator are seeded; the
  critics use temperature 0.

---

## Roadmap

- [ ] Pareto frontier across (fidelity, realism, diversity) instead of joint optimization
- [ ] Human-in-the-loop adjudication for verifier and discriminator
- [ ] Embedding-based diversity floor (sentence-transformers integration is wired but disabled by default)
- [ ] Multi-language generation
- [ ] Per-attribute calibration with held-out judges
- [ ] Downstream classifier harness (RQ4 protocol)

---

## Citation

If you use AttrForge in academic work, please cite:

```bibtex
@misc{attrforge2026,
  title  = {AttrForge: Multi-Objective Prompt Debugging for Realistic,
            Diverse, and Attribute-Controlled Synthetic Data Generation},
  author = {AttrForge Contributors},
  year   = {2026},
  note   = {Research preview, \url{https://github.com/yourusername/attrforge}}
}
```

A full project description, including the formal problem definition, attribute
schema examples, evaluation phases, and the taxonomy of synthetic-data
artifacts, is available in [`attrforge_project_description.md`](attrforge_project_description.md).

---

## License

MIT. See [LICENSE](LICENSE).
