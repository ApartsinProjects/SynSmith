<div align="center">

<img src="assets/hero.png" alt="AttrForge: a forge anvil with a glowing crystalline prompt being shaped, surrounded by three critic spirits (cyan attribute verifier, amber diversity auditor, magenta realism discriminator) sending structured feedback back into the prompt." width="100%" />

<br />

# AttrForge

**Multi-Objective Prompt Debugging for Realistic, Diverse, and Attribute-Controlled Synthetic Data Generation**

[![Paper](https://img.shields.io/badge/paper-HTML-blue.svg)](https://apartsinprojects.github.io/PromptForge/)
[![DOCX](https://img.shields.io/badge/paper-DOCX-blue.svg)](https://apartsinprojects.github.io/PromptForge/attrforge.docx)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: research preview](https://img.shields.io/badge/status-research%20preview-orange.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

*An open-source framework that reframes synthetic data generation as iterative,*
*critic-guided prompt optimization.*

**Paper:** [Adversarial Prompt Debugging for LLM Synthetic Data Generation](https://apartsinprojects.github.io/PromptForge/) (Apartsin & Aperstein) - HTML with KaTeX math, or [download .docx](https://apartsinprojects.github.io/PromptForge/attrforge.docx)

[Paper](https://apartsinprojects.github.io/PromptForge/) · [Overview](#overview) · [Method](#method) · [Quickstart](#quickstart) · [Architecture](#architecture) · [Research questions](#research-questions) · [Citation](#citation)

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
attribute vectors. Seven LLM critics, three baseline (attribute verifier,
realism discriminator, diversity auditor) and four GAN-style adversaries
(Pack Discriminator, Mode-Seeking, Mode Hunter with persistent memory,
Coverage Hole Finder via density-ratio estimation), score the batch along
independent axes. A prompt updater consumes their structured feedback and
rewrites the generator prompt for the next round.

The result is a GAN-style process in which the optimized variable is the
prompt rather than the weights, with four simultaneous objectives:

> **attribute fidelity · realism · diversity · batch-level coverage**

The framework ships two example datasets, customer-support intent
classification (5 classes, 40 real examples) and Banking77 cards-and-payments
(10 classes, 300 real-train, 400 held-out test). The released artifacts
include all per-seed runs, raw critic outputs, aggregation scripts, and the
cross-condition classifier ensemble harness that reaches macro F1
$0.947 \pm 0.056$ on customer-support at $N = 10$ seeds.

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

| Critic                  | Question it answers                                | Failure mode it prevents                                     | Class            |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ---------------- |
| Attribute Verifier      | Does the text reflect the requested vector?        | Metadata-only labels: the right attribute string with mismatched text | baseline |
| Realism Discriminator   | Can a judge separate synthetic from real?          | Over-polished, template-y, telltale LLM phrasing             | baseline         |
| Diversity Auditor       | Does the batch cover the attribute space?          | Mode collapse, shallow paraphrases, missing rare/edge cases  | baseline         |
| Pack Discriminator      | Can a judge separate k-sample packs of real vs synthetic? | Batch-level homogeneity invisible to per-sample realism judges | PacGAN analog    |
| Mode-Seeking            | Does attribute variation produce surface variation? | Attribute-deaf generation: same text for different attribute vectors | MSGAN analog     |
| Mode Hunter             | Which LLM tics appear in synth but not real?       | Recurring banned phrasings, opener tics, structural templates | ban-list training |
| Coverage Hole Finder    | Which real examples does the synthetic batch fail to cover? | Distributional coverage holes the discriminator alone misses | density-ratio coverage |

Removing any one critic produces a measurably degraded distribution along its
axis; the seven-critic loop is referenced in the paper as `full_attrforge` and
is the default configuration when `attrforge run` is invoked without an
ablation flag.

---

## Quickstart

### Install

```bash
git clone https://github.com/ApartsinProjects/PromptForge.git
cd PromptForge
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
attrforge run examples/customer_support/config.yaml --iterations 3
attrforge inspect runs/<run_id>
```

### Run the seven-condition paper experiment (customer-support or Banking77)

```bash
# Customer-support, 10 seeds, 7 conditions, 3 iterations, 16 samples/iter
python scripts/run_experiments.py \
  --config examples/customer_support/config.yaml \
  --conditions naive few_shot self_critique realism_only diversity_only full_classic full_attrforge \
  --seeds 17 23 41 53 89 101 109 127 137 149 \
  --iterations 3 --samples-per-iteration 16 \
  --run-id main_run_002

# Banking77 cards-and-payments, 5 seeds
python scripts/run_experiments.py \
  --config examples/banking77/config.yaml \
  --conditions naive few_shot self_critique realism_only diversity_only full_classic full_attrforge \
  --seeds 17 23 41 53 89 \
  --iterations 3 --samples-per-iteration 16 \
  --run-id banking77_run_001

# Cross-condition classifier ensembling (the headline analysis)
python scripts/ensemble_deep.py --base main_run_002
```

### Programmatic API

```python
from attrforge import AttrForge

forge = AttrForge.from_config("examples/customer_support/config.yaml")
result = forge.run(iterations=3)

print(result.final_prompt)
print(result.metric_history[-1])
# {'attribute_match_rate': 0.92,
#  'discriminator_accuracy': 0.58,
#  'pack_accuracy': 0.53,
#  'mode_seeking_ratio': 0.18,
#  'hunter_library_size': 11,
#  'coverage_auroc': 0.99,
#  'near_duplicate_rate': 0.04,
#  'combination_coverage': 0.83, ...}
```

### Adding your own critic

Every critic implements the same protocol (`name`, `evaluate(batch, real, attrs) -> StructuredFeedback`). To
add a fifth GAN-style adversary or a domain-specific verifier:

```python
# attrforge/critics/my_critic.py
from attrforge.schema import Critic, StructuredFeedback, NamedComplaint

class MyCritic(Critic):
    name = "my_critic"
    def evaluate(self, batch, real, attrs):
        return StructuredFeedback(
            critic=self.name,
            metrics={"my_score": 0.42},
            complaints=[NamedComplaint(tag="opener-tic", reason="every sample opens 'Hi team'")],
        )
```

Then wire it into `attrforge/baselines.py` and add a flag in the ablation table.
The updater template will render its complaints alongside the existing critics
automatically; no change to the loop or the prompt-update logic is required.

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

| Baseline                | `--conditions` flag       | Description                                              |
| ----------------------- | ------------------------- | -------------------------------------------------------- |
| Naive prompting         | `naive`                   | One manually written prompt, no critic loop              |
| Few-shot prompting      | `few_shot`                | 8-exemplar few-shot, no iterative refinement             |
| Self-critique           | `self_critique`           | Only the deterministic diversity-auditor; no LLM judges  |
| Diversity-only          | `diversity_only`          | Coverage-guided refinement, no realism / verifier critics |
| Realism-only            | `realism_only`            | Realism discriminator + auditor; no verifier / GAN adversaries |
| Full classic (3-critic) | `full_classic`            | Verifier + realism + auditor (3 baseline critics)         |
| Full AttrForge (7-critic) | `full_attrforge`        | All 7 critics: 3 baseline + 4 GAN-style adversaries (default) |

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
│   ├── auditor.py       batch-level coverage and near-duplicate audit
│   ├── pack.py          Pack Discriminator (PacGAN analog)
│   ├── mode_seeking.py  attribute-distance / text-distance ratio (MSGAN)
│   ├── mode_hunter.py   persistent banned-phrasings library
│   └── coverage_hole.py density-ratio-based coverage finder
├── updater.py           prompt rewriter and versioned history
├── baselines.py         ablation builders for every named baseline
├── loop.py              orchestrator, persistence, run manifests
├── metrics.py           per-iteration scalar metrics
├── prompts/templates.py canonical prompt strings for every component
├── eval/downstream.py   sentence-transformer + LR downstream evaluator
└── cli.py               attrforge run | inspect | schema
examples/
├── customer_support/    5-class intent, 40 real seeds (30 train + 10 test)
└── banking77/           10-class card/payment subset, 300 train + 400 test
scripts/
├── run_experiments.py        per-condition runs across seeds
├── ensemble_deep.py          cross-condition logit-average ensemble
├── augmentation_eval.py      real + synthetic downstream eval
├── per_class_aug_eval.py     per-class F1 augmentation analysis
├── scarce_real_eval.py       n_real sweep for augmentation
├── reaudit_fixed.py          Vendi + MS-emb + 5-fold AUROC re-audit
├── diversity_metrics.py      distinct-n + self-BLEU-4
├── mmd_per_feature_space.py  MMD with TF-IDF word/char + sentence-transformer
└── worst_class_eval.py       worst-class F1 sweep
tests/                        schema, planner, end-to-end offline loop
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

- [x] Downstream classifier harness (sentence-transformer + LR; multi-classifier ablation)
- [x] Embedding-based diversity floor (sentence-transformer Gram-matrix Vendi score)
- [x] Banking77 cross-domain replication (10-class card/payment subset)
- [x] Cross-condition classifier ensembling
- [x] Post-hoc adversary audit with real-vs-real null reference
- [ ] Pareto frontier across (fidelity, realism, diversity) instead of joint optimization
- [ ] Human-in-the-loop adjudication for verifier and discriminator
- [ ] Multi-vendor judge ensembling (claude-haiku + gpt-4o-mini + gemini-flash)
- [ ] Verbalized Sampling + retrieval-augmented persona generator (see scout report)
- [ ] Calibrated 3-judge debate Realism Critic with KS-stopping
- [ ] Scendi-score diversity decomposition
- [ ] Per-rewrite causal attribution via prompt-diff

---

## Citation

If you use AttrForge in academic work, please cite:

```bibtex
@misc{apartsin2026attrforge,
  title  = {Adversarial Prompt Debugging for LLM Synthetic Data Generation},
  author = {Apartsin, Alexander and Aperstein, Yehudit},
  year   = {2026},
  url    = {https://github.com/ApartsinProjects/PromptForge},
  note   = {Holon Institute of Technology and Afeka College of Engineering, Israel.
            Paper: \url{https://apartsinprojects.github.io/PromptForge/}.}
}
```

A full project description, including the formal problem definition, attribute
schema examples, evaluation phases, and the taxonomy of synthetic-data
artifacts, is available in [`attrforge_project_description.md`](attrforge_project_description.md).

---

## License

MIT. See [LICENSE](LICENSE).
