# Contributing to AttrForge

Thanks for considering contributing. AttrForge's value to the community
comes from being a pluggable, reusable framework, not just a frozen
research artifact. The single highest-leverage contribution is a new
critic. The rest of this document walks through how to add one end-to-end,
plus the standard housekeeping (issues, PRs, tests, code style).

## Adding a new critic

A critic is anything that consumes a synthetic batch (and optionally the
real seed set) and emits **structured named complaints** plus optional
scalar metrics. The updater renders every critic's complaints into a
single template, so adding a critic does NOT require changing the loop
or the prompt-update logic.

### 1. Pick a failure mode

Critics are best understood by the failure mode they prevent. The
seven that ship with the framework cover:

| Critic | Prevents |
|---|---|
| Verifier | Right attributes, wrong text |
| Realism Discriminator | Detectably synthetic phrasing |
| Diversity Auditor | Attribute-space coverage gaps |
| Pack Discriminator | Batch-level homogeneity (PacGAN analog) |
| Mode-Seeking | Attribute-deaf generation (MSGAN analog) |
| Mode Hunter | Recurring banned phrasings across iterations |
| Coverage Hole Finder | Distribution coverage gaps the discriminator misses |

If your critic prevents a failure mode that is not on this list (e.g.,
factual correctness, persona consistency, length bias), it is a
candidate for inclusion.

### 2. Implement the protocol

Every critic implements the same interface in `attrforge/schema.py`:

```python
from attrforge.schema import Critic, StructuredFeedback, NamedComplaint

class FactualCritic(Critic):
    name = "factual"

    def evaluate(self, batch, real, attrs):
        """Score the batch and return a StructuredFeedback.

        Parameters
        ----------
        batch : list[SyntheticSample]
            The samples produced in this iteration.
        real : list[RealExample]
            The full real seed set (training half).
        attrs : AttributeSchema
            The attribute schema this run is conditioned on.

        Returns
        -------
        StructuredFeedback
            critic: this critic's name
            metrics: dict[str, float] of scalar quantities the loop
                will track in metric_history
            complaints: list[NamedComplaint] each with a tag and a
                one-sentence reason. The updater renders these into the
                next prompt as actionable items.
        """
        # Your scoring logic here.
        score = some_factual_check(batch, real)
        complaints = []
        for s in batch:
            if not is_factual(s, real):
                complaints.append(NamedComplaint(
                    tag=f"unsupported_claim:{s.id}",
                    reason=f"sample {s.id} contains a claim not in real exemplars",
                ))
        return StructuredFeedback(
            critic=self.name,
            metrics={"factual_score": score},
            complaints=complaints,
        )
```

### 3. Wire it into the ablation table

Open `attrforge/baselines.py` and add an `enable_factual` flag to
`AttrForgeConfig` (in `attrforge/loop.py`), then thread it through:

```python
# attrforge/baselines.py

def full_attrforge(cfg):
    out = ...
    out.enable_factual = True       # NEW
    return out

def no_factual(cfg):
    """full_attrforge minus the factual critic, for the leave-one-out ablation."""
    out = full_attrforge(cfg)
    out.label = "no_factual"
    out.enable_factual = False
    return out

BASELINES["no_factual"] = no_factual
```

### 4. Register with the loop

Open `attrforge/loop.py` and instantiate your critic when `cfg.enable_factual`
is true:

```python
if cfg.enable_factual:
    self._critics.append(FactualCritic(...))
```

The loop will call your critic's `evaluate` once per iteration and pass
its `StructuredFeedback` to the updater alongside the others. No prompt
template changes are needed.

### 5. Test it

Add a unit test under `tests/test_critics/test_factual.py`:

```python
from attrforge.critics.factual import FactualCritic
from attrforge.schema import SyntheticSample, RealExample, AttributeSchema

def test_factual_critic_basic():
    critic = FactualCritic()
    batch = [SyntheticSample(text="x", requested_attributes={"intent": "A"}, ...)]
    real = [RealExample(text="x", label="A")]
    attrs = AttributeSchema(...)
    fb = critic.evaluate(batch, real, attrs)
    assert fb.critic == "factual"
    assert "factual_score" in fb.metrics
```

Run `pytest tests/test_critics/test_factual.py -v` and confirm it passes
without an OpenAI key (use stubbed inputs).

### 6. Add a paragraph to the paper

If you upstream the critic for inclusion in the main framework, the
contribution should be documented in the paper's `§5` (GAN-style
adversaries) or a new section if the critic comes from a different
literature. State: the failure mode it prevents, the structured output
it emits, the analog from prior work (if any).

## Other useful contributions

- **Examples on new datasets.** Add `examples/<dataset>/` with a
  config.yaml + schema.yaml + real_examples.jsonl, matching the layout
  of `examples/customer_support/`. The runner picks them up
  automatically.
- **Backend adapters.** `attrforge/llm.py` currently ships `openai`,
  `anthropic`, and `echo`. Adding `gemini` or `mistral` is a single-file
  change. Match the existing `LLMClient` protocol.
- **Downstream evaluators.** `attrforge/eval/downstream.py` is TF-IDF +
  LogReg. A sentence-transformer-backed evaluator is in
  `attrforge/eval/ensemble.py`; an end-to-end fine-tuning evaluator
  would be a useful addition.
- **Aggregation scripts.** Add scripts to `scripts/` for new analyses
  (calibration ECE/Brier, OOD-detection AUROC, robustness to test-time
  perturbation). The convention is one script per analysis, with a
  `--base` argument that picks up `experiments/<base>_seed*/`.

## Code style

- Python 3.10+ syntax. Type hints on every public function.
- Pydantic v2 for typed data models. Avoid stringly-typed dicts in the
  public API.
- Tests should not require network access or an API key; use the `echo`
  backend or stubbed inputs.
- Keep the loop deterministic-on-seed: pass `seed` through every random
  source (planner, generator, classifier).

## PR checklist

- [ ] `pytest tests/` passes locally.
- [ ] New code has unit tests under `tests/`.
- [ ] If the change adds a critic / baseline / backend, the README's
      "Adding your own critic" section either covers it or is updated.
- [ ] Public API additions are exported from the appropriate
      `__init__.py` and listed in `__all__`.
- [ ] No `print()` statements in library code; use `logging` instead.
- [ ] No new dependencies in `pyproject.toml` unless absolutely needed;
      pin to a major-version-stable range.
- [ ] Em-dashes (U+2014) and double-hyphens (`--`, outside numeric page
      ranges) are not introduced in prose, docstrings, or generated
      paper text.

## Reporting issues

Open a GitHub issue with:

- A reproducible minimal example (an `examples/<dataset>/config.yaml`
  or a `tests/test_<bug>.py` that fails on the current main).
- The output of `pip show attrforge` and `python --version`.
- The expected vs actual behavior.

## License

MIT. By contributing, you agree your changes are licensed under MIT.
