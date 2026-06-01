# Sanity Audit of Empirical Results

Performed on `experiments/sim_run_002_*` (3 seeds, 7 conditions, sim backend).
Reading the raw per-iteration artifacts (`samples.jsonl`, `mode_hunter.json`,
`metrics.json`, `audit_summary.json`) against the headline claims in `docs/index.html`.

## Summary of findings

| ID  | Severity | Finding                                                                                  | Status            |
| --- | -------- | ---------------------------------------------------------------------------------------- | ----------------- |
| A   | Blocker  | Post-hoc audit pack accuracies are identical across all 3 seeds (std dev = 0.000)        | Bug in audit RNG  |
| B   | Blocker  | For seeds 23 and 41, full_attrforge/full_classic/realism_only/diversity_only produce IDENTICAL samples across every iteration | Bug + artifact    |
| C   | Major    | Mode Hunter is too sample-limited at batch_size=16: detected 0 tics for seed 23          | Bug (low signal)  |
| D   | Major    | Per-iter F1 trajectories are identical across all 5 iterated conditions for seeds 23, 41 | Downstream artifact |
| E   | Minor    | "Pack accuracy 0.75 → 0.50" headline finding is partly RNG, partly real                  | Re-validate needed |

## Detailed findings

### A. Audit RNG state evolves between conditions

The post-hoc audit's `PackDiscriminator` is instantiated once with `seed=99` and reused across
all conditions in a single audit script invocation. Each `attack()` call advances the internal
RNG state. So conditions processed later in the audit get a different RNG state than
conditions processed earlier, even though the audit is supposed to score every condition's
data with identical machinery.

**Proof**: Pack accuracies are identical across 3 different experiment seeds:

```
condition          seed17     seed23     seed41
diversity_only     0.312      0.312      0.312
few_shot           0.625      0.625      0.625
full_attrforge     0.500      0.500      0.500
full_classic       0.438      0.438      0.438
naive              0.750      0.750      0.750
realism_only       0.688      0.688      0.688
self_critique      0.562      0.562      0.562
```

Std dev across seeds = 0.000 for every condition. The cross-condition spread is a function
of processing order, not seed or content.

**Fix**: Re-seed the pack discriminator (and Mode Hunter, and any RNG-bearing component)
before each condition's audit. `scripts/posthoc_audit.py` lines 110-115.

### B. Conditions produce identical samples (simulator determinism)

For seed 23, every iteration of `full_attrforge`, `full_classic`, `realism_only`, and
`diversity_only` produces 16/16 identical samples. Only `self_critique` differs (it has
a different instruction set, missing the "Match the style and surface form" clause that the
auditor's coverage hole finder adds).

**Cause**: The simulator's `_generate_text` function picks a base utterance deterministically
from `(seed, sample_id, label, style, noise, scenario)` and then applies surface mutations.
For seed 23, Mode Hunter found 0 tics, so the "Forbidden phrasings" clause was empty in
`full_attrforge`'s prompt, making it functionally identical to `full_classic`'s prompt from
the simulator generator's perspective.

This means the paper's claim that "AttrForge variants reduce pack accuracy toward chance" is
partly tautological in this run: it's measured on data that's NOT actually different from
`full_classic`'s data.

**Fix options**:
1. Increase batch size from 16 to 32-48 so Mode Hunter has more signal.
2. Lower Mode Hunter `min_repeats` from 2 to 1 (catches every tic occurrence).
3. Concentrate the simulator's tic distribution so the same tic appears multiple times.
4. Reduce the simulator's randomness (e.g. always emit a tic when not forbidden) so prompt
   suppression has measurable effect.

### C. Mode Hunter signal sparsity

`DEFAULT_LLM_TICS` has 6 entries. Simulator emits a tic with 35% probability, randomly chosen.
At batch_size=16, expected occurrences per tic = 16 × 0.35 / 6 ≈ 0.93. Below `min_repeats=2`
detection threshold in expectation.

**Empirical**:
- Seed 17 Mode Hunter: found 2 tics (`'Thanks for reaching out'`, `'Please rest assured'`)
- Seed 23 Mode Hunter: found 0 tics
- Seed 41 Mode Hunter: found 2 tics (`'I appreciate your patience'`, `'I apologize for any inconvenience'`)

In 1/3 runs, Mode Hunter fails to find any tics at all, making the "persistent banned-phrasings
library" claim seed-dependent.

### D. Downstream F1 saturates regardless of condition

For seed 23 and seed 41, all 5 iterated conditions produce identical per-iteration downstream
F1 trajectories. This is downstream from finding B (identical samples → identical F1) and the
TF-IDF classifier's keyword-saturation behavior on a 10-item test set.

**Implication**: The downstream metric in this experiment provides zero information about
which critic stack is in use, beyond the trivial "iterated vs not iterated" distinction. The
paper already acknowledges saturation; the audit confirms it more strongly.

### E. Headline claim re-assessment

The paper currently claims:

> "AttrForge variants reduce pack-discriminator accuracy from 0.75 (naive baseline)
> toward chance (0.50), and that the full AttrForge stack drives the realism discriminator
> closer to chance than any baseline while matching downstream classifier performance."

After this audit, the truthful version is:

- The 0.75 vs 0.50 spread IS deterministic (Pack discriminator score for the same data is
  deterministic when re-seeded), so the architecture-level claim holds.
- BUT the per-condition spread among iterated conditions in the audit (0.31 to 0.69) is
  partly an RNG-order artifact (Bug A).
- The realism discriminator finding (0.69 ± 0.06 for full_attrforge vs 0.72 ± 0.10 for
  full_classic) is within seed-std, NOT statistically meaningful.
- The "matching downstream classifier performance" claim is technically true but trivial:
  the downstream metric does not distinguish any of the iterated conditions in 2/3 seeds.

## Action items

1. Fix Bug A: re-seed audit's pack discriminator per condition.
2. Make Mode Hunter more sensitive: `min_repeats=1`, batch_size up to 32.
3. Run a live-LLM experiment with a real model so the simulator's artifacts no longer drive
   the headline numbers. This is the only path to TMLR-quality empirical claims.
4. Rewrite the paper's empirical claims to match what the data actually supports,
   pending the live-LLM run. The architectural contribution remains valid; the simulator
   results need a more honest framing.

## Code locations

- `scripts/posthoc_audit.py:108-114` — pack discriminator RNG bug
- `attrforge/sim_backend.py:_generate_text` — deterministic-by-target generation
- `attrforge/critics/mode_hunter.py:ModeHunterConfig.min_repeats` — sensitivity knob
- `attrforge/critics/pack_discriminator.py:PackDiscriminator.__init__` — RNG state owner
