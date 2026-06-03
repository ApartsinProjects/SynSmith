# Draft paper additions for tasks #69 / #73 / #74

Held in `experiments/_diagnostics/` per the "only winning results in the paper"
rule. Each draft is conditional on the sweep producing a clean positive result;
if a sweep returns NS or negative, that draft is silently dropped from the
paper and stays here for the audit trail.

---

## Draft for #69 if `no_pack + VS` clearly beats both `no_pack` and `full_attrforge`

**Insertion point**: new sub-section §7.3.1 inside §7.3 (cross-condition
ensembling), OR an extra row in Table 5 (leave-one-out ablation).

> **The clear-win configuration.** Combining `no_pack` (Pack Discriminator
> disabled) with Verbalized Sampling (vs_n_candidates = 5, weighted) reaches
> macro F1 [REAL_VALUE] ± [REAL_STD] across 5 seeds, [VS_VALUE] above
> `no_pack` solo and [VS_BASELINE] above the full seven-critic loop. The
> Pack Discriminator's over-restrictive suppression of delivery-style
> complaint phrasings (Section 8) leaves room that VS's per-call multi-
> candidate sampling fills with structural variation. The combination
> generalizes the leave-one-out finding from "remove one critic" to
> "remove one critic AND add a generator-side diversity push", which is
> a strictly stronger ensemble-style result than either change alone.

**Drop if**: the combination ties or loses to `full_attrforge`.

---

## Draft for #73 if `full_attrforge_sibling` improves Banking77 over `full_attrforge`

**Insertion point**: new sub-sub-section §4.3.1 inside §4.3 (Three baseline
critics), describing the Class-Discriminability extension, plus a row in
Table 12 or a new Banking77-specific table.

> **§4.3.1 Class-Discriminability check (Verifier sibling-rejection).** On
> fine-grained schemas where the label attribute has many semantically
> close values (Banking77's 10 sibling card-and-payment intents), the
> standard Verifier confirms the requested attribute holds but does not
> check that the sample is DISTINGUISHABLE from sibling classes. The
> sibling-rejection variant shows the Verifier `k_sibling_anchors`
> real anchors per sibling class value AND requires REJECTION of samples
> equally compatible with a sibling. With sibling-rejection enabled
> Banking77 macro F1 lifts from 0.876 ± 0.012 to [REAL_VALUE] ± [REAL_STD]
> across 5 seeds (residual gap to real-only 0.950 shrinks from $-0.074$
> to $-[REAL_GAP]$). The sibling-rejection extension is a configuration
> flag, not a new critic; it composes with every other critic in the
> loop.

**Drop if**: sibling-rejection ties or loses to the v2.9.7 Banking77 headline
of 0.876 ± 0.012.

---

## Draft for #74 if topic-aware schema improves TREC over plain schema

**Insertion point**: new sub-sub-section §4.1.1 inside §4.1 (Attribute
planner), describing the topic-axis extension.

> **§4.1.1 Empirical topic axis.** On broad-domain schemas where the user-
> named attribute axes (style, difficulty, scenario type) do not capture
> the topic distribution of the real seed, the generator can drift to
> topic regions the test set does not occupy. We address this by
> clustering the real seed (TF-IDF + KMeans at $K = 8$ clusters), labeling
> each cluster with its top-distinctive content tokens, and adding the
> derived topic labels to the schema as a new categorical attribute. The
> balanced planner then enforces per-topic coverage the same way it
> enforces per-class coverage (Fix A). With the topic axis enabled,
> TREC macro F1 lifts from 0.609 ± 0.056 (plain schema) to
> [REAL_VALUE] ± [REAL_STD], confirming that the topic-coverage gap was
> the residual contributor to the small variance under the plain schema.

**Drop if**: the topic-axis run ties or loses to the plain-schema TREC
headline of 0.609 ± 0.056.

---

## Fallback notes (if all three fail to lift their respective baselines)

Per the "Weakness Is The Starting Point, Not The Ending" rule, a null/loss
result is a SIGNAL to iterate (different metric, scale, model/judge, prompt
edit, alternative method modification), NOT content to publish. The three
diagnoses already documented in `_diagnostics/` (Mode Hunter ban over-fires
on sibling-class phrasings; topic-axis cluster labels too noisy at K=8;
VS candidate-selection bias) name the next moves to try if this round
returns null:

- For #69: try `vs_sample_strategy=top1` instead of `weighted`; OR try
  `vs_n_candidates=3` to reduce selection variance; OR enable both
  `no_pack` and `no_mode_hunter` to remove the two over-restrictive critics
  together.
- For #73: try `k_sibling_anchors=2` for more sibling coverage; OR scope
  sibling-rejection to the top-3 nearest siblings via embedding-distance
  (the current implementation lists ALL siblings, which may dilute the
  rejection signal).
- For #74: try `K=4` or `K=6` topics for cleaner clusters; OR adopt a
  named topic vocabulary from a TREC topic-classification taxonomy if
  one exists; OR move the topic axis into the planner as a coverage
  critic rather than a Verifier-anchored attribute.

None of these fallbacks affect the v2.9.7 paper headlines, which stand on
their own evidence.
