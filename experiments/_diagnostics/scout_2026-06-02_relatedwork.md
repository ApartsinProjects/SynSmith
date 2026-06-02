# Scout: 2024-2026 related-work positioning for PromptForge

Date: 2026-06-02
Source: web-researcher subagent

## Category A. Direct competitors

**A1. PACE / Synthline** (Bencheikh et al., arXiv:2506.21138, EMNLP 2025). Multi-sample prompting + actor-critic prompt-editing. 4 Requirements-Engineering tasks; F1 gains 6 to 43.8pp; synthetic beats human by up to 15.4pp. Names the "diversity-as-metric vs utility-as-goal" hazard.
**Position:** closest concurrent peer. PromptForge's 7-critic decomposition mitigates by separating attribute/realism/diversity.

**A2. Genetic Prompt: Attributes as Textual Genes** (Liu et al., arXiv:2509.02040, EMNLP 2025 Findings). Treats class-conditional attributes as gene sequences; LLM-driven crossover + active-learning parent selection. Benchmarks against AttrPrompt + Curated LLM, claims SOTA.
**Position:** strongest 2025 attribute-conditioned baseline. PromptForge inverts search (critic-removal vs genetic-recombination); contrast 7-critic structured feedback against blind crossover.

**A3. SynAlign** (arXiv:2502.08661). 3-stage: GP-uncertainty diverse demos → LLM latent-attribute summarization → MMD-weighted post-hoc resampling.
**Position:** alternative formulation of "match the real distribution"; PromptForge's Coverage Hole Finder addresses the same goal inside the loop, not after.

**A4. CoT-Self-Instruct** (Yu et al., arXiv:2507.23751, Meta FAIR). CoT seed reasoning + synthesis with Answer-Consistency / Rejecting-Instruction-Preferences filters. Beats s1k, OpenMathReasoning, Self-Instruct on MATH500, AIME24, GPQA-Diamond, AlpacaEval 2.0, Arena-Hard.
**Position:** PromptForge's per-iteration critic ensemble is roughly the classification-domain analogue of RIP. High-profile reasoning-domain parallel for "critic-as-filter".

## Category B. Structural-precedent work

**B1. PerFine: Iterative Critique-Refine for LLM Personalization** (arXiv:2510.24469). Profile-conditioned critic emits structured complaints along 4 dimensions (tone, vocabulary, structure, topicality) + cross-iteration knockout. Yelp/Goodreads/Amazon: +7.8 to +13.4pp G-Eval.
**Position:** closest analog of "named complaints not scalar rewards". Cite as evidence that structured multi-dimensional critique beats scalar reward in non-data-generation settings.

**B2. Conditional Vendi Score** (Ospanov et al., arXiv:2411.02817). Decomposes Vendi as H(X) = H(X|T) + I(X;T), separating model-intrinsic diversity from prompt-driven diversity.
**Position:** REQUIRED co-citation with Scendi. PromptForge's "102% intrinsic-model-driven" decomposition rests on the same idea this paper formalizes information-theoretically. Cite alongside Scendi as concurrent formalizations.

**B3. Synthetic Eggs in Many Baskets** (arXiv:2511.01490). Multi-source synthetic perplexity 4.42 to 5.11 vs single-source 5.71 to 6.88; Self-BLEU 79.6 synthetic vs 89.2 human.
**Position:** already cited for model-collapse. ALSO useful to motivate ensemble-of-conditions (self_critique + full_attrforge logit-average IS multi-source synthesis at inference).

**B4. Density Ratio Framework for Utility of Synthetic Data** (Volker et al., arXiv:2408.13167). DRE as unified global+local utility measure; open-source R package.
**Position:** PromptForge's Coverage Hole Finder uses density-ratio coverage; this paper transports the same machinery to evaluation. Cite to ground Coverage Hole Finder in an evaluator-side counterpart.

## Category C. Evaluation methodology

**C1. SynQuE** (arXiv:2511.03928). Annotation-free quality estimators (Mean Distance to Medoid, MMD, Proxy-A-Distance, LLM-evaluated Lens); Spearman 0.38 to 0.68 with downstream classifier on sentiment; 32-dataset financial-tweets suite.
**Position:** cheap diagnostic to add alongside Vendi/Scendi; consider Mean-Distance-to-Medoid as a fast no-label complement.

**C2. Surveying Quality, Diversity, Complexity in Synthetic Data** (Havrilla et al., arXiv:2412.02980). QDC framework: Q drives in-distribution generalization, D drives OOD, Complexity helps both.
**Position:** PromptForge's 7 critics map cleanly onto QDC (verifier+realism = Q; diversity-auditor+MSGAN+PacGAN+Mode-Hunter+Coverage = D). Cite as conceptual anchor for "why 7 critics rather than 1".

**C3. Persona Diversity** (Mitra et al., arXiv:2505.17390). Persona prompting raises diversity over no-persona, but fine-grained persona detail "yields minimal gains compared to simply specifying a length cutoff"; synthetic prompts remain "significantly less diverse than human-written ones".
**Position:** defuses the "why didn't you just persona-prompt?" reviewer comment. Cite as external evidence that the popular shortcut is weaker than it looks.

**C4. Judging the Judges + CalibraEval** (Shi et al., arXiv:2406.07791; Li et al., 2025). Position-bias and selection-bias remedies for LLM judges (permutation-balanced calibration, split-and-merge).
**Position:** declare which bias controls PromptForge has in place, or scope as future work.

## Recommended additions to Related Work (priority order)

1. **PACE / Synthline** (arXiv:2506.21138) - direct concurrent peer.
2. **Genetic Prompt** (arXiv:2509.02040) - strongest 2025 attribute-conditioned head-to-head reference.
3. **Conditional Vendi Score** (arXiv:2411.02817) - required co-cite for Scendi decomposition.
4. **PerFine** (arXiv:2510.24469) - structured-vs-scalar critique evidence.
5. **CoT-Self-Instruct** (arXiv:2507.23751) - reasoning-domain critic-as-filter.
6. **SynQuE** (arXiv:2511.03928) - no-label evaluator.
7. **QDC Survey** (arXiv:2412.02980) - Q x D x C conceptual anchor.

## Gaps

- Genetic Prompt's specific quantitative gains vs AttrPrompt not surfaced on arXiv landing; pull from EMNLP Findings PDF or trust-nlp/Genetic-Prompt repo.
- DCScore (arXiv:2502.08512) abstract accessible but vs distinct-n/Vendi comparison not extractable.
