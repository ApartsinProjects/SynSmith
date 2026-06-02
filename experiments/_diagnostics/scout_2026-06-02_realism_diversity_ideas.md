# Scout: Ideas for improving realism + diversity in the PromptForge / AttrForge loop

Date: 2026-06-02
Source: web-researcher subagent scout across 2024-2026 literature

## Summary

Four directions for improving the seven-critic loop, each with 3-5 concrete citation-backed ideas and a one-line "how it slots in" note. Recommended pursue-first targets called out at the end.

## DIRECTION 1: Better REALISM through prompt-level techniques

1. **Verbalized Sampling (VS)** (arXiv:2510.01171, GitHub CHATS-lab/verbalized-sampling). Training-free, asks the LLM to verbalize a distribution over candidate outputs (e.g., "Generate 5 jokes with their probabilities"), 1.6-2.1x diversity over direct prompting while preserving quality. Diagnoses mode collapse as a typicality bias from RLHF preference data. **Slot:** swap per-class generator from "produce one utterance" to "produce 5 utterances with probabilities; sample one".

2. **SynAlign / Few-shot Distribution Matching** (arXiv:2502.08661). Quantifies the gap in linguistic-attribute proportions between LLM synthetic and real data; Exploration-aware Sampling + Latent Attribute Reasoning + Synthetic Distribution Alignment modules. **Slot:** add a "linguistic-attribute gap" critic comparing real-vs-synthetic distributions of attributes (length, hedges, sentiment).

3. **SynthesizRR (Retrieval-Augmented Realism)** (arXiv:2405.10040). Retrieves real exemplars from a corpus and conditions generation on them, producing more lexically and semantically diverse synthetic data than vanilla few-shot. **Slot:** before each gpt-4o-mini call, retrieve k=2 real Banking77 utterances by class via dense retrieval; condition the generator.

4. **PersonaHub (1B personas)** (arXiv:2406.20094). Web-curated 1B-persona library for distributed perspectives. Caveat: arXiv:2505.17390 shows persona prompting alone has worse diversity than human writing unless combined. **Slot:** sample (persona, scenario) pair per generation; Mode Hunter flags persona-style leakage.

**Variant for PromptForge:** Replace generator prompt with a **Verbalized-Sampling + retrieval-augmented persona** template: (i) retrieve 2 real exemplars; (ii) condition on a sampled persona; (iii) ask for 5 candidates with probabilities; (iv) route candidates to the existing critic stack and keep top-1 by realism score.

## DIRECTION 2: GAN-analogous ideas not yet in our paper

1. **Manifold-Entropy Discriminator** (arXiv:2208.12055). Generalizes the discriminator into a feature embedder maximizing entropy of the embedding distribution. **Slot:** add a "feature-entropy" critic on sentence-transformer embeddings; planner penalizes generations that fail to raise it.

2. **EBGAN-style energy discriminator + margin loss** (Zhao et al. EBGAN, reviewed in 2025 ML survey DOI 10.1007/s10994-025-06772-7). Energy function with low energy near real data and a hinge margin elsewhere. **Slot:** convert Realism Discriminator from binary fake/real to a continuous energy score (0-1 with margin); hinge-margin reward for retained vs discarded candidates.

3. **SENTRA contrastive pre-trained detector** (arXiv:2509.12385). Selected-Next-Token Transformer with contrastive pretraining for cross-domain LLM-text detection; strong OOD generalization. **Slot:** train a small contrastive detector on (real Banking77, synthetic); use its score as a second realism critic uncorrelated with gpt-4o-mini.

4. **Adaptive Multi-Adversarial Training** (arXiv:2112.14406). Multiple discriminators trained adaptively; each specializes in a failure region; generator must satisfy all. **Slot:** add an **adaptive weighting layer** that upweights whichever critic showed the worst-class drop on the previous validation slice.

**Variant for PromptForge:** Replace the boolean Realism Discriminator with an **EBGAN-style continuous energy critic + a contrastive SENTRA-style co-judge**; aggregate as `min(E_gpt4o, E_contrastive)` to enforce worst-case realism.

## DIRECTION 3: Realism with LLM-as-judge improvements

1. **Multi-Agent Debate with Adaptive Stability Stopping** (arXiv:2510.12697). Beta-Binomial mixture model + Kolmogorov-Smirnov statistic adaptively stops debate when judge consensus stabilizes. **Slot:** replace the single Realism Discriminator with a 3-judge debate (claude-haiku + gpt-4o-mini + gemini-flash) using KS-stopping.

2. **CalibraEval (selection-bias calibration, ACL 2025)** (arXiv:2409.16788). Calibrates the prediction distribution to remove position and verbosity biases without retraining. **Slot:** apply content-swap averaging to every pairwise call (realism A vs B), debias logits via CalibraEval.

3. **Confidence-Diversity Calibration** (arXiv:2508.02029). Uses intra-ensemble confidence + disagreement diversity as a calibrated signal; linear combination predicts human-agreement at R^2 = 0.979. **Slot:** treat judge-vs-judge disagreement as an uncertainty score; route high-disagreement items to a separate ambiguous-pool critic.

4. **Linear-probe judge calibration** (arXiv:2512.22245). Linear probe on judge hidden states for calibrated uncertainty without ensembling. **Slot:** if compute is tight, replace 3-judge debate with one judge + linear probe over logits.

**Variant for PromptForge:** Introduce a **Calibrated Debate Realism Critic**: 3 judges with content-swap, KS-adaptive stopping, disagreement-as-uncertainty routing of borderline candidates to a verifier-grounded fallback.

## DIRECTION 4: Diversity through attributed / structured prompt sampling

1. **Scendi Score (Schur-complement prompt-aware diversity)** (arXiv:2412.18645, GitHub aziksh-ospanov/scendi-score). Prompt-conditioned diversity via Schur-complementing the prompt's effect out of a joint kernel; ICCV 2025. **Slot:** replace the current diversity-auditor's plain Vendi metric with Scendi over (prompt, output) pairs.

2. **Vendi-cousins (Hill numbers + similarity)** (arXiv:2310.12952). Family of similarity-based diversity metrics with tunable sensitivity to rare items. **Slot:** Coverage Hole Finder gets a rare-item-sensitive Vendi-cousin (q < 1 order) as its scoring head.

3. **Attributes-as-Textual-Genes (LLM as GA simulator)** (arXiv:2509.02040). Treats attributes as a genome; uses LLM as crossover/mutation operator over attribute vectors. **Slot:** AttrForge's attribute planner becomes a genetic-algorithm sampler; realism + coverage critics serve as fitness.

4. **DoAug (diversity-oriented paraphraser fine-tune)** (arXiv:2502.11671). Fine-tunes an LLM as a diverse paraphraser; +10.5% downstream gain over runner-up baselines. **Slot:** add a post-hoc DoAug paraphrase pass over kept synthetic candidates.

5. **DATG (Dynamic Attribute Graphs)** (ACL 2024, GitHub IAAR-Shanghai/DATG). Graph over attribute co-occurrences guides controlled generation; supports hierarchical attribute structure. **Slot:** attribute planner samples from a learned DATG over Banking77 attribute co-occurrence statistics instead of independent attribute draws.

**Variant for PromptForge:** Replace the current independent attribute sampler with an **Attributes-as-Genes planner + Scendi-scored diversity critic + Vendi-cousin coverage head**.

## Pursue-first ranking (effort vs leverage)

1. **Verbalized Sampling + retrieval-augmented persona generator (D1.1 + D1.3 + D1.4)** — Highest leverage at cheap implementation. Single-prompt fix demonstrated 1.6-2.1x diversity gain on creative-writing in the cited paper. Direct slot into existing `generator.py`.

2. **3-judge debate Realism Critic with KS-stopping (D3.1)** — Provably better than single judge or majority vote. Maps cleanly to the cloud-LLM multi-vendor anti-bias rule in CLAUDE.md.

3. **Scendi score on (prompt, output) pairs (D4.1)** — Adds prompt-vs-intrinsic diversity decomposition. Direct upgrade to the diversity-auditor with an open-source reference impl.

4. **Attributes-as-Genes planner (D4.3)** — Conceptually clean; aligns AttrForge's attribute schema with a known GA-style sampler.

## Gaps (no robust 2025 citation found)

- "Wasserstein LLM-as-judge" — closest analog is the linear-probe / Bayesian Win-Rate work.
- Energy-based discriminators specifically for text in 2025 are sparse; EBGAN translation is conceptual, not a verified text result.
- The DATG GitHub README quality was not separately verified.
