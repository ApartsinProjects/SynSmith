# Scout: published baselines for multi-task parity claim

Date: 2026-06-02
Source: web-researcher subagent

## Top-3 pursue-first task families

| Rank | Family | Cost | Published baselines |
|------|--------|------|---------------------|
| **1** | **SST-2 binary sentiment** | Cheapest add (drop-in to harness) | DistilBERT-base 91.3% acc (full real, [arXiv:1910.01108]); SuperGen 92.8% acc (synth-only, NeurIPS 2022, [arXiv:2202.04538]) |
| **2** | **MNLI 3-class NLI** | Small harness change (concat premise+hypothesis) | DistilBERT-base 82.2% matched (full real, [arXiv:1910.01108]); SuperGen 72.3 / 73.8 m/mm (synth-only, [arXiv:2202.04538]) |
| **3** | **SST-5 5-class sentiment** | Same family as SST-2, ordinal | BERT-base 54.9% / DistilBERT ~51% / RoBERTa SOTA 60.2% ([Munikar et al. arXiv:1910.03474]) |

## Skipped for now (require harness rewrite)
- **CoNLL-2003 NER**: per-token labels, not single-text classification. DistilBERT ~89% F1; DistilRoBERTa 90.74% ([HF philschmid model card](https://huggingface.co/philschmid/distilroberta-base-ner-conll2003)).
- **SQuAD QA**: passage-conditioned span generation. DistilBERT-base 79.6 EM / 86.99 F1 ([HF distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad)).

## Framing template (per the scout)

> "Trained only on AttrForge synthetic data (n=K per class), a DistilBERT-base classifier reaches **X%** of the **full-supervised DistilBERT-base reference of 91.3% on SST-2** [DistilBERT, arXiv:1910.01108], and is **within Y points of SuperGen's 92.8% synthetic-only baseline** [SuperGen, NeurIPS 2022, arXiv:2202.04538]."

## Detailed citations

- **SST-2**: HF `glue/sst2`, 67k/872/1821, Apache-2.0. Metric: accuracy on validation.
- **MNLI**: HF `glue/mnli`, 392k/9815/9832, GLUE. Metric: matched + mismatched accuracy.
- **SST-5**: HF `SetFit/sst5`, 8.5k/1.1k/2.2k. Metric: root-level accuracy.
- **CoNLL-2003**: HF `conll2003`, 14k/3.2k/3.5k. Metric: seqeval span-F1.
- **SQuAD v1**: HF `rajpurkar/squad`, 87k/10.5k, CC-BY-SA 4.0. Metric: EM + token F1.

## Key supporting papers

- [DistilBERT (Sanh et al. 2019), arXiv:1910.01108](https://arxiv.org/abs/1910.01108) — supervised baselines on SST-2, MNLI, SQuAD
- [SuperGen (Meng et al. NeurIPS 2022), arXiv:2202.04538](https://arxiv.org/abs/2202.04538) — synth-only baselines on SST-2 (92.8) and MNLI (72.3/73.8); RoBERTa-large + 6000 synthetic examples per class
- [ProGen (Ye et al. EMNLP-Findings 2022), arXiv:2210.12329](https://arxiv.org/abs/2210.12329) — progressive synthetic data; DistilBERT-base; on-par with baselines at 1% the synthetic-data size
- [LM-BFF (Gao et al. ACL 2021), arXiv:2012.15723](https://arxiv.org/abs/2012.15723) — few-shot real (K=16/class) reference
- [Munikar et al. SST-5 BERT, arXiv:1910.03474](https://arxiv.org/abs/1910.03474)
- HF model cards: [DistilBERT-SST2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), [DistilBERT-SQuAD-distilled](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad)

## Scout-flagged gaps (verify before final publication)

- ProGen per-dataset table not fetched; pull from [ACL Anthology PDF](https://aclanthology.org/2022.findings-emnlp.269.pdf) to lock exact numbers.
- SuperGen Table 1 full RTE/QNLI/QQP cells not confirmed; cited only SST-2 + MNLI from abstract.
- LM-BFF K=16 exact per-task numbers not pulled; Table 3 of the LM-BFF paper is the source.
