# Task #73: Class-Discriminability sibling-rejection on Banking77 — null result

**Status**: completed, dropped from paper (per "only winning results" rule).
**Date**: 2026-06-04.
**Codebase**: v2.10.1 + F1-F7 fix bundle.

## Hypothesis

Adding a sibling-class anchor block to the Verifier (so each verification call
sees one real example from each of the K-1 sibling classes the sample is NOT
labelled as) helps fine-grained classifiers distinguish near-classes on
Banking77's 10-class card-and-payment subset. Implemented as
`enable_sibling_rejection=True` + `k_sibling_anchors=1` in `verifier.py`.

## Setup

- Banking77 10-class card/payment subset.
- Real-train 30 per class, test 400 held-out.
- Conditions: `full_attrforge` (SynSmith reference) vs `full_attrforge_sibling`.
- 5 seeds: 17, 23, 41, 53, 89.
- OpenAI Batch API, `gpt-4o-mini`.

## Headline result

| seed | SynSmith | SynSmith+sib | delta |
|---:|---:|---:|---:|
| 17 | 0.705 | 0.673 | -0.032 |
| 23 | 0.750 | 0.712 | -0.038 |
| 41 | 0.721 | 0.661 | -0.060 |
| 53 | 0.647 | 0.694 | +0.046 |
| 89 | 0.662 | 0.681 | +0.019 |

- Mean: 0.697 +/- 0.042 (SynSmith) vs 0.684 +/- 0.020 (SynSmith+sib).
- Paired-t: mean diff -0.013, sd 0.044, t = -0.66 (NS, p ~ 0.55).
- **Variance reduction**: 2x tighter seed variance (0.042 -> 0.020); the only
  positive signal.

## Conclusion

The Class-Discriminability extension does not improve macro F1 on Banking77 at
this scale. Mean delta is small and not statistically significant; variance
reduction is the only positive signal but does not warrant a paper claim.

Per "only winning results in the paper" rule, dropped from the paper. Code
stays in `synsmith/critics/verifier.py` as an experimental feature behind the
`enable_sibling_rejection` flag; users who want to try it on a different
dataset can opt in via config.

## Artifacts

- Raw runs: `experiments/task73_v2_10_1_seed{17,23,41,53,89}/`
- Aggregation script: `scripts/aggregate_task73.py`
- Per-seed summaries: `all_summaries.json` in each seed directory
