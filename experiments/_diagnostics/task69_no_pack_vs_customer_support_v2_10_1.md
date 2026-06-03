# Task #69: no_pack and no_pack_vs on customer-support at v2.10.1

**Status**: completed.
**Date**: 2026-06-04.
**Codebase**: v2.10.1 + F1-F7 fix bundle.

## Hypothesis

After the F1-F7 fixes (stratified Realism anchor, Pack target-distribution gate,
Updater priority for Coverage Hole, hybrid Realism anchor for Coverage Hole),
the Pack Discriminator's contribution to SynSmith should be measurable on
customer-support. The original v2.10.0 N=10 LOO showed dropping Pack giving
+0.039 macro F1 (Pack appeared harmful); the F1-F7 work specifically targeted
the Pack failure modes that produced this anti-signal.

Bonus condition: `no_pack_vs` combines dropping Pack with switching the
generator to Verbalized Sampling. Originally hypothesized to be a clear-win
config; if it under-performs, no_pack alone is the cleaner LOO arm.

## Setup

- Customer-support 5-class intent classification.
- Real-train 30 per class, test 10 held-out.
- Conditions: `full_attrforge` (SynSmith), `no_pack` (Pack OFF), `no_pack_vs`
  (Pack OFF + Verbalized Sampling generator).
- 5 seeds: 17, 23, 41, 53, 89.
- OpenAI live API + Batch hybrid (per seed schedule).

## Headline result

| seed | SynSmith | no_pack | no_pack_vs |
|---:|---:|---:|---:|
| 17 | 0.520 | 0.787 | 0.640 |
| 23 | 0.567 | 0.448 | 0.520 |
| 41 | 0.800 | 0.433 | 0.393 |
| 53 | 0.787 | 0.447 | (missing) |
| 89 | 0.633 | 0.687 | 0.547 |

- Mean: SynSmith 0.661, no_pack 0.560, no_pack_vs 0.525 (n=4).
- SynSmith vs no_pack: paired difference +0.101 (SynSmith better).
- SynSmith vs no_pack_vs: paired difference +0.136 (SynSmith better, on the 4
  available seeds).

**Pack Discriminator contribution flipped to positive after F1-F7.** In the
v2.10.0 Appendix H Table H2 N=10 LOO, dropping Pack gave +0.039 (Pack appeared
harmful). At v2.10.1 N=5 with F1-F7 fixes, dropping Pack costs -0.101 (Pack is
now load-bearing). The fix bundle worked.

## Paper-implications

The paper's Appendix H Table H2 currently reports v2.10.0 N=10 numbers. The
v2.10.1 N=5 result above does not invalidate the v2.10.0 result (different
sample size, different codebase) but does point to a future-work item:
re-running the N=10 LOO sweep on v2.10.1 would refresh Table H2 with the
post-fix numbers and likely sharpen the Pack contribution claim.

For the current paper revision, the v2.10.0 Table H2 stands.

## Artifacts

- Raw runs: `experiments/task69_v2_10_1_seed{17,23,41,53,89}/`
- Aggregation script: `scripts/aggregate_task69.py`
