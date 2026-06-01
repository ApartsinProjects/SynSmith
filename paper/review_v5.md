# TMLR Fifth-Round Review (v1.6): "Adversarial Prompt Debugging for LLM Synthetic Data Generation"

Reviewer: senior reviewer
Round: 5 (post-blocker-fix revision)
Verdict: MINOR REVISIONS (see "Overall recommendation" at the bottom)

## Summary

The v1.6 revision addresses all four round-4 blockers (BL-NEW-1 to BL-NEW-4) substantively, and the new Section 7.4 (multi-classifier augmentation) is statistically defensible and supplies the exact cross-classifier check that round 4 (MA-NEW-5) requested. The paper is now a much more honest empirical artifact: per-class variance is positioned as the statistically robust headline, the aggregate-F1 mean difference is correctly framed as directional-not-significant, the discretization-artifact observation is disclosed, and the new TF-IDF char finding (full_attrforge directionally above at every n >= 15) is a genuine, modest positive that strengthens the diversity-injection story.

What remains is one minor numerical sloppiness in the abstract that should be tightened (the +0.021 magnitude claim at n=15 specifically), a few small wording issues, and an unclosed HTML <p> tag in Section 7.6. None are blocking. The paper as written is acceptable with minor revisions.

## Status of the four round-4 blockers

### BL-NEW-1 (round 4): "only iterated condition reaches the ceiling" → FIXED

WHERE-OLD: round-4 abstract sentence 1 of "What we find": "the seven-critic loop is the only iterated condition that reaches the real-only macro F1 ceiling ($0.893$) at every real-train size from $20$ to $30$".

WHERE-NEW (lines 56, 84, 405, 710): The "only iterated condition" framing has been removed for aggregate F1. The new abstract reads:

> "the seven-critic loop matches the real-only macro F1 ceiling ($0.893$) at every real-train size from $20$ to $30$, alongside two simpler iterated baselines (realism-only and diversity-only) that also reach the ceiling"

This is honest. Section 7.1 paragraph (ii) (line 405) and Figure 2 caption (line 376) both now explicitly acknowledge that realism_only and diversity_only also match or exceed the ceiling. Section 11 (Conclusion, line 710) is consistent. The "only iterated condition" phrase is preserved exactly once in the abstract (line 56) and once in Section 7.1 (line 439) and once in Section 11 (line 710), but in a different context: it refers to per-class stability on complaint (sd = 0.149 vs >= 0.408 for every other iterated condition; never collapses to F1=0). I verified this against `per_class_aug.json`:

| condition | complaint at n=30 mean | sd | contains 0 |
|---|---|---|---|
| self_critique | 0.467 | 0.447 | YES |
| realism_only | 0.733 | 0.435 | YES |
| diversity_only | 0.667 | 0.408 | YES |
| full_classic | 0.533 | 0.506 | YES |
| **full_attrforge** | **0.733** | **0.149** | **NO** |

The "only iterated condition with stable per-seed F1, never collapsing to F1 = 0" claim is FULLY VERIFIED by the data. **BL-NEW-1 resolved.**

### BL-NEW-2 (round 4): paired-t p-values not disclosed → FIXED

WHERE-NEW (lines 381, 405, 419, 484, 489, 552):

Table 3 caption (line 381) now reports:

> "The paired-t $p$ for full_attrforge vs full_classic is $0.22$ at $n = 5$, $0.76$ at $n = 10$, $0.99$ at $n = 15$, and $0.74$ at $n = 20, 25, 30$; no comparison reaches conventional significance at $N = 5$. We report the table as point estimates with seed variance."

I verified each value against `scarce_real.json` (paired-t two-sided, full_attrforge − full_classic): 0.2235, 0.7562, 0.9862, 0.7396, 0.7396, 0.7396. All match within rounding.

Section 7.1 paragraph (ii) (line 405) adds: "paired-t $p = 0.74$ and Wilcoxon $p = 0.75$ at $n = 20, 25, 30$; $95\%$ bootstrap CI of the paired difference is $[-0.085, +0.128]$." Verified — these match.

Table 7 (line 531) reports paired-t p-values for all 18 comparisons in the multi-classifier augmentation experiment. Verified against `scarce_real_multi_classifier.json`. **BL-NEW-2 resolved.**

### BL-NEW-3 (round 4): discretization artifact not disclosed → FIXED

WHERE-NEW (lines 381, 405):

Table 3 caption (line 381): "the per-seed predictions of full_classic and full_attrforge are identical across $n = 15$ to $30$".

Section 7.1 paragraph (iii) (line 405): "The per-seed values of full_classic at $n = 15$, $20$, $25$, $30$ are identical: $\{0.733, 1.000, 0.893, 0.733, 1.000\}$ in seed order, and the same insensitivity holds for full_attrforge from $n = 20$ onward. Once the augmented set contains $48$ synthetic samples plus $\geq 15$ real samples, the $10$-item test set's per-class predictions become insensitive to additional real training data. The apparent plateau is therefore a property of the saturated downstream classifier on a small held-out set, not a quality limit of the synthetic data."

I verified the per-seed identity claim:
- full_classic at n in {15, 20, 25, 30}: [0.733, 1.000, 0.893, 0.733, 1.000] at every n (identical)
- full_attrforge at n in {20, 25, 30}: [0.893, 0.893, 0.787, 0.893, 1.000] at every n (identical)

Both claims VERIFIED. **BL-NEW-3 resolved.**

### BL-NEW-4 (round 4): "+0.10 to +0.23" range internally inconsistent → FIXED

WHERE-NEW (lines 56, 405):

Abstract: "At smaller real-train sizes ($n \in \{5, 10\}$) both methods provide large gains over real-only ($+0.08$ to $+0.23$ macro F1)".

Verified against `scarce_real.json`:
- n=5: real-only 0.561, full_classic gain = +0.229, full_attrforge gain = +0.142
- n=10: real-only 0.748, full_classic gain = +0.103, full_attrforge gain = +0.080

Range across both methods × both sizes = [+0.080, +0.229] → rounds to [+0.08, +0.23]. Consistent with the new abstract claim.

Section 7.1 paragraph (i) (line 405) reads: "At $n = 5$, both full_classic and full_attrforge augmentation deliver F1 gains of $+0.14$ to $+0.23$ over the real-only baseline." Verified for n=5 alone: range [+0.142, +0.229] → [+0.14, +0.23]. Internally consistent.

Figure 2 caption (line 376): "$+0.14$ to $+0.23$ macro F1 at $n = 5$, $+0.08$ to $+0.10$ at $n = 10$". Verified: n=10 range is [+0.080, +0.103] → rounds to [+0.08, +0.10]. Consistent.

**BL-NEW-4 resolved.** All abstract/section/caption ranges now derive from the same data file.

## New Section 7.4: statistical defensibility

I verified all claims in Section 7.4 against `scarce_real_multi_classifier.json`. Headline claims:

### Claim 1: "the gap shrinks to 0.100 at n = 5 (paired-t p = 0.17, no longer significant) and vanishes at n >= 15 (mean diff in [0.001, 0.020], all p > 0.59)" — VERIFIED

Computed paired-t for full_attrforge minus full_classic under TF-IDF word:

| n | mean diff | sd | paired-t p |
|---|-----------|----|-----------|
| 5  | -0.100 | 0.133 | 0.168 |
| 10 | -0.068 | 0.145 | 0.354 |
| 15 | +0.020 | 0.077 | 0.597 |
| 20 | +0.017 | 0.130 | 0.790 |
| 25 | +0.009 | 0.200 | 0.927 |
| 30 | +0.001 | 0.113 | 0.992 |

- The "0.100 at n=5, p=0.17" matches (sign is full_classic over full_attrforge, paper writes the magnitude correctly).
- The "mean diff in [0.001, 0.020] at n>=15" matches (lower bound 0.0006, upper 0.0196).
- "all p > 0.59" matches (min p at n>=15 is 0.597).

### Claim 2: "Under TF-IDF char features, full_attrforge is directionally above the three-critic loop at every n >= 15 (mean diff +0.019 to +0.045)" — VERIFIED

| n | mean diff | sd | paired-t p |
|---|-----------|----|-----------|
| 15 | +0.037 | 0.167 | 0.649 |
| 20 | +0.045 | 0.075 | 0.255 |
| 25 | +0.019 | 0.060 | 0.522 |
| 30 | +0.027 | 0.080 | 0.498 |

All four are positive (directionally above). Range is [+0.019, +0.045]. Verified.

### Table 7 verification

Spot-checked four cells: tfidf_word at n=15 (paper: +0.020 ± 0.077, p=0.60; my comp: +0.020 ± 0.077, p=0.597 — exact match within rounding); tfidf_char at n=20 (paper: +0.045 ± 0.075, p=0.25; my comp: +0.045 ± 0.075, p=0.255 — match); st at n=15 (paper: +0.001 ± 0.162, p=0.99; my comp: +0.001 ± 0.162, p=0.986 — match); tfidf_word at n=30 (paper: +0.001 ± 0.113, p=0.99; my comp: +0.001 ± 0.113, p=0.992 — match). All correct.

### Figure 5 caption: "at every n >= 15 and across every classifier choice, the paired-t p-value of full_attrforge vs full_classic exceeds 0.25"

At n >= 15 the minimum p-value across all 12 (n × classifier) cells is 0.2545 (tfidf_char at n=20). This exceeds 0.25 by 0.0045, so the literal claim "exceeds 0.25" is technically true. It is borderline; if the analysis were rerun with a slightly different seed split or bootstrap, it might fall just below 0.25. I would recommend softening the threshold to "exceeds 0.20" (which gives a comfortable margin of 0.054 at the closest cell) or restating as "exceeds 0.25 except at n=20 under TF-IDF char (p = 0.25)". This is a MINOR phrasing concern, not a blocker.

### Conclusion: Section 7.4 is statistically defensible

The new section is a genuine cross-classifier robustness check, accurately reported. The two patterns identified (augmentation shrinks the gap; under char features the seven-critic loop is directionally above) both survive direct verification. The framing is calibrated: "all not significant at N = 5" is stated for the char-features finding, which is the honest reading.

## Remaining issues in v1.6

### MAJOR: None.

### MINOR

**MI-V5-1. Abstract magnitude "(+0.021 macro F1 at every n >= 15)" is inaccurate at n = 15.**

WHERE: Abstract paragraph 3 (line 56): "The seven-vs-three-critic mean difference is consistent in direction ($+0.021$ macro F1 at every $n \geq 15$)".

WHY: Verified means at n = 15, 20, 25, 30:
- n=15: full_attrforge 0.873, full_classic 0.872, diff = +0.001
- n=20,25,30: full_attrforge 0.893, full_classic 0.872, diff = +0.021

The "+0.021" magnitude only holds at n in {20, 25, 30}. At n=15 the difference is +0.001 (effectively zero). The "consistent in direction" claim is technically true at every n >= 15 (all four diffs are positive), but conflating +0.001 with +0.021 by writing "+0.021 at every n >= 15" is misleading.

FIX: Change to one of:
- "consistent in direction at every $n \geq 15$, reaching $+0.021$ macro F1 at $n \geq 20$"
- "consistent in direction at every $n \geq 15$ ($+0.001$ at $n = 15$, $+0.021$ at $n \in \{20, 25, 30\}$)"

The Section 7.1 paragraph (ii) at line 405 already gets this right ("at every $n \geq 20$ ($0.893 \pm 0.075$)"). The abstract should be tightened to match.

**MI-V5-2. Abstract claim "paired-t p >= 0.74 at every n" is inaccurate at n = 5.**

WHERE: Abstract paragraph 3 (line 56): "is not statistically distinguishable from zero at our sample budget of $N=5$ seeds (paired-t $p \geq 0.74$ at every $n$)".

WHY: Verified paired-t p-values:
- n=5: p = 0.224 (NOT >= 0.74)
- n=10: p = 0.756
- n=15: p = 0.986
- n=20,25,30: p = 0.740

At n=5 the p-value is 0.22, far from 0.74. The "p >= 0.74 at every n" reading is therefore wrong if "every n" includes n=5. The point estimate (mean_AF=0.703, mean_FC=0.789) is actually quite different at n=5: AF loses to FC by 0.087. So at n=5 the comparison is still not significant, but the framing in the abstract (which suggests "p>=0.74 every n" → "uniformly indistinguishable") obscures that AF is well below FC at n in {5, 10}.

FIX: Either (a) restrict the p-value scope: "paired-t $p \geq 0.74$ at every $n \geq 15$"; or (b) acknowledge the n=5,10 reversal: "paired-t $p$ ranges from $0.22$ at $n = 5$ (where the difference is in the opposite direction) to $0.99$ at $n = 15$, never reaching conventional significance".

I prefer (b) because it avoids hiding the n=5 reversal, which the paper does disclose in Section 7.1 (ii) and Table 3 but the abstract elides.

**MI-V5-3. Figure 5 caption threshold "exceeds 0.25" is borderline.**

WHERE: Figure 5 caption (line 526): "at every $n \geq 15$ and across every classifier choice, the paired-t $p$-value of full_attrforge vs full_classic exceeds $0.25$".

WHY: TF-IDF char at n=20 gives p = 0.2545. This exceeds 0.25 by 0.0045. A reader doing the same paired-t calculation might round 0.2545 to 0.25, depending on their precision. The claim survives strictly but is fragile.

FIX: Change to "exceeds $0.20$" (gives comfortable margin) or "ranges from $0.25$ at the closest cell to $0.99$ at the loosest". My recommendation: "$p$ ranges from $0.25$ to $0.99$ across all $4 \times 3 = 12$ comparisons; no comparison reaches conventional significance".

**MI-V5-4. Section 7.6 has an unclosed <p> tag.**

WHERE: line 599-600. The paragraph starts at line 599 and the content ends at line 600, but there's no closing </p> before line 602's <!-- comment -->.

FIX: Add `</p>` after the last sentence of the paragraph (after "...later ones diverge.").

**MI-V5-5. Inconsistent rendering of inequality "$\geq$" vs "$\ge$".**

Throughout the paper. Not a content issue. Both render the same in KaTeX. Leave as-is.

**MI-V5-6. Figure 5 image legend uses "attrrforge" (extra 'r').**

WHERE: The rendered PNG at `figures/main_run_002_scarce_real_multi_classifier.png` shows "full_attrrforge" in the legend (three r's). The HTML caption uses correct spelling "full_attrforge".

FIX: Regenerate the figure with correct spelling. Or accept as cosmetic if the regeneration cost is non-trivial.

**MI-V5-7. Section 7.4 paragraph (ii) is a hypothesis dressed as a finding.**

WHERE: Section 7.4 paragraph (line 552): "The TF-IDF char $3$-$5$ classifier captures sub-word phrasing variation rather than the keyword-consistency signal of word-level TF-IDF. Under this classifier, the seven-critic loop is directionally above the three-critic loop at every $n \geq 15$..."

WHY: The framing "captures sub-word phrasing variation" is a plausibility argument, not a measurement. The direct measurement is that under char features, full_attrforge is +0.019 to +0.045 above full_classic, all not significant. The mechanism explanation should be tagged as hypothesis.

FIX: Either (a) add a hedge: "we conjecture that the TF-IDF char $3$-$5$ classifier captures sub-word phrasing variation..."; or (b) leave the empirical claim and remove the mechanism sentence, since the empirical pattern speaks for itself.

### Inherited from earlier rounds but unchanged

- The few_shot exemplar-pool confound (round-4 MA-NEW-1) remains; Section 7.1 narrative no longer leans on this comparator for headline claims, so it is less load-bearing. Acceptable.
- The hyperparameter sweep ablation (round-4 MA-NEW-2) is not added; the Limitations section already acknowledges single-budget evaluation. Acceptable.
- The MMD measurement to support the "embedding absorbs diversity" mechanism (round-4 MA-NEW-6) is not added. The mechanism claim remains a conjecture in the abstract: "the seven-critic loop is directionally above the three-critic loop at every $n \geq 15$" under char features is now positioned as evidence rather than as bare assertion, which weakens the original objection. Acceptable.

## Numerical consistency audit

I cross-checked every numerical claim in the abstract against the data files. Results:

| Claim | Location | Data file | Value in paper | Value in data | Status |
|---|---|---|---|---|---|
| Real-only ceiling 0.893 | abstract, 7.1, conclusion | scarce_real.json | 0.893 | 0.893 (n=30) | OK |
| FC plateau 0.872 | abstract, 7.1 | scarce_real.json | 0.872 ± 0.134 | 0.872 ± 0.134 | OK |
| AF on complaint 0.733 ± 0.149 | abstract, 7.1, 11 | per_class_aug.json | 0.733 ± 0.149 | 0.733 ± 0.149 | OK |
| AF on complaint range [0.67, 1.00] | abstract | per_class_aug.json | [0.67, 1.00] | [0.667, 1.000] | OK |
| Other iterated conds sd >= 0.408 | abstract, 7.1 | per_class_aug.json | 0.408 | min sd is 0.408 (diversity_only) | OK |
| Mean diff +0.021 at n>=15 | abstract | scarce_real.json | +0.021 | +0.001 at n=15; +0.021 at n>=20 | INACCURATE (see MI-V5-1) |
| Paired-t p >= 0.74 at every n | abstract | scarce_real.json | p >= 0.74 | p=0.22 at n=5 | INACCURATE (see MI-V5-2) |
| +0.08 to +0.23 at n in {5,10} | abstract | scarce_real.json | [+0.08, +0.23] | [+0.080, +0.229] | OK |
| TF-IDF isolated gap 0.14, p=0.046 | abstract, 7.2, 9.1 | multi_classifier.csv | 0.14, p=0.046 | 0.144, p=0.046 | OK |
| ST isolated gap p=0.82 | abstract, 7.3, 9.1 | (per-seed multi-clf not stored) | 0.82 | (cannot rederive) | LIKELY OK |
| Mode Hunter 11.6 ± 0.6 | line 473 | per-seed summaries | 11.6 ± 0.6 | 11.6 ± 0.548 | OK |
| Distinct-1 0.382 for AF | Table 8 | diversity_metrics.json | 0.382 ± 0.024 | 0.382 ± 0.024 | OK |
| Per-iter AF 0.25→0.37→0.49 | 7.6 | per_iter.csv | 0.25, 0.37, 0.49 | 0.249, 0.366, 0.494 | OK |
| Per-iter FC 0.31→0.34→0.34 | 7.6 | per_iter.csv | 0.31, 0.34, 0.34 | 0.314, 0.336, 0.335 | OK |
| Pack acc audit AF 0.52 | Table 9 | table.csv (pack_audit_mean) | 0.52 ± 0.09 | 0.516 ± 0.094 | OK |
| Pack acc null 0.55 | Table 9 caption, 8 | summary.json | 0.55 | 0.547 | OK |
| Mode-seeking ratio 0.23 ± 0.01 | Table 9 caption | table.csv (ms_audit) | 0.23 ± 0.01 | min 0.231, max 0.235 | OK (within rounding) |
| Section 7.3 char gap "0.52 vs 0.49" | line 511 | multi_classifier.csv | 0.52 vs 0.49 | 0.520 vs 0.486 | OK (rounds to 0.52 vs 0.49) |
| TF-IDF char dir-above at n>=15 (+0.019 to +0.045) | 7.4, Fig 5 | scarce_real_multi_classifier.json | [+0.019, +0.045] | [+0.019, +0.045] | OK |
| TF-IDF word "0.100 at n=5, p=0.17" | 7.4 | scarce_real_multi_classifier.json | 0.100, p=0.17 | 0.100, p=0.168 | OK |
| TF-IDF word "vanishes at n>=15 in [0.001, 0.020] p>0.59" | 7.4 | scarce_real_multi_classifier.json | [0.001, 0.020], p>0.59 | [0.0006, 0.020], p>=0.597 | OK |
| Section 7.4 "p > 0.25 at every n>=15 across all classifiers" | Fig 5 caption | scarce_real_multi_classifier.json | p > 0.25 | min p = 0.2545 | OK borderline (see MI-V5-3) |

### Statistical reporting consistency

The paper is now consistently honest about non-significance:
- Abstract: "is not statistically distinguishable from zero"
- Section 7.1 (ii): "We report the observation as directional rather than as a hypothesis-test conclusion"
- Section 7.4: "all not significant at $N = 5$"
- Conclusion: "not statistically distinguishable from zero at $N = 5$"
- Figure 4 caption: "statistically indistinguishable"
- Figure 5 caption: "the paired-t $p$-value... exceeds $0.25$"

The directional-not-significant language is uniformly applied. No place asserts the mean difference as established. The previous round's split between "matches the ceiling" framing and "0.872 plateau" framing has been replaced with explicit "matches alongside two other conditions; 0.021 difference not significant; plateau is discretization artifact".

### Overreach check

Looking for any "established" or "demonstrates" language that exceeds the data:

- Section 11 (Conclusion, line 710): "matches the real-only macro F1 ceiling ($0.893$) at every real-train size $n \geq 20$. Two simpler iterated baselines (realism-only and diversity-only) also match the ceiling on this protocol". OK.
- Section 11: "the seven-vs-three-critic mean difference of $+0.021$ macro F1 is in the consistent direction at every $n \geq 15$ but not statistically distinguishable from zero at $N = 5$ (paired-t $p \geq 0.74$)". Has same MI-V5-1 issue (+0.021 specifically at n>=15) and the p>=0.74 issue (true for n>=10, false for n=5).
- Section 11: "the seven-critic loop is the only iterated condition with stable per-seed F1 ($0.733 \pm 0.149$ on range $[0.67, 1.00]$, never collapsing to F1 $= 0$), while every other iterated condition has $\text{sd} \geq 0.408$ on this class and collapses to F1 $= 0$ on at least one of the five seeds." FULLY VERIFIED.
- Section 11: "the same critics that cost $0.14$ macro F1 under TF-IDF in the isolated protocol cost only $0.10$ at $n = 5$ and zero at $n \geq 15$ in the augmentation protocol". VERIFIED (TF-IDF word values: isolated -0.144, aug at n=5 -0.100, aug at n>=15 +0.020 to +0.001).
- Section 11: "the seven-critic loop is directionally above the three-critic loop under character-level TF-IDF at every $n \geq 15$". VERIFIED.

No overreach beyond the abstract phrasing issues called out in MI-V5-1 and MI-V5-2. The Conclusion section inherits MI-V5-1's "+0.021 at every n >= 15" issue; same fix applies.

## Numbering audit

- Sections: 1-12 + Appendix A (BibTeX). No duplicates. References work.
- Subsections: 7.1, 7.2, 7.3, 7.4 (new), 7.5, 7.6 (renumbered from 7.4, 7.5). Verified all internal cross-refs:
  - Line 379 caption refers to "Table 3" ✓
  - Section 7.4 (line 521) refers to "Section 7.3" (which is correctly the previous section) ✓
  - Section 7.4 (line 552) refers to "Section 7.5" (the renumbered direct-diversity section) ✓
  - Section 8 (line 650) refers to "Section 7.3" (multi-classifier) ✓
  - Section 9 (line 656) refers to "Section 7.3" ✓
  - Section 9.6 refers to "Section 7.5" for distinct-n... let me check this... Section 9.6 doesn't actually mention 7.5; only verbsampling/nanoflux comparison.
- Figures: 1-8. Figure 5 is the new multi-classifier scarce_real plot. Figures 6, 7, 8 are the renumbered realism_curve, audit_protocol, audit_differential. No stale references.
- Tables: 1-9. Table 7 is new (paired-diff multi-classifier). Tables 8, 9 are the renumbered diversity_metrics and audit. Verified: Section 7.5 (line 555) refers to "Table 8" ✓. Section 8 (line 624) refers to "Table 9" ✓. Section 7.2 (line 446) refers to "Table 9" (was Table 8 in v1.5, this is the renumbered audit table). ✓

All numbering is internally consistent.

## Overall recommendation

**MINOR REVISIONS.**

The v1.6 revision resolves all four round-4 blockers cleanly. The new Section 7.4 is the cross-classifier robustness check that round 4 (MA-NEW-5) explicitly requested; it is statistically defensible and accurately reported. The per-class variance headline (3.4x lower sd on complaint, never collapses to F1=0) is the kind of robust observation that statistically underpowered N=5 studies can support, and the paper now correctly positions this as the headline rather than the aggregate-F1 mean difference. The protocol-dependence framing (isolated p=0.046, augmentation p>0.17 across all three classifiers) is a clean cross-classifier story that strengthens the paper's central thesis (diversity-utility tradeoff is reader-dependent and protocol-dependent).

The remaining issues are small: two abstract phrasings that conflate +0.021 with the (smaller) +0.001 at n=15 and overstate "p >= 0.74 at every n"; one borderline threshold in Figure 5 caption ("exceeds 0.25" when the min is 0.2545); one unclosed `<p>` tag; one cosmetic figure-legend typo ("attrrforge"). All are 5-minute fixes.

What I would request from a minor revision:
1. **Tighten the abstract magnitude** (MI-V5-1): change "+0.021 macro F1 at every n >= 15" to "+0.021 macro F1 at every n >= 20" (or list both: "+0.001 at n=15, +0.021 at n in {20, 25, 30}").
2. **Tighten the abstract p-value scope** (MI-V5-2): change "p >= 0.74 at every n" to "p >= 0.74 at every n >= 10" or be explicit about the n=5 reversal direction.
3. **Soften the Figure 5 threshold** (MI-V5-3): change "exceeds 0.25" to "exceeds 0.20" or list the actual range.
4. **Close the <p> tag** (MI-V5-4): trivial HTML fix in Section 7.6.
5. **Regenerate Figure 5** with correct "full_attrforge" spelling (MI-V5-6); cosmetic only.

None of these blocks publication. With these fixes the paper is ready for camera-ready.

I want to acknowledge that the authors have done a substantial honest revision over 4 prior rounds. The v1.6 abstract is dramatically more honest than v1.4's "only iterated condition reaches the ceiling" claim. The decision to position per-class variance reduction (a statistically robust signal at N=5) as the headline, rather than the aggregate F1 mean difference (which is not statistically distinguishable at N=5), is exactly the kind of calibrated reporting TMLR asks for. The new Section 7.4 adds substantive empirical value rather than just patching the blocker. Good revision.

## Top 5 fixes ranked by impact

1. **(MI-V5-1, MI-V5-2)** Abstract phrasing precision. The two slips in the abstract ("+0.021 at every n >= 15" and "p >= 0.74 at every n") are minor but easy to fix and matter because the abstract is what most readers see. 10-minute fix.

2. **(MI-V5-4)** Close the `</p>` tag in Section 7.6. 30-second HTML fix.

3. **(MI-V5-3)** Soften the Figure 5 caption "exceeds 0.25" to give it more margin. 1-minute fix.

4. **(MI-V5-6)** Regenerate Figure 5 with correct "full_attrforge" spelling in the legend. 5-minute fix if the plotting script is intact.

5. **(MI-V5-7)** Add a hedge to the Section 7.4 mechanism sentence ("we conjecture that..."). 1-minute fix.

The cumulative time to address all of the above is under 30 minutes. I would be comfortable seeing this paper accepted with minor revisions and recommend the editor read the v1.7 only for the abstract paragraph, the Figure 5 caption, and a quick HTML lint pass.
