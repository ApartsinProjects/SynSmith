# F1-F6 smoke evidence report

Run dir: `E:\Projects\PromptForge\experiments\_diagnostics\smoke_fixes_f1_f6\runs\20260603T054407Z`
Wall-clock: 5.9 min

## iter 0

### F2 - Pack Discriminator shared_patterns (should EXCLUDE target-distribution patterns)
- pack_accuracy: 0.75
- 25 patterns reported:
  - `'long, complex sentences with multiple clauses'` (in 3 pairs)
  - `'formal tone with detailed explanations'` (in 1 pairs)
  - `"use of 'i am' or 'i recently' as sentence openers"` (in 1 pairs)
  - `"repetitive structure in describing issues (e.g., 'i am experiencing an unusual...')"` (in 1 pairs)
  - `"excessive use of filler phrases like 'i was just wondering if you could help me'"` (in 1 pairs)
  - `'emotional expressions that are overly detailed or verbose'` (in 1 pairs)
  - `'short, abrupt sentences with minimal context'` (in 1 pairs)
  - `'direct demands without polite phrasing'` (in 1 pairs)

### F1 - Realism real-anchor IDs this iter (should be SAME across iters)
- 10 unique real anchor IDs: ['R001', 'R002', 'R005', 'R007', 'R011', 'R014', 'R017', 'R019', 'R028', 'R029']

### F1 - Discriminator reasons (sample of 3)
  - casual tone; real samples are more formal and structured.
  - casual phrasing; real samples exhibit a more formal request style.
  - informal structure; real samples are more polished and detailed.

### F6 - Coverage Hole exemplars (should be stratified across classes)
- classifier_auroc: 1.0
- 4 exemplars, label distribution: {'account_issue': 1, 'complaint': 1, 'general_question': 1, 'refund_request': 1}
  - [account_issue] I changed my email and now I can't receive verification codes on the new one
  - [complaint] i paid for the upgrade but features are still locked
  - [general_question] is the pro plan billed monthly or yearly?
  - [refund_request] the item arrived damaged. need a refund or replacement asap. photos attached.

## iter 1

### F2 - Pack Discriminator shared_patterns (should EXCLUDE target-distribution patterns)
- pack_accuracy: 0.75
- 20 patterns reported:
  - `'long, complex sentences with multiple clauses'` (in 6 pairs)
  - `'formal tone with complete sentences and proper punctuation'` (in 2 pairs)
  - `'short, direct sentences with minimal elaboration'` (in 2 pairs)
  - `'detailed explanations of issues with specific requests for resolution'` (in 1 pairs)
  - `'expressions of frustration or emotional appeal'` (in 1 pairs)
  - `"use of informal greetings like 'hi' or 'hey' at the beginning of inquiries"` (in 1 pairs)
  - `'frequent use of technical jargon without explanation'` (in 1 pairs)
  - `'questions posed in a straightforward manner without additional context'` (in 1 pairs)

### F1 - Realism real-anchor IDs this iter (should be SAME across iters)
- 9 unique real anchor IDs: ['R001', 'R002', 'R007', 'R013', 'R014', 'R017', 'R019', 'R028', 'R029']

### F1 - Discriminator reasons (sample of 3)
  - casual tone; real samples are more formal and structured.
  - informal language; real samples are more formal and detailed.
  - informal and casual language; real samples are more formal.

### F6 - Coverage Hole exemplars (should be stratified across classes)
- classifier_auroc: 1.0
- 4 exemplars, label distribution: {'account_issue': 1, 'complaint': 1, 'general_question': 1, 'refund_request': 1}
  - [account_issue] two factor codes never come through, I've been locked out for 2 days now
  - [complaint] I want my money back. this product does not do what your ads claim.
  - [general_question] Hi team, quick one - does the API have a rate limit on the search endpoint?
  - [refund_request] got the wrong size, want my money back not a store credit thanks

## iter 2

### F2 - Pack Discriminator shared_patterns (should EXCLUDE target-distribution patterns)
- pack_accuracy: 0.5
- 25 patterns reported:
  - `'use of lowercase letters at the beginning of sentences'` (in 2 pairs)
  - `'formal tone with complete sentences and polite requests'` (in 1 pairs)
  - `"use of 'i would like to' or 'i would like to formally request' as openers"` (in 1 pairs)
  - `'lengthy explanations or descriptions of issues'` (in 1 pairs)
  - `'structured format with clear separation of issues and requests'` (in 1 pairs)
  - `"use of informal language with minimal capitalization (e.g., 'i' instead of 'i')"` (in 1 pairs)
  - `"expressions of frustration without detailed context (e.g., 'really frustrating')"` (in 1 pairs)
  - `"questions posed in a casual manner (e.g., 'is there a way to access...')"` (in 1 pairs)

### F1 - Realism real-anchor IDs this iter (should be SAME across iters)
- 8 unique real anchor IDs: ['R001', 'R007', 'R011', 'R013', 'R017', 'R026', 'R028', 'R029']

### F1 - Discriminator reasons (sample of 3)
  - Casual tone; real samples are more formal.
  - Informal language; real samples are more structured.
  - Casual phrasing; real samples are more formal.

### F6 - Coverage Hole exemplars (should be stratified across classes)
- classifier_auroc: 1.0
- 4 exemplars, label distribution: {'account_issue': 1, 'complaint': 1, 'general_question': 1, 'refund_request': 1}
  - [account_issue] two factor codes never come through, I've been locked out for 2 days now
  - [complaint] still no answer on ticket #4421 from last monday. anyone there??
  - [general_question] is the pro plan billed monthly or yearly?
  - [refund_request] ordered 2 items, only got 1. want a refund on the missing one, not a reship.

### F4/F5 - rewritten generator prompt at iter 1 (look for 'Preferred phrasings' lead)
```
"""Generate realistic customer support messages that match the requested intent and style. The example should be plausible for a customer support channel (email, chat, in-app form). Keep it concise and structured, avoiding overly casual language. Output a JSON object with the sample id, the text, and the attribute values.

**Preferred phrasings:**
- I changed my email and now I can't receive verification codes on the new one.
- I paid for the upgrade but features are still locked.
- Is the pro plan billed monthly or yearly?
- The item arrived damaged. Need a refund or replacement asap. Photos attached.

**Forbidden phrasings:**
- "I can't believe I have to deal with this again."

Ensure that the generated messages cover a variety of scenarios, including:
- A formal and detailed complaint scenario with specific examples of service failures.
- A technical problem scenario involving a unique or uncommon issue not typically reported.
- A general question that explores a complex topic rather than simple inquiries.

Focus on reducing redundancy in phrasing and sentence structure. Encourage varied sentence constructions and vocabulary to enhance diversity. Aim for a formal tone with detailed explanations, avoiding excessive filler phrases and emotional verbosity. Strive for clarity and specificity in all messages to align with the target distribution's characteristics."""
```

### F4/F5 - rewritten generator prompt at iter 2 (look for 'Preferred phrasings' lead)
```
""""""Generate realistic customer support messages that match the requested intent and style. The example should be plausible for a customer support channel (email, chat, in-app form). Keep it concise and structured, ensuring a formal tone with detailed explanations. Output a JSON object with the sample id, the text, and the attribute values.

**Preferred phrasings:**
- Two factor codes never come through, I've been locked out for 2 days now.
- I want my money back. This product does not do what your ads claim.
- Hi team, quick one - does the API have a rate limit on the search endpoint?
- Got the wrong size, want my money back, not a store credit, thanks.

**Forbidden phrasings:**
- "I can't believe I have to deal with this again."
- "I have reached out multiple times regarding my order issue."

Ensure that the generated messages cover a variety of scenarios, including:
- A formal complaint structure with detailed evidence of service failures.
- Concise technical queries without emotional language, addressing unique issues.
- Emotional appeals that include personal anecdotes to enhance relatability.

Focus on reducing redundancy in phrasing and sentence structure. Encourage varied sentence constructions and vocabulary to enhance diversity. Strive for clarity and specificity in all messages to align with the target distribution's characteristics while avoiding excessive filler phrases and emotional verbosity. Aim for a structured approach that captures the nuances of customer inquiries and complaints.
```

### Sanity downstream macro F1 (iter_2 attribute-match-rate)
- attribute_match_rate: 0.9523809523809523
- combination_coverage: 0.8666666666666667
