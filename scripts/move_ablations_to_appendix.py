"""Execute Option A:

- Collapse §7.3 body (lines 420-487 currently) to a 1-paragraph result summary.
- Move Tables 4 and 5 + supporting prose to new Appendix H (Tables H1, H2).
- Renumber body Table 6 -> Table 4 (cross-task headline is now Table 4 of the body).
- Fix abstract / §1 / §11 "6 to 30" -> "10 to 30" range bug (the abstract names
  only SST-2 / Banking77 / TREC; their per-class real-train is 30, 30, 10).
- Add Figure 2 reference to the abstract's pointer.
- Update §1 (C2) and §11 Conclusion to cite Appendix H instead of Table 5.
"""
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "docs" / "index.html"
text = HTML.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# 1. CAPTURE §7.3 block (from <h3 sec-ensemble> up to next <h3 sec-banking77>)
# ----------------------------------------------------------------------
SEC73_START = '<h3 id="sec-ensemble">'
SEC74_START = '<h3 id="sec-banking77">'
i_73 = text.index(SEC73_START)
i_74 = text.index(SEC74_START)
old_73_block = text[i_73:i_74]

# Sanity: contains Tables 4 and 5 captions
assert "Table 4:" in old_73_block, "missing Table 4 caption in §7.3"
assert "Table 5:" in old_73_block, "missing Table 5 caption in §7.3"

# ----------------------------------------------------------------------
# 2. NEW §7.3 SUMMARY (1 paragraph, points to Appendix H)
# ----------------------------------------------------------------------
new_73_block = """<h3 id="sec-ensemble"><span class="subsection-num">7.3</span>Cross-condition ensembling and per-component attribution</h3>

<p>
  Two ablation findings carry the per-component attribution on customer-support intent classification at $N = 10$ seeds. <strong>Cross-condition classifier ensembling.</strong> Logit-averaging the downstream classifiers from two iterated conditions reaches macro F1 $0.947 \\pm 0.056$, with $1.65\\times$ lower seed variance than any solo condition and a $+0.233$ paired lift on worst-class F1 (BCa $95\\%$ CI $[+0.067, +0.500]$, excludes zero). <strong>Component leave-one-out ablation.</strong> Dropping each GAN-style adversary in turn from the seven-critic loop attributes the downstream lift to iteration under the structured-feedback contract, the Pack Discriminator, and the Mode Hunter; Mode-Seeking and Coverage Hole Finder do not measurably differentiate conditions in our experiments. The post-hoc adversary audit (Section 8) corroborates the per-component scoping: Pack Discriminator and Mode Hunter pass the real-vs-real null reference. Full ensemble pair table, statistical comparisons, leave-one-out attribution, and the component-ablation table are in <a href="#sec-appendix-ablations">Appendix H</a>.
</p>

"""

# ----------------------------------------------------------------------
# 3. NEW APPENDIX H from §7.3's old content
# Rename: Table 4 -> Table H1, Table 5 -> Table H2.
# ----------------------------------------------------------------------
# Drop the section header line and replace with appendix header.
# The content after "<h3 id=...>...</h3>" starts at the first <p>.

# Start the appendix with a custom intro paragraph + the existing prose.
# We need to extract from old_73_block:
#   - intro paragraph (line 422-424)
#   - method paragraph (426-428)
#   - Table 4 div
#   - statistical-comparisons paragraph
#   - leave-one-out attribution paragraph
#   - why-ensembling-helps paragraph
#   - component-loo intro paragraph
#   - Table 5 div
#   - closing discussion paragraph
# Drop the <h3>...</h3> heading line, keep everything else.

appendix_h_body = re.sub(
    r'^<h3[^>]*>.*?</h3>\s*\n',
    "",
    old_73_block,
    count=1,
    flags=re.DOTALL,
)

# Rename Tables 4 and 5 -> Tables H1, H2 within the appendix block.
# Order matters: rename Table 5 first (so "Table 5" doesn't briefly look
# like a prefix of "Table 51" or similar -- safe here since 5 is highest).
appendix_h_body = appendix_h_body.replace("Table 5:", "Table H2:")
appendix_h_body = appendix_h_body.replace("Table 4:", "Table H1:")
# Cross-references to Table 4 / Table 5 inside the appendix prose:
appendix_h_body = appendix_h_body.replace("Table 4, macro F1", "Table H1, macro F1")
appendix_h_body = appendix_h_body.replace("(Table 5)", "(Table H2)")
# The closing paragraph (line 486 currently) says "the headline downstream
# result (Table 4, macro F1 0.947 ...)". After rename: "(Table H1, ...)".
# Also the LOO caption mentions "Section 8" which still resolves; leave alone.

appendix_h_wrapped = f"""<!-- ============================================================ -->
<h2 id="sec-appendix-ablations"><span class="section-num">H</span>Appendix: Ablation studies (customer-support)</h2>

<p>
  This appendix carries the two ablation studies that Section 7.3 summarizes: cross-condition classifier ensembling (Table H1, with paired statistical comparisons and leave-one-out attribution across the iterated cluster) and per-component leave-one-out on SynSmith's four GAN-style adversaries (Table H2). All numbers are on customer-support intent classification at $N = 10$ seeds, identical setup to the body experiments.
</p>

<h3><span class="subsection-num">H.1</span>Cross-condition classifier ensembling</h3>

{appendix_h_body.split('<h3', 1)[0].rstrip()}

<h3><span class="subsection-num">H.2</span>Component leave-one-out ablation</h3>

"""

# The "<h3" split above stops at the first <h3 inside the old §7.3 content,
# which is... actually old §7.3 doesn't have inner <h3>. Let's not rely on
# that split. Just use the whole body and split on "Component leave-one-out"
# heading paragraph (the <strong>Component leave-one-out ablation.</strong>
# inline header) to keep H.1 / H.2 sections distinct.

# Redo more carefully: the old §7.3 has these blocks in order:
# - intro paragraph
# - method paragraph
# - Table 4 (now H1)
# - Statistical comparisons paragraph
# - Leave-one-out attribution paragraph
# - Why ensembling helps paragraph
# - Component leave-one-out ablation intro paragraph (<strong>Component...</strong>)
# - Table 5 (now H2)
# - Closing discussion paragraph

# Split point: the "<strong>Component leave-one-out ablation.</strong>"
# inline header marks the start of H.2.
SPLIT_MARK = "<strong>Component leave-one-out ablation.</strong>"
if SPLIT_MARK not in appendix_h_body:
    raise SystemExit("split marker not found in §7.3 content")
h1_part, h2_part = appendix_h_body.split(SPLIT_MARK, 1)

# Trim trailing empty paragraph wrappers from h1_part
h1_part = h1_part.rstrip()
# h2_part begins with the rest of the LOO intro paragraph (after the <strong>...</strong>);
# wrap the LOO intro back into a <p>.
h2_part = "<p>\n  <strong>Component leave-one-out ablation.</strong>" + h2_part
h2_part = h2_part.lstrip()

appendix_h = f"""<!-- ============================================================ -->
<h2 id="sec-appendix-ablations"><span class="section-num">H</span>Appendix: Ablation studies (customer-support)</h2>

<p>
  This appendix carries the two ablation studies that Section 7.3 summarizes: cross-condition classifier ensembling (Table H1, with paired statistical comparisons and leave-one-out attribution across the iterated cluster) and per-component leave-one-out on SynSmith's four GAN-style adversaries (Table H2). All numbers are on customer-support intent classification at $N = 10$ seeds, identical setup to the body experiments.
</p>

<h3><span class="subsection-num">H.1</span>Cross-condition classifier ensembling</h3>

{h1_part}

<h3><span class="subsection-num">H.2</span>Component leave-one-out ablation</h3>

{h2_part}
"""

# ----------------------------------------------------------------------
# 4. REPLACE §7.3 BLOCK in text
# ----------------------------------------------------------------------
text = text[:i_73] + new_73_block + text[i_74:]

# ----------------------------------------------------------------------
# 5. INSERT APPENDIX H before footer
# ----------------------------------------------------------------------
FOOTER_ANCHOR = '<footer class="paper-footer">'
i_footer = text.index(FOOTER_ANCHOR)
# Walk back to start of comment marker that precedes footer
COMMENT_LINE = "<!-- ============================================================ -->\n"
i_pre_footer = text.rfind(COMMENT_LINE, 0, i_footer)
# Insert appendix at i_pre_footer (so order is ... Appendix G ... Appendix H ... footer)
text = text[:i_pre_footer] + appendix_h + "\n" + text[i_pre_footer:]

# ----------------------------------------------------------------------
# 6. RENUMBER body Table 6 -> Table 4
# Body cross-refs: abstract, caption, §7.5 prose, §11 Conclusion.
# Use direct string replace; "Table 6" appears 4-5 times in body only.
# (Appendix H now has its own Tables H1, H2 -- no conflict.)
# ----------------------------------------------------------------------
text = text.replace("Table 6", "Table 4")

# ----------------------------------------------------------------------
# 7. ABSTRACT bug fix: "6 to 30 real-train" -> "10 to 30 real-train"
#    and add Figure 2 reference.
# Same fix for §1 (C1) and §11 Conclusion (those name only SST-2/Banking77/TREC).
# ----------------------------------------------------------------------
text = text.replace("$6$ to $30$ real-train", "$10$ to $30$ real-train")
text = text.replace(
    "the $95\\%$ confidence interval over five seeds (Table 4).",
    "the $95\\%$ confidence interval over five seeds (Figure 2, Table 4).",
)

# ----------------------------------------------------------------------
# 8. UPDATE §1 (C2) line: (Section 7.3, Table 5) -> (Section 7.3, Appendix H)
#    Same for §11 Conclusion.
# Note: after step 6 above, there are NO "Table 5" mentions in the body
# (Table 5 has been moved to Appendix H as Table H2). Any remaining
# "Table 5" reference is stale; update to "Appendix H".
# ----------------------------------------------------------------------
text = text.replace("(Section 7.3, Table 5)", "(Section 7.3, Appendix H)")
# Defensive: any other stale "Table 5" in body prose?
# Audit: print remaining "Table 5" hits (should be zero in body).

# Write
HTML.write_text(text, encoding="utf-8")

# ----------------------------------------------------------------------
# 9. Sanity audit
# ----------------------------------------------------------------------
import subprocess
out = subprocess.check_output(
    ["grep", "-cE", "Table 4|Table 5|Table 6|Table H1|Table H2", str(HTML)],
    encoding="utf-8",
).strip()
print(f"Total Table 4/5/6/H1/H2 lines: {out}")

# Per-label counts
for label in ["Table 4", "Table 5", "Table 6", "Table H1", "Table H2", "Appendix H"]:
    cnt = text.count(label)
    print(f"  {label}: {cnt}")

# Verify no stale "6 to 30" remains
if "6 to 30" in text:
    print("WARNING: stale '6 to 30' still in file")
else:
    print("OK: '6 to 30' fully replaced")

# Verify Appendix H exists with both subsections
assert '<h2 id="sec-appendix-ablations">' in text
assert 'H.1</span>Cross-condition' in text
assert 'H.2</span>Component leave-one-out' in text
print("OK: Appendix H present with H.1 + H.2 subsections")

# Verify body §7.3 is the new short summary
i_73_new = text.index('<h3 id="sec-ensemble">')
i_74_new = text.index('<h3 id="sec-banking77">')
body_73 = text[i_73_new:i_74_new]
assert "Table H1" not in body_73, "appendix table accidentally still in body"
assert "Table H2" not in body_73, "appendix table accidentally still in body"
print(f"OK: §7.3 body is {len(body_73)} chars (was {len(old_73_block)})")
