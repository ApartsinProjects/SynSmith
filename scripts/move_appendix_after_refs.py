"""Move Appendix A and Appendix B from before §12 References to after it.

Boundaries are HTML section markers:
- Appendix A start: <h2 id="sec-appendix">
- Appendix B start: <h2 id="sec-appendix-b">
- §12 References start: <h2 id="sec-references">
- Appendix C start: <h2 id="sec-appendix-framework-reference">
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "docs" / "index.html"

text = HTML.read_text(encoding="utf-8")

# Anchors
A_START_ANCHOR = '<h2 id="sec-appendix">'
B_START_ANCHOR = '<h2 id="sec-appendix-b">'
REFS_START_ANCHOR = '<h2 id="sec-references">'
C_START_ANCHOR = '<h2 id="sec-appendix-framework-reference">'

# Locate
i_A = text.index(A_START_ANCHOR)
i_B = text.index(B_START_ANCHOR)
i_R = text.index(REFS_START_ANCHOR)
i_C = text.index(C_START_ANCHOR)

assert i_A < i_B < i_R < i_C, f"unexpected order: A={i_A} B={i_B} R={i_R} C={i_C}"

# Each section is preceded by a comment-block marker line. Capture
# the comment marker BEFORE the h2 so we keep the visual separator.
COMMENT = "<!-- ============================================================ -->\n"

# Find the start of the comment block that precedes Appendix A
# Look backwards from i_A for the comment marker
search_start_A = text.rfind(COMMENT, 0, i_A)
assert search_start_A != -1
A_block_start = search_start_A

# End of Appendix B = start of comment marker before References
search_start_R = text.rfind(COMMENT, 0, i_R)
assert search_start_R != -1
B_block_end = search_start_R

# So the A+B block to move is text[A_block_start:B_block_end]
ab_block = text[A_block_start:B_block_end]

# Find the comment marker before Appendix C — we'll insert A+B BEFORE that marker
# (so visually it reads: References block / [marker] / Appendix A / [marker] / Appendix B / [marker] / Appendix C)
search_start_C = text.rfind(COMMENT, 0, i_C)
assert search_start_C != -1
C_block_start = search_start_C

# Cut and re-paste
before = text[:A_block_start]
between = text[B_block_end:C_block_start]
after = text[C_block_start:]

new_text = before + between + ab_block + after

# Sanity: same total length
assert len(new_text) == len(text), f"length mismatch: {len(new_text)} vs {len(text)}"

# Sanity: order is now §11 ... §12 References ... Appendix A ... Appendix B ... Appendix C ...
new_i_R = new_text.index(REFS_START_ANCHOR)
new_i_A = new_text.index(A_START_ANCHOR)
new_i_B = new_text.index(B_START_ANCHOR)
new_i_C = new_text.index(C_START_ANCHOR)

assert new_i_R < new_i_A < new_i_B < new_i_C, (
    f"new order wrong: R={new_i_R} A={new_i_A} B={new_i_B} C={new_i_C}"
)

HTML.write_text(new_text, encoding="utf-8")
print(f"Moved Appendices A + B to after References.")
print(f"New order: References ({new_i_R}) -> Appendix A ({new_i_A}) "
      f"-> Appendix B ({new_i_B}) -> Appendix C ({new_i_C}).")
