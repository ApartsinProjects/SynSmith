"""Rename CLI-flag-style baseline identifiers to academic display names.

Same pattern as the Baseline-3C / SynSmith renames:
- Prose: use the display name (no <code> wrapper, no underscores).
- CLI flag literals: keep in §6.3 footnote and §12 release notes.
- File paths: keep verbatim.

Display-name mapping:
- naive            -> Naive
- few_shot         -> Few-shot
- self_critique    -> Self-critique
- realism_only     -> Realism-only
- diversity_only   -> Diversity-only
- attribute_only   -> Attribute-only
- no_pack          -> Pack-OFF  (LOO ablation row; "Pack Discriminator OFF" in tables)
- no_mode_seeking  -> Mode-Seeking-OFF
- no_mode_hunter   -> Mode-Hunter-OFF
- no_coverage_hole -> Coverage-Hole-OFF

Substring-collision considerations:
- no_pack vs no_pack_vs: must NOT rename no_pack_vs (it's a CLI variant).
  We only rename "<code>no_pack</code>" (with closing tag) so no_pack_vs is safe.
- All other identifiers are distinct strings.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "docs" / "index.html"
text = HTML.read_text(encoding="utf-8")

# Pre-audit counts
import re
old_text = text


def count_in(pat: str, s: str) -> int:
    return s.count(pat)


print("Pre-edit audit:")
for name in ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "attribute_only", "no_pack",
             "no_mode_seeking", "no_mode_hunter", "no_coverage_hole"]:
    print(f"  <code>{name}</code>: {count_in(f'<code>{name}</code>', text)}")
print()

# ----------------------------------------------------------------------
# 1. Handle compound ensemble-pair strings in Appendix H Table H1 first.
# These have multiple names inside one <code> wrapper.
# ----------------------------------------------------------------------
COMPOUND_FIXES = [
    ("<code>self_critique + diversity_only</code>",
     "Self-critique + Diversity-only"),
    ("<code>self_critique + realism_only</code>",
     "Self-critique + Realism-only"),
    ("<code>realism_only + diversity_only</code>",
     "Realism-only + Diversity-only"),
]
for old, new in COMPOUND_FIXES:
    cnt = text.count(old)
    if cnt:
        text = text.replace(old, new)
        print(f"  compound: {old!r} -> {new!r}   x{cnt}")

# ----------------------------------------------------------------------
# 2. Handle Table H2 LOO ablation row labels: drop the <code>no_X</code>
# prefix, keep the parenthetical OFF label as the row text.
# ----------------------------------------------------------------------
LOO_ROW_FIXES = [
    ("<code>no_pack</code> (Pack Discriminator OFF)",
     "Pack Discriminator OFF"),
    ("<code>no_mode_seeking</code> (Mode-Seeking critic OFF)",
     "Mode-Seeking critic OFF"),
    ("<code>no_mode_hunter</code> (Mode Hunter OFF)",
     "Mode Hunter OFF"),
    ("<code>no_coverage_hole</code> (Coverage Hole Finder OFF)",
     "Coverage Hole Finder OFF"),
]
for old, new in LOO_ROW_FIXES:
    cnt = text.count(old)
    if cnt:
        text = text.replace(old, new)
        print(f"  loo-row: {old!r} -> {new!r}   x{cnt}")

# ----------------------------------------------------------------------
# 3. Bulk replace each remaining standalone <code>X</code> -> DisplayName
# ----------------------------------------------------------------------
DISPLAY = {
    "naive":            "Naive",
    "few_shot":         "Few-shot",
    "self_critique":    "Self-critique",
    "realism_only":     "Realism-only",
    "diversity_only":   "Diversity-only",
    "attribute_only":   "Attribute-only",
    # LOO variants outside the explicit row labels above (e.g. "(no_pack,
    # no_mode_seeking, no_mode_hunter, no_coverage_hole)" lists in prose)
    # become parenthetical descriptors. Keep them readable.
    "no_pack":          "Pack-OFF",
    "no_mode_seeking":  "Mode-Seeking-OFF",
    "no_mode_hunter":   "Mode-Hunter-OFF",
    "no_coverage_hole": "Coverage-Hole-OFF",
}
for cli, display in DISPLAY.items():
    pat = f"<code>{cli}</code>"
    cnt = text.count(pat)
    if cnt:
        text = text.replace(pat, display)
        print(f"  bulk: {pat!r} -> {display!r}   x{cnt}")

# ----------------------------------------------------------------------
# 4. Update §6.3 footnote so the new display names are listed + the CLI
# literals are preserved for backward compatibility.
# ----------------------------------------------------------------------
# Current §6.3 footnote has this clause (after the Baseline-3C rename):
#     The remaining condition labels (<code>naive</code>, <code>few_shot</code>,
#     <code>self_critique</code>, <code>realism_only</code>, <code>diversity_only</code>,
#     the six <code>no_X</code> leave-one-out variants) are stable identifiers
#     used in the open-source CLI flag <code>--conditions</code>.
#
# After our bulk replace above, the "<code>X</code>" strings have already been
# changed to display names. The footnote now reads with display names + a
# stale "<code>no_X</code>" placeholder. Replace the whole sentence cleanly.
OLD_FOOTNOTE_SEN = (
    "The remaining condition labels (Naive, Few-shot, "
    "Self-critique, Realism-only, Diversity-only, "
    "the six <code>no_X</code> leave-one-out variants) are stable identifiers "
    "used in the open-source CLI flag <code>--conditions</code>."
)
NEW_FOOTNOTE_SEN = (
    "The remaining baselines (<strong>Naive</strong>, <strong>Few-shot</strong>, "
    "<strong>Self-critique</strong>, <strong>Realism-only</strong>, "
    "<strong>Diversity-only</strong>, <strong>Attribute-only</strong>) and the "
    "four leave-one-out variants (<strong>Pack-OFF</strong>, "
    "<strong>Mode-Seeking-OFF</strong>, <strong>Mode-Hunter-OFF</strong>, "
    "<strong>Coverage-Hole-OFF</strong>) are stable identifiers in the "
    "open-source CLI flag <code>--conditions</code> under their historical "
    "underscore names (<code>naive</code>, <code>few_shot</code>, "
    "<code>self_critique</code>, <code>realism_only</code>, "
    "<code>diversity_only</code>, <code>attribute_only</code>, "
    "<code>no_pack</code>, <code>no_mode_seeking</code>, "
    "<code>no_mode_hunter</code>, <code>no_coverage_hole</code>); the display "
    "names in this paper use hyphens for readability."
)
cnt = text.count(OLD_FOOTNOTE_SEN)
if cnt:
    text = text.replace(OLD_FOOTNOTE_SEN, NEW_FOOTNOTE_SEN)
    print(f"  footnote §6.3: rewritten x{cnt}")
else:
    print(f"  WARNING: §6.3 footnote sentence not matched verbatim; check manually")

# ----------------------------------------------------------------------
# 5. Verify
# ----------------------------------------------------------------------
print()
print("Post-edit audit:")
for name in DISPLAY:
    pat = f"<code>{name}</code>"
    cnt = text.count(pat)
    if cnt:
        print(f"  {pat}: {cnt} remaining (legitimate CLI literals)")

# Save
HTML.write_text(text, encoding="utf-8")
print()
print("File written.")

# Also count total <code>...</code> hits for any of the underscore names
# that remain (after our edits, all such hits should be CLI-literal contexts
# in §6.3 footnote and §12 release notes).
total_underscore_code = 0
for name in DISPLAY:
    total_underscore_code += text.count(f"<code>{name}</code>")
print(f"Remaining <code>NAME</code> for renamed identifiers: {total_underscore_code}")
