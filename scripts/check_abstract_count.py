"""Count words in the current abstract for SC-11 budget check."""
import re
import pathlib

text = pathlib.Path("docs/index.html").read_text(encoding="utf-8")
m = re.search(r'<section class="abstract">.*?</section>', text, re.DOTALL)
if m is None:
    raise SystemExit("abstract block not found")
abstract = m.group(0)

# Strip HTML tags
prose = re.sub(r"<[^>]+>", " ", abstract)
# Drop KaTeX math delimiters and the math content (count as 1 token)
prose = re.sub(r"\\left|\\right", "", prose)
prose = re.sub(r"\$[^$]*\$", "X", prose)
# Drop residual LaTeX commands
prose = re.sub(r"\\[a-zA-Z]+", "", prose)

words = [w for w in prose.split() if w.strip()]
# Count sentences by terminal punctuation
sentences = re.findall(r"[A-Z][^.!?]*[.!?]", prose)
print(f"word count (prose, math collapsed): {len(words)}")
print(f"sentence count: {len(sentences)}")
print()
print("Sentences:")
for i, s in enumerate(sentences, 1):
    s = re.sub(r"\s+", " ", s).strip()
    print(f"  {i}. ({len(s.split())} words) {s[:100]}...")
