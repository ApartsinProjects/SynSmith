import re
h = open("docs/index.html").read()
m = re.search(r'<section class="abstract">.*?</section>', h, re.DOTALL)
text = re.sub(r"<[^>]+>", "", m.group())
text = re.sub(r"\$[^$]+\$", "X", text)  # collapse KaTeX as placeholder X
text = re.sub(r"\s+", " ", text).strip()
words = text.split()
print(f"Abstract word count (KaTeX expressions counted as 1 word): {len(words)}")
print()
print(text)
