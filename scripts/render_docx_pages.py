"""Render DOCXs to PDF via Word COM, then rasterize a few pages to PNG for
visual confirmation.

Per the html2doc SKILL.md robustness rule "Verify the rendered output, not just
the element counts" and the global CLAUDE.md "Verify a build artifact's CONTENTS,
not just that it ran".
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

DOCXS = [DOCS / "synsmith.docx", DOCS / "synsmith_2col.docx"]


def docx_to_pdf(docx_path: Path, pdf_path: Path) -> None:
    """Use Word COM ExportAsFixedFormat (17 = wdExportFormatPDF)."""
    import win32com.client  # pywin32

    word = win32com.client.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(docx_path.resolve()), ReadOnly=True)
        try:
            doc.ExportAsFixedFormat(
                OutputFileName=str(pdf_path.resolve()),
                ExportFormat=17,  # wdExportFormatPDF
            )
        finally:
            doc.Close(SaveChanges=False)
    finally:
        word.Quit()


def rasterize_pages(pdf_path: Path, out_dir: Path, pages: list[int]) -> list[Path]:
    """Use PyMuPDF (fitz). Pages are 1-indexed for user convention, converted
    to 0-indexed for fitz."""
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    doc = fitz.open(pdf_path)
    try:
        for p1 in pages:
            i = p1 - 1
            if i < 0 or i >= len(doc):
                continue
            page = doc[i]
            pix = page.get_pixmap(dpi=140)
            out = out_dir / f"{pdf_path.stem}_p{p1:02d}.png"
            pix.save(out)
            written.append(out)
        n = len(doc)
    finally:
        doc.close()
    print(f"  pages total: {n}")
    return written


for docx in DOCXS:
    if not docx.exists():
        print(f"{docx.name}: missing")
        continue
    pdf = docx.with_suffix(".pdf")
    out_dir = DOCS / "_verify_pages"

    print(f"=== {docx.name} ===")
    print(f"  -> {pdf.name}")
    docx_to_pdf(docx, pdf)
    print(f"  PDF size: {pdf.stat().st_size / 1024 / 1024:.2f} MB")
    pages = rasterize_pages(pdf, out_dir, [1, 2, 5, 10, 15, 20])
    print(f"  rendered pages: {[p.name for p in pages]}")
    print()
