from pathlib import Path
import fitz


def extract_pdf(pdf_path: str):
    path = Path(pdf_path)
    doc = fitz.open(path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text, "source": path.name})
    return pages
