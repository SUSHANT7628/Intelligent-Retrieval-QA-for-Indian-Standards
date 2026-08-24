from pathlib import Path
from src.ingestion.pdf_loader import extract_pdf
from src.chunking.text_chunker import chunk_pages
from src.embeddings.store import VectorStore


RAW_DIR = Path("data/raw")


def main():
    store = VectorStore()
    pdfs = list(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw")
        return
    for pdf in pdfs:
        print(f"Processing {pdf.name}...")
        pages = extract_pdf(str(pdf))
        chunks = chunk_pages(pages)
        store.add(chunks)
        print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()
