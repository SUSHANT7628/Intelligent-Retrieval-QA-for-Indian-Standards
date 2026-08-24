def chunk_pages(pages, chunk_size=900, overlap=150):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                })
            if end == len(text):
                break
            start = end - overlap
    return chunks
