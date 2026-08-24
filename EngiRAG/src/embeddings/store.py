from sentence_transformers import SentenceTransformer
import chromadb


class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", db_path="chroma_db"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("engineering_docs")

    def add(self, chunks):
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, normalize_embeddings=True).tolist()
        ids = [f"{c['source']}-{c['page']}-{i}" for i, c in enumerate(chunks)]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
        self.collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    def search(self, query, k=5):
        vector = self.model.encode([query], normalize_embeddings=True).tolist()[0]
        return self.collection.query(query_embeddings=[vector], n_results=k)
