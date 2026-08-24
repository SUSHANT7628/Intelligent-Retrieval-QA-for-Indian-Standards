from rank_bm25 import BM25Okapi
import re


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """Combine semantic retrieval with BM25 keyword retrieval using reciprocal rank fusion."""
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._bm25 = None
        self._docs = []

    def build_keyword_index(self):
        data = self.vector_store.collection.get(include=["documents", "metadatas"])
        self._docs = [
            {"text": d, "metadata": m}
            for d, m in zip(data.get("documents", []), data.get("metadatas", []))
        ]
        self._bm25 = BM25Okapi([tokenize(d["text"]) for d in self._docs]) if self._docs else None

    @staticmethod
    def _rrf(results, k=60):
        scores = {}
        payload = {}
        for rank, item in enumerate(results, 1):
            key = item["id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            payload[key] = item
        return sorted(payload.values(), key=lambda x: scores[x["id"]], reverse=True)

    def search(self, query: str, k: int = 5):
        semantic = self.vector_store.search(query, k=max(k, 10))
        semantic_items = []
        for i, doc in enumerate(semantic.get("documents", [[]])[0]):
            semantic_items.append({
                "id": semantic["ids"][0][i],
                "text": doc,
                "metadata": semantic["metadatas"][0][i],
            })

        if self._bm25 is None:
            self.build_keyword_index()
        keyword_items = []
        if self._bm25:
            scores = self._bm25.get_scores(tokenize(query))
            for idx in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:max(k, 10)]:
                d = self._docs[idx]
                keyword_items.append({
                    "id": f"{d['metadata'].get('source')}-{d['metadata'].get('page')}-{idx}",
                    "text": d["text"],
                    "metadata": d["metadata"],
                })

        return self._rrf(semantic_items + keyword_items)[:k]
