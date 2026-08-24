import os
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query, results, top_k=5):
        if not results:
            return []
        pairs = [(query, r["text"]) for r in results]
        scores = self.model.predict(pairs)
        ranked = []
        for result, score in zip(results, scores):
            item = dict(result)
            item["rerank_score"] = float(score)
            ranked.append(item)
        return sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
