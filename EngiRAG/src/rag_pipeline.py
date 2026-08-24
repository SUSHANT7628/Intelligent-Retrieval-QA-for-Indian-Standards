import os
from dotenv import load_dotenv
from src.embeddings.store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.generation.rag import RAGGenerator

load_dotenv()

class EngiRAG:
    def __init__(self, use_reranker=None):
        self.store = VectorStore(db_path=os.getenv("CHROMA_PATH", "chroma_db"))
        self.retriever = HybridRetriever(self.store)
        self.generator = None
        self.use_reranker = (
            os.getenv("USE_RERANKER", "true").lower() == "true"
            if use_reranker is None else use_reranker
        )
        self.reranker = None

    def retrieve(self, question, k=5, candidate_k=15):
        results = self.retriever.search(question, k=candidate_k)
        if self.use_reranker and results:
            if self.reranker is None:
                from src.reranking.cross_encoder import CrossEncoderReranker
                self.reranker = CrossEncoderReranker()
            return self.reranker.rerank(question, results, top_k=k)
        return results[:k]

    def answer(self, question, k=5):
        contexts = self.retrieve(question, k=k)
        if not contexts:
            return {"answer": "No relevant information was found.", "sources": [], "contexts": []}
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "answer": "OPENAI_API_KEY is not configured. Retrieval succeeded; add an API key to enable generated answers.",
                "sources": [c["metadata"] for c in contexts], "contexts": contexts
            }
        if self.generator is None:
            self.generator = RAGGenerator()
        result = self.generator.answer(question, contexts)
        from src.citations.verifier import verify_citations
        result = {"answer": result, "sources": [c["metadata"] for c in contexts], "contexts": contexts}
        result["citation_check"] = verify_citations(result["answer"], contexts)
        return result
