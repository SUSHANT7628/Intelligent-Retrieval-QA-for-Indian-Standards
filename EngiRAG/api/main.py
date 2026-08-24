from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.rag_pipeline import EngiRAG
from src.query.rewriter import QueryRewriter
from src.comparison.document_compare import compare_texts

app = FastAPI(title='EngiRAG API', version='0.5.0')
rag = EngiRAG()
rewriter = QueryRewriter()

class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)

class CompareRequest(BaseModel):
    document_a: str = Field(min_length=1)
    document_b: str = Field(min_length=1)

@app.get('/health')
def health():
    return {'status':'ok', 'reranker':rag.use_reranker, 'query_rewriting':rewriter.enabled}

@app.post('/retrieve')
def retrieve(req: QueryRequest):
    queries=rewriter.rewrite(req.question)
    merged=[]; seen=set()
    for q in queries:
        for item in rag.retrieve(q, k=req.top_k, candidate_k=max(15, req.top_k*3)):
            key=(item['metadata'].get('source'), item['metadata'].get('page'), item['text'][:80])
            if key not in seen:
                seen.add(key); merged.append(item)
    return {'queries':queries, 'results':merged[:req.top_k]}

@app.post('/ask')
def ask(req: QueryRequest):
    return rag.answer(req.question, req.top_k)

@app.post('/compare')
def compare(req: CompareRequest):
    return compare_texts(req.document_a, req.document_b)
