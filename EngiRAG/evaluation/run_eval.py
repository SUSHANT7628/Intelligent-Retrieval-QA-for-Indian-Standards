import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_pipeline import EngiRAG
from src.evaluation.metrics import recall_at_k, precision_at_k, reciprocal_rank

def main():
    cases = json.loads(Path("evaluation_questions.json").read_text())
    engine = EngiRAG()
    rows=[]
    for case in cases:
        results=engine.retrieve(case["question"], k=5, candidate_k=15)
        ids=[r["metadata"].get("source") for r in results]
        expected=case.get("expected_sources", [])
        rows.append({"question":case["question"], "recall@5":recall_at_k(ids, expected, 5), "precision@5":precision_at_k(ids, expected, 5), "rr":reciprocal_rank(ids, expected), "retrieved":ids})
    print(json.dumps(rows, indent=2))

if __name__ == "__main__": main()
