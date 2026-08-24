import json, time
from pathlib import Path
from src.rag_pipeline import EngiRAG


def run_benchmark(path='evaluation_questions.json'):
    cases=json.loads(Path(path).read_text())
    engine=EngiRAG()
    rows=[]
    for c in cases:
        t=time.perf_counter()
        result=engine.answer(c['question'], k=5)
        latency_ms=(time.perf_counter()-t)*1000
        check=result.get('citation_check', {})
        rows.append({
            'question':c['question'],
            'latency_ms':round(latency_ms,1),
            'citation_coverage':check.get('citation_coverage', None),
            'valid_citations':check.get('valid_citations', []),
            'sources':[x.get('source') for x in result.get('sources', [])]
        })
    return rows

if __name__=='__main__':
    print(json.dumps(run_benchmark(), indent=2))
