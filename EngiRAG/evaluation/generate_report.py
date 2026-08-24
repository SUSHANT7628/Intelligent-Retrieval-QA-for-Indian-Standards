import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_pipeline import EngiRAG
from src.evaluation.metrics import recall_at_k, precision_at_k, reciprocal_rank


def run(cases_path='evaluation_questions.json', output='evaluation/report.json'):
    cases=json.loads(Path(cases_path).read_text())
    engine=EngiRAG()
    rows=[]
    for case in cases:
        results=engine.retrieve(case['question'], k=5, candidate_k=15)
        ids=[r['metadata'].get('source') for r in results]
        expected=case.get('expected_sources', [])
        rows.append({
            'question': case['question'],
            'recall_at_5': recall_at_k(ids, expected, 5),
            'precision_at_5': precision_at_k(ids, expected, 5),
            'mrr': reciprocal_rank(ids, expected),
            'retrieved_sources': ids,
        })
    def avg(key):
        vals=[r[key] for r in rows if r[key] is not None]
        return round(statistics.mean(vals), 4) if vals else None
    report={
        'corpus_cases': len(rows),
        'metrics': {'recall_at_5':avg('recall_at_5'),'precision_at_5':avg('precision_at_5'),'mrr':avg('mrr')},
        'cases': rows,
        'note':'Metrics are corpus-dependent; do not use them on a resume until evaluated on your final benchmark corpus.'
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__=='__main__': run()
