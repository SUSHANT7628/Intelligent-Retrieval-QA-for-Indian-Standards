import json, sys, time, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_pipeline import EngiRAG
from src.evaluation.metrics import recall_at_k, precision_at_k, reciprocal_rank


def evaluate_mode(cases, use_reranker):
    engine = EngiRAG(use_reranker=use_reranker)
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        results = engine.retrieve(case['question'], k=5, candidate_k=15)
        elapsed = (time.perf_counter() - t0) * 1000
        ids = [r['metadata'].get('source') for r in results]
        expected = case.get('expected_sources', [])
        rows.append({
            'question': case['question'],
            'recall@5': recall_at_k(ids, expected, 5),
            'precision@5': precision_at_k(ids, expected, 5),
            'mrr': reciprocal_rank(ids, expected),
            'latency_ms': elapsed,
        })
    return rows


def summarize(rows):
    if not rows:
        return {}
    return {
        'n': len(rows),
        'mean_recall@5': round(statistics.mean(r['recall@5'] for r in rows), 4),
        'mean_precision@5': round(statistics.mean(r['precision@5'] for r in rows), 4),
        'mean_mrr': round(statistics.mean(r['mrr'] for r in rows), 4),
        'median_latency_ms': round(statistics.median(r['latency_ms'] for r in rows), 1),
    }


def main(path='evaluation_questions.json'):
    cases = json.loads(Path(path).read_text())
    reports = {}
    for label, enabled in [('without_reranker', False), ('with_reranker', True)]:
        rows = evaluate_mode(cases, enabled)
        reports[label] = {'summary': summarize(rows), 'rows': rows}
    Path('evaluation/ablation_report.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps({k: v['summary'] for k, v in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
