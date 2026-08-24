# EngiRAG Evaluation Protocol

Build the benchmark from your actual engineering corpus. Do not invent results.

## Recommended 50-question split

- 15 direct factual questions
- 10 numerical/technical questions
- 10 multi-document questions
- 10 comparison questions
- 5 unanswerable questions

For each answerable question, list the document(s) that contain the evidence in `expected_sources`.

## Experiments

### Production retrieval

```bash
python evaluation/run_eval.py
```

### Reranker ablation

```bash
python evaluation/ablation.py
```

The ablation compares the same hybrid retriever with and without the cross-encoder reranker.

### Full latency/citation benchmark

```bash
python -m src.evaluation.benchmark
```

Report:

- Recall@5
- Precision@5
- MRR
- Median retrieval latency
- End-to-end latency
- Citation coverage

For unanswerable questions, separately inspect whether the model declines to answer rather than measuring source recall.
