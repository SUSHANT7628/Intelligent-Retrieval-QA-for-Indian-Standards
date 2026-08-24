# EngiRAG — Engineering Document Intelligence

Portfolio-grade RAG system for engineering documents. The system combines PDF extraction, overlapping chunks, semantic retrieval, BM25 keyword retrieval, reciprocal-rank fusion, cross-encoder reranking, optional query rewriting, cited LLM answers, citation validation, and retrieval/latency evaluation.

## Architecture

PDFs → extraction → chunks → embeddings + BM25 → hybrid retrieval → reranking → LLM → cited answer → evaluation

## Features
- Page-aware PDF ingestion
- Semantic + keyword hybrid retrieval
- Cross-encoder reranking
- Query rewriting for difficult technical questions
- Page/source metadata preserved through the pipeline
- Citation validation
- FastAPI API
- Streamlit UI
- Retrieval and latency benchmark
- Basic table extraction via PyMuPDF where supported

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.ingest
streamlit run frontend/app.py
```

For generated answers, configure `OPENAI_API_KEY` in `.env`.

## Evaluation

See `evaluation/README.md` for the benchmark protocol. Populate `evaluation_questions.json` with questions and expected source filenames. Then run:

```bash
python evaluation/run_eval.py
python -m src.evaluation.benchmark
```

Do not claim performance numbers until the benchmark has been run on your own corpus.

## v0.5 — Portfolio / deployment stage

New additions:
- Automated evaluation report generation
- Retrieval benchmark scaffolding
- `/compare` API endpoint
- Dockerfile and docker-compose configuration
- Cleaner separation between evaluation and application runtime

### Evaluation workflow

1. Put your final permitted engineering corpus in `data/raw/`.
2. Create `evaluation_questions.json` with expected source filenames.
3. Build the index:

```bash
python -m src.ingest
```

4. Run retrieval metrics:

```bash
python evaluation/run_eval.py
python evaluation/ablation.py
python evaluation/generate_report.py
```

5. Inspect `evaluation/report.json`.

Do not report benchmark numbers until the benchmark contains enough representative questions and has been manually checked.

### Docker

```bash
docker compose up --build
```

API docs: `http://localhost:8000/docs`

## Portfolio evidence to capture

For your final README, include screenshots of:
- Architecture
- Streamlit UI
- Retrieved evidence with page metadata
- Example cited answer
- Evaluation report
- API Swagger page
- Docker deployment


## Recommended portfolio experiment

Compare four retrieval configurations on the same benchmark: semantic-only, BM25-only, hybrid RRF, and hybrid + cross-encoder reranking. Record Recall@5, MRR, precision, and latency. Then add 5 unanswerable questions to test refusal/grounding behavior.

## Resume positioning

Use measured results only. A strong final bullet structure is: “Built a page-aware engineering-document RAG system using hybrid semantic/BM25 retrieval and cross-encoder reranking; evaluated retrieval on a 50-question benchmark and measured [actual result] Recall@5 with [actual result] median latency.”
