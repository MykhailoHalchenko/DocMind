# DocMind — Knowledge Base Query System

AI-powered system for indexing and querying scientific documents using RAG (Retrieval-Augmented Generation).

## Architecture

```
project/
├── main.py                  # CLI entrypoint
├── frontend/
│   ├── index.html           # Web interface
│   ├── script.js
│   └── style.css
├── mcp-serv/
│   ├── app.py               # FastAPI server + MCP server
│   ├── chunking.py          # Text splitting
│   ├── classifier.py        # LLM-1: chunk tagging + user intent
│   ├── config.py            # Settings via pydantic-settings
│   ├── dataset_loader.py    # PDF / JSON / CSV / TXT / MD loader
│   ├── embeddings.py        # OpenAI embeddings
│   ├── evaluation.py        # LLM-as-a-judge evaluation
│   ├── llm.py               # LLM-2: RAG answer + Map-Reduce summarize
│   ├── pdf_parser.py        # PyMuPDF text extraction
│   └── vector_db.py         # Qdrant async client
└── datasets/                # Sample datasets
```

**Pipeline:**
```
PDF/JSON/CSV/TXT/MD → clean → chunk → NER (optional) → classify (LLM-1)
    → embed (OpenAI) → store (Qdrant)

Query → intent (LLM-1) → semantic search (Qdrant)
    → RAG answer (LLM-2 / GLM-5.1) → cited response
```

---

## Requirements

- Python 3.11+
- [Qdrant](https://qdrant.tech/) (cloud or local via Docker)
- API keys: OpenAI + NVIDIA NIM (for Qwen + GLM)

---

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd <project-dir>

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (only needed if using --ner)
python -m spacy download en_core_web_sm
```

---

## Configuration

Create a `.env` file in the project root (copy from `.env.example`):

```env
# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=knowledge_base

# OpenAI (embeddings)
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Fast LLM — Qwen (classification, chunk tagging)
FAST_LLM_MODEL=qwen/qwen3.5-397b-a17b
FAST_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
FAST_LLM_API_KEY=nvapi-...

# Strong LLM — GLM-5.1 (RAG answers, summarization)
STRONG_LLM_MODEL=z-ai/glm-5.1
STRONG_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
STRONG_LLM_API_KEY=nvapi-...

# Chunking
CHUNK_SIZE=200
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

> **Note:** The `.env` file is already gitignored. Never commit API keys.

---

## Usage

### Start the web server

```bash
python main.py server
```

Opens the browser automatically at `http://localhost:8000`.

```bash
# Custom port, no browser auto-open
python main.py server --port 9000 --no-browser
```

### Index documents via CLI

```bash
# Index a PDF
python main.py index path/to/paper.pdf

# Index with chunk classification (adds section_type tags)
python main.py index path/to/paper.pdf

# Index with NER (extracts persons, orgs, dates)
python main.py index path/to/paper.pdf --ner

# Skip classification (faster)
python main.py index path/to/paper.pdf --no-classify
```

Supported formats via web upload: `.pdf`, `.json`, `.csv`, `.txt`, `.md`

### Run evaluation suite

```bash
python main.py eval
```

---

## API Endpoints

Server base URL: `http://localhost:8000`

### `POST /query` — Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the sample size in the study?",
    "filters": {"category": "Results"}
  }'
```

Response:
```json
{
  "answer": "The study enrolled 120 participants [chunk_abc123].",
  "intent": {"intent": "FIND_FACT", "filters": {"category": "Results"}, "complexity": "simple"},
  "sources": [{"id": "chunk_abc123", "score": 0.91, "text": "...", "metadata": {}}],
  "token_usage": {"input_tokens": 12, "output_tokens": 24, "total_tokens": 36}
}
```

### `POST /upload` — Upload and index a file

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@paper.pdf"
```

Response:
```json
{
  "file": "paper.pdf",
  "type": "pdf",
  "status": "success",
  "chunks_indexed": 47
}
```

### `POST /index` — Index raw documents (JSON body)

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{"text": "Your text here", "source": "manual"}],
    "auto_chunk": true,
    "classify_chunks": true
  }'
```

### `POST /summarize` — Map-Reduce summarization

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"chunks": ["First chunk text...", "Second chunk text..."]}'
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## How the pipeline works

### Indexing
1. **Parse** — PDF text extracted page by page, headers/footers removed
2. **Chunk** — Text split into overlapping word windows (`CHUNK_SIZE=200`, `CHUNK_OVERLAP=50`)
3. **Classify** *(optional)* — Qwen tags each chunk: `category`, `keywords`, `sentiment`
4. **NER** *(optional)* — spaCy extracts persons, orgs, dates, locations
5. **Embed** — OpenAI `text-embedding-3-small` converts chunks to 1536-dim vectors
6. **Store** — Vectors + metadata saved to Qdrant

### Querying
1. **Intent** — Qwen classifies user query: intent + section hint + keywords
2. **Search** — Query embedded → semantic search in Qdrant (filters applied from intent)
3. **Answer** — GLM-5.1 receives top-K chunks as context, answers with citations `[chunk_id]`
4. If facts not found → responds `"Insufficient data in the provided sources."`

### Summarization (Map-Reduce)
1. **Map** — Each chunk summarized independently by GLM-5.1
2. **Reduce** — All partial summaries merged into one structured final summary

---

## Evaluation

The evaluation framework (`evaluation.py`) scores system answers without an external judge API — using local heuristics:

| Metric | Description |
|--------|-------------|
| `hallucination` | `none` / `partial` / `yes` — numbers in answer vs source |
| `coverage_score` | 0–10 — key words from expected answer found in response |
| `citation_accuracy` | `accurate` / `missing` — `[chunk_id]` format present |
| `verdict` | `pass` if coverage ≥ 7 and citations present |

```bash
python main.py eval
```

To add your own test cases, edit `run_eval()` in `main.py`:

```python
test_cases = [
    {
        "question": "What method was used?",
        "context": "The study used a randomized controlled trial with 200 subjects.",
        "answer": "A randomized controlled trial was used [chunk_1].",
        "expected": "randomized controlled trial 200",
    },
]
```

---

## Datasets

The `datasets/` folder includes sample files for testing:

| File | Description |
|------|-------------|
| `arXiv_scientific dataset.csv` | Scientific paper metadata from arXiv |
| `arxiv-metadata-oai-snapshot.json` | Full arXiv metadata snapshot |
| `IMDB Dataset.csv` | Movie reviews (sentiment testing) |
| `ner_dataset.csv` | Named entity recognition test data |
| `test_sciq.csv` / `train_sciq.csv` | SciQ science Q&A benchmark |

Index a sample dataset:
```bash
# Via web upload at http://localhost:8000
# Or via API:
curl -X POST http://localhost:8000/upload \
  -F "file=@datasets/arXiv_scientific dataset.csv"
```

---

## Troubleshooting

**`Settings validation error` on startup**
→ Check that `.env` exists in the project root and all required keys are set.

**`Collection does not exist` error**
→ The collection is created automatically on first run. Check `QDRANT_URL` and `QDRANT_API_KEY`.

**Empty search results**
→ Index documents first via `/upload` or `python main.py index <file>`.

**spaCy model not found**
→ Run `python -m spacy download en_core_web_sm` (only needed with `--ner` flag).

**`RateLimitError` from OpenAI or NVIDIA**
→ Embeddings and LLM calls have automatic retry with exponential backoff. Wait and retry.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Server | FastAPI + uvicorn |
| MCP Protocol | FastMCP |
| Vector DB | Qdrant (async) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Fast LLM | Qwen 3.5 via NVIDIA NIM |
| Strong LLM | GLM-5.1 via NVIDIA NIM |
| PDF Parsing | PyMuPDF (fitz) |
| NER | spaCy `en_core_web_sm` |
| Config | pydantic-settings |
| Token counting | tiktoken |
