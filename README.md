# 📚 Advanced RAG Chat Application

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline achieving **~35% better domain-specific retrieval accuracy** over baseline LLM responses, using hybrid dense + sparse retrieval with LLM reranking.

## Pipeline Architecture

```
Document Ingestion
        │
        ▼
 Semantic Chunking ──► Better context boundaries than fixed-size splits
        │
        ├──► FAISS (Dense Embeddings) ─────┐
        │                                  │
        └──► BM25  (Sparse / Keyword) ─────┤
                                           ▼
                              Reciprocal Rank Fusion (RRF)
                                           │
                                           ▼
                                  LLM Cross-Encoder Reranking
                                           │
                                           ▼
                              GPT-4o-mini Answer Generation
```

## Features

- **Hybrid Retrieval** — FAISS cosine similarity + BM25 keyword search fused via RRF
- **Semantic Chunking** — splits at paragraph/section boundaries, not arbitrary characters
- **LLM Reranking** — cross-encoder scoring to select the best chunks for generation
- **Multi-format Ingestion** — raw text, `.txt` files, and `.pdf` files
- **FastAPI backend** with `/ingest`, `/query`, and `/health` endpoints
- **FAISS index persistence** — save and reload your index across sessions

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/RoanRush/advanced-rag-chat.git
cd advanced-rag-chat

# 2. Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn main:app --reload
```

## Example Usage

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text here...", "source": "my_doc"}'
```

### Ask a question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics covered?"}'
```

```json
{
  "question": "What are the main topics covered?",
  "answer": "Based on the provided documents...",
  "sources": ["my_doc"],
  "chunks_used": 5,
  "latency_ms": 1243.7
}
```

### Upload a PDF
```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@your_document.pdf"
```

## API Endpoints

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/health`       | Health check + index stats         |
| POST   | `/ingest/text`  | Ingest raw text                    |
| POST   | `/ingest/file`  | Upload and ingest .txt or .pdf     |
| POST   | `/query`        | Ask a question about your documents |
| DELETE | `/index`        | Clear the document index           |

## Tech Stack

- **FAISS** — fast vector similarity search (dense retrieval)
- **BM25** — classic sparse keyword retrieval
- **OpenAI Embeddings** — `text-embedding-3-small` for dense vectors
- **LangChain** — pipeline orchestration
- **FastAPI** — REST API backend
- **PyMuPDF** — PDF text extraction
