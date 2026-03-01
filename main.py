import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import AdvancedRAGPipeline


pipeline = AdvancedRAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Advanced RAG Chat API starting...")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Advanced RAG Chat API",
    description="Hybrid RAG pipeline using FAISS dense retrieval, BM25 sparse retrieval, and LLM reranking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual"


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    latency_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "advanced-rag-chat",
        "documents_indexed": len(pipeline.documents),
        "ingested_files": pipeline.ingested_files,
    }


@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        count = pipeline.ingest_text(request.text, source=request.source)
        return {
            "message": f"Ingested {count} chunks from source '{request.source}'.",
            "chunks_created": count,
            "total_documents": len(pipeline.documents),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    content = await file.read()
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"

    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        count = pipeline.ingest_file(tmp_path)
        return {
            "message": f"Ingested {count} chunks from '{file.filename}'.",
            "chunks_created": count,
            "total_documents": len(pipeline.documents),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    start = time.perf_counter()
    try:
        result = pipeline.query(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        session_id=session_id,
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        latency_ms=round(latency_ms, 2),
    )


@app.delete("/index")
def clear_index():
    global pipeline
    pipeline = AdvancedRAGPipeline()
    return {"message": "Index cleared."}
