from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from utils.chunking import semantic_chunk
from utils.reranker import rerank_documents


EMBEDDING_DIM = 1536  # text-embedding-3-small


class AdvancedRAGPipeline:
    """
    Hybrid RAG pipeline combining FAISS dense retrieval and BM25 sparse retrieval,
    fused with Reciprocal Rank Fusion and LLM-based reranking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_final: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_final = top_k_final

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.documents: list[Document] = []
        self.doc_embeddings: list[np.ndarray] = []
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ingested_files: list[str] = []

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Ingest raw text. Returns number of chunks created."""
        chunks = semantic_chunk(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        docs = [
            Document(page_content=chunk, metadata={"source": source, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

        return self._index_documents(docs, source)

    def ingest_file(self, file_path: str) -> int:
        """Ingest a text or PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            text = self._extract_pdf(file_path)
        else:
            text = path.read_text(encoding="utf-8")

        count = self.ingest_text(text, source=path.name)
        self.ingested_files.append(path.name)
        return count

    def _extract_pdf(self, path: str) -> str:
        try:
            import fitz
            doc = fitz.open(path)
            return "\n\n".join(page.get_text() for page in doc)
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install pymupdf")

    def _index_documents(self, docs: list[Document], source: str) -> int:
        texts = [doc.page_content for doc in docs]
        embeddings = self.embeddings.embed_documents(texts)

        vecs = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vecs)

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)

        self.faiss_index.add(vecs)
        self.documents.extend(docs)
        self.doc_embeddings.extend(embeddings)

        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents, k=self.top_k_sparse
        )

        print(f"Indexed {len(docs)} chunks from '{source}'")
        return len(docs)

    def retrieve(self, query: str) -> list[Document]:
        """Hybrid retrieval: fuse dense and sparse results, then rerank."""
        if self.faiss_index is None or not self.documents:
            raise RuntimeError("No documents ingested. Call ingest_text() or ingest_file() first.")

        q_emb = np.array(
            [self.embeddings.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(q_emb)
        scores, indices = self.faiss_index.search(q_emb, self.top_k_dense)

        dense_docs = [
            self.documents[i] for i in indices[0] if i < len(self.documents)
        ]

        sparse_docs = self.bm25_retriever.invoke(query) if self.bm25_retriever else []

        fused = self._reciprocal_rank_fusion(dense_docs, sparse_docs)
        return rerank_documents(query, fused, top_k=self.top_k_final)

    def _reciprocal_rank_fusion(
        self,
        dense: list[Document],
        sparse: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """Merge dense and sparse rankings using RRF scoring."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            doc_map[key] = doc

        ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in ranked_keys]

    def query(self, question: str) -> dict:
        """Run the full RAG pipeline and return an answer with sources."""
        retrieved = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in retrieved
        )

        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context doesn't contain enough information to answer, say so. "
            "Ground your answers in the context and cite sources where possible."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, accurate answer based on the context above."
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        sources = list({doc.metadata.get("source", "unknown") for doc in retrieved})

        return {
            "answer": response.content,
            "sources": sources,
            "chunks_used": len(retrieved),
        }

    def save_index(self, path: str = "faiss_index.bin"):
        if self.faiss_index:
            faiss.write_index(self.faiss_index, path)

    def load_index(self, path: str = "faiss_index.bin"):
        if Path(path).exists():
            self.faiss_index = faiss.read_index(path)
