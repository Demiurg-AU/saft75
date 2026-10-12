"""
Vector storage for the Temporal Knowledge Graph.
Supports ChromaDB (recommended) and a simple in-memory fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class VectorDocument:
    """Represents a document stored in the vector store."""

    def __init__(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        created_at: Optional[datetime] = None,
    ) -> None:
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding
        self.created_at = created_at or datetime.now(timezone.utc)


class SimpleEmbedder:
    """
    Minimal TF-IDF-inspired embedding for fallback when
    sentence-transformers is not available.
    Produces a deterministic 64-dim bag-of-words vector.
    """

    DIM = 64

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = [0.0] * self.DIM
        for token in tokens:
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class InMemoryVectorStore:
    """
    Pure-Python in-memory vector store.
    Falls back to cosine similarity search when ChromaDB is unavailable.
    """

    def __init__(self, embedder: Optional[Any] = None) -> None:
        self._embedder = embedder or SimpleEmbedder()
        self._docs: Dict[str, VectorDocument] = {}

    def add_documents(self, documents: List[VectorDocument]) -> None:
        texts = [d.text for d in documents]
        embeddings = self._embedder.encode(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self._docs[doc.doc_id] = doc

    def search(self, query: str, top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        if not self._docs:
            return []
        query_emb = self._embedder.encode([query])[0]
        scores: List[Tuple[VectorDocument, float]] = []
        for doc in self._docs.values():
            if doc.embedding is None:
                continue
            score = _cosine(query_emb, doc.embedding)
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._docs:
            del self._docs[doc_id]
            return True
        return False

    def count(self) -> int:
        return len(self._docs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "metadata": d.metadata,
                    "created_at": d.created_at.isoformat(),
                }
                for d in self._docs.values()
            ]
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        docs = []
        for item in data.get("documents", []):
            doc = VectorDocument(
                doc_id=item["doc_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            docs.append(doc)
        if docs:
            self.add_documents(docs)


class ChromaVectorStore:
    """
    Vector store backed by ChromaDB with sentence-transformers embeddings.
    Recommended for production use.
    """

    def __init__(self, collection_name: str = "tkg_memory", persist_directory: Optional[str] = None) -> None:
        if not HAS_CHROMADB:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        settings = Settings(anonymized_telemetry=False)
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory, settings=settings)
        else:
            self._client = chromadb.Client(settings=settings)
        self._collection = self._client.get_or_create_collection(collection_name)
        if HAS_SENTENCE_TRANSFORMERS:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self._embedder = SimpleEmbedder()

    def add_documents(self, documents: List[VectorDocument]) -> None:
        texts = [d.text for d in documents]
        embeddings = self._embedder.encode(texts)
        ids = [d.doc_id for d in documents]
        metadatas = [
            {**d.metadata, "_created_at": d.created_at.isoformat()}
            for d in documents
        ]
        self._collection.add(
            ids=ids,
            embeddings=[list(e) for e in embeddings],
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[VectorDocument, float]]:
        query_emb = self._embedder.encode([query])[0]
        results = self._collection.query(
            query_embeddings=[list(query_emb)],
            n_results=min(top_k, self._collection.count()),
        )
        output: List[Tuple[VectorDocument, float]] = []
        if not results["ids"]:
            return output
        for doc_id, text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            created_at_str = meta.pop("_created_at", None)
            doc = VectorDocument(
                doc_id=doc_id,
                text=text,
                metadata=meta,
                created_at=datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc),
            )
            score = 1.0 - distance
            output.append((doc, score))
        return output

    def count(self) -> int:
        return self._collection.count()


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


def create_vector_store(
    backend: str = "memory",
    collection_name: str = "tkg_memory",
    persist_directory: Optional[str] = None,
) -> Any:
    """
    Factory function to create a vector store.

    Args:
        backend: "memory" or "chromadb"
        collection_name: Collection name for ChromaDB
        persist_directory: Persistence directory for ChromaDB

    Returns:
        An InMemoryVectorStore or ChromaVectorStore instance.
    """
    if backend == "chromadb":
        return ChromaVectorStore(collection_name=collection_name, persist_directory=persist_directory)
    return InMemoryVectorStore()
