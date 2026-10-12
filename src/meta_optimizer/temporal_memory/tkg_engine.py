"""
Main Temporal Knowledge Graph engine.

Provides the high-level TKG interface combining graph storage,
vector embeddings, hybrid retrieval, and temporal queries.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .graph_store import GraphEdge, GraphNode, NetworkXGraphStore, create_edge, create_node
from .hybrid_retriever import HybridResult, HybridRetriever
from .temporal_query import TemporalQuery
from .vector_store import VectorDocument, create_vector_store


class EntityType(str, Enum):
    """Predefined entity types for TKG nodes."""
    OPTIMIZATION = "optimization"
    MODEL = "model"
    EXPERIMENT = "experiment"
    INSIGHT = "insight"
    SESSION = "session"
    CONCEPT = "concept"
    TOOL = "tool"
    RESULT = "result"
    PERSON = "person"
    TASK = "task"
    GENERIC = "generic"


class TemporalKnowledgeGraph:
    """
    Temporal Knowledge Graph combining graph + vector storage.

    Provides:
    - Knowledge ingestion with automatic entity extraction
    - Hybrid search (graph + vector)
    - Temporal queries ("what changed since last week?")
    - Cross-session synthesis
    - Relationship modeling
    - Edge invalidation

    Usage::

        tkg = TemporalKnowledgeGraph()
        tkg.add_knowledge(
            text="MobileNet optimized with INT8 achieving 4.2x speedup",
            entity_type=EntityType.OPTIMIZATION,
            metadata={"model": "MobileNet", "speedup": 4.2},
        )
        results = tkg.search("MobileNet optimization")
        recent = tkg.what_changed_since(days_ago=7)
    """

    def __init__(
        self,
        backend: str = "networkx",
        vector_backend: str = "memory",
        storage_path: Optional[str] = None,
        alpha: float = 0.5,
    ) -> None:
        """
        Args:
            backend: Graph backend ("networkx" or "neo4j").
            vector_backend: Vector backend ("memory" or "chromadb").
            storage_path: Directory for persistent storage (optional).
            alpha: Hybrid retrieval weight — 1.0=pure vector, 0.0=pure graph.
        """
        if backend != "networkx":
            raise NotImplementedError(
                f"Backend '{backend}' is not yet implemented. Use 'networkx'."
            )
        self._graph = NetworkXGraphStore()
        self._vector = create_vector_store(backend=vector_backend)
        self._retriever = HybridRetriever(self._graph, self._vector, alpha=alpha)
        self._temporal = TemporalQuery(self._graph)
        self._storage_path = storage_path

        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------
    # Knowledge ingestion
    # ------------------------------------------------------------------

    def add_knowledge(
        self,
        text: str,
        entity_type: EntityType = EntityType.GENERIC,
        metadata: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Add a piece of knowledge to the TKG.

        Args:
            text: Human-readable description of the knowledge.
            entity_type: Category of the entity.
            metadata: Additional structured metadata.
            relationships: Optional list of {"target_id": ..., "relation": ...} dicts.

        Returns:
            The node_id of the created node.
        """
        meta = metadata or {}
        node = create_node(
            entity_type=entity_type.value if isinstance(entity_type, EntityType) else entity_type,
            label=text[:200],
            properties=meta,
        )
        node_id = self._graph.add_node(node)

        # Add to vector store
        doc = VectorDocument(
            doc_id=str(uuid.uuid4()),
            text=text,
            metadata={**meta, "node_id": node_id, "entity_type": node.entity_type},
            created_at=node.created_at,
        )
        self._vector.add_documents([doc])

        # Add relationships
        if relationships:
            for rel in relationships:
                target_id = rel.get("target_id")
                relation = rel.get("relation", "related_to")
                if target_id and self._graph.get_node(target_id):
                    edge = create_edge(node_id, target_id, relation)
                    self._graph.add_edge(edge)

        if self._storage_path:
            self._save()

        return node_id

    def update_knowledge(
        self,
        node_id: str,
        properties: Dict[str, Any],
    ) -> bool:
        """Update properties of an existing knowledge node."""
        result = self._graph.update_node(node_id, properties)
        if result and self._storage_path:
            self._save()
        return result

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a directed relationship between two existing nodes."""
        if not self._graph.get_node(source_id) or not self._graph.get_node(target_id):
            return None
        edge = create_edge(source_id, target_id, relation_type, weight, properties)
        edge_id = self._graph.add_edge(edge)
        if self._storage_path:
            self._save()
        return edge_id

    def invalidate_relationship(self, edge_id: str) -> bool:
        """Mark a relationship as no longer valid (soft delete)."""
        result = self._graph.invalidate_edge(edge_id)
        if result and self._storage_path:
            self._save()
        return result

    # ------------------------------------------------------------------
    # Search and retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[EntityType] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search over the TKG.

        Returns a ranked list of result dicts.
        """
        et = entity_type.value if isinstance(entity_type, EntityType) else entity_type
        results = self._retriever.retrieve(query, top_k=top_k, entity_type=et, since=since)
        return [r.to_dict() for r in results]

    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Walk the graph from node_id and return neighboring nodes."""
        neighbors = self._retriever.retrieve_neighbors(node_id, relation_type, depth)
        return [n.to_dict() for n in neighbors]

    # ------------------------------------------------------------------
    # Temporal queries
    # ------------------------------------------------------------------

    def what_changed_since(
        self,
        days_ago: int = 7,
        entity_type: Optional[EntityType] = None,
    ) -> List[Dict[str, Any]]:
        """Return everything that was added or updated since ``days_ago`` days ago."""
        et = entity_type.value if isinstance(entity_type, EntityType) else entity_type
        return self._temporal.what_changed_since(days_ago=days_ago, entity_type=et)

    def when_was_last(self, topic: str) -> Optional[Dict[str, Any]]:
        """Find when the most recent node related to ``topic`` was updated."""
        return self._temporal.when_was_last(topic)

    def get_timeline(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        entity_type: Optional[EntityType] = None,
    ) -> List[Dict[str, Any]]:
        """Return all knowledge entries within the given time window."""
        et = entity_type.value if isinstance(entity_type, EntityType) else entity_type
        return self._temporal.get_timeline(start=start, end=end, entity_type=et)

    # ------------------------------------------------------------------
    # Statistics and diagnostics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics about the TKG."""
        graph_stats = self._graph.get_statistics()
        return {
            **graph_stats,
            "vector_documents": self._vector.count(),
            "storage_path": self._storage_path,
        }

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    def migrate_from_json(self, json_path: str) -> Dict[str, int]:
        """
        Migrate knowledge from a legacy learning_data.json file.

        Returns stats dict with counts of migrated records.
        """
        if not os.path.exists(json_path):
            return {"sessions": 0, "lessons": 0, "error": "File not found"}

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        sessions_count = 0
        lessons_count = 0

        for session in data.get("sessions", []):
            self.add_knowledge(
                text=session.get("summary", str(session)),
                entity_type=EntityType.SESSION,
                metadata={"source": json_path, **{k: v for k, v in session.items() if k != "summary"}},
            )
            sessions_count += 1

        for lesson in data.get("lessons", []):
            self.add_knowledge(
                text=lesson.get("text", str(lesson)),
                entity_type=EntityType.INSIGHT,
                metadata={"source": json_path, **{k: v for k, v in lesson.items() if k != "text"}},
            )
            lessons_count += 1

        return {"sessions": sessions_count, "lessons": lessons_count}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if not self._storage_path:
            return
        graph_path = os.path.join(self._storage_path, "graph.json")
        vector_path = os.path.join(self._storage_path, "vectors.json")
        with open(graph_path, "w", encoding="utf-8") as fh:
            json.dump(self._graph.to_dict(), fh, indent=2)
        with open(vector_path, "w", encoding="utf-8") as fh:
            json.dump(self._vector.to_dict(), fh, indent=2)

    def _load(self) -> None:
        if not self._storage_path:
            return
        graph_path = os.path.join(self._storage_path, "graph.json")
        vector_path = os.path.join(self._storage_path, "vectors.json")
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as fh:
                self._graph.from_dict(json.load(fh))
        if os.path.exists(vector_path):
            with open(vector_path, "r", encoding="utf-8") as fh:
                self._vector.from_dict(json.load(fh))
