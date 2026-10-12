"""
Hybrid retriever that combines graph-based and vector-based search
for the Temporal Knowledge Graph.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .graph_store import GraphNode, NetworkXGraphStore
from .vector_store import VectorDocument


class HybridResult:
    """Combined result from graph + vector retrieval."""

    def __init__(
        self,
        node: Optional[GraphNode],
        document: Optional[VectorDocument],
        graph_score: float = 0.0,
        vector_score: float = 0.0,
        alpha: float = 0.5,
    ) -> None:
        self.node = node
        self.document = document
        self.graph_score = graph_score
        self.vector_score = vector_score
        self.alpha = alpha

    @property
    def combined_score(self) -> float:
        """Weighted combination of graph and vector scores."""
        return self.alpha * self.vector_score + (1.0 - self.alpha) * self.graph_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node.node_id if self.node else None,
            "doc_id": self.document.doc_id if self.document else None,
            "label": self.node.label if self.node else (self.document.text[:80] if self.document else ""),
            "entity_type": self.node.entity_type if self.node else None,
            "text": self.document.text if self.document else (self.node.label if self.node else ""),
            "metadata": self.document.metadata if self.document else (self.node.properties if self.node else {}),
            "graph_score": self.graph_score,
            "vector_score": self.vector_score,
            "combined_score": self.combined_score,
            "created_at": (
                self.node.created_at.isoformat() if self.node
                else (self.document.created_at.isoformat() if self.document else None)
            ),
        }


class HybridRetriever:
    """
    Retrieves knowledge using a combination of graph traversal and
    vector similarity search.

    The ``alpha`` parameter controls the blend:
    - alpha=1.0  → pure vector search
    - alpha=0.0  → pure graph search
    - alpha=0.5  → equal weight (default)
    """

    def __init__(
        self,
        graph_store: NetworkXGraphStore,
        vector_store: Any,
        alpha: float = 0.5,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[HybridResult]:
        """
        Retrieve the most relevant knowledge entries for ``query``.

        Graph candidates are found by label substring match; vector
        candidates by embedding cosine similarity. Both result sets are
        merged and re-ranked by the combined score.
        """
        # --- Vector search ---
        vector_results: Dict[str, Tuple[VectorDocument, float]] = {}
        try:
            for doc, score in self._vector.search(query, top_k=top_k * 2):
                vector_results[doc.doc_id] = (doc, score)
        except Exception:
            pass

        # --- Graph search ---
        graph_results: Dict[str, Tuple[GraphNode, float]] = {}
        try:
            nodes = self._graph.find_nodes(
                entity_type=entity_type,
                label_contains=query,
                since=since,
            )
            for node in nodes:
                # Simple relevance: 1.0 if exact match, lower otherwise
                score = 1.0 if query.lower() in node.label.lower() else 0.5
                graph_results[node.node_id] = (node, score)
        except Exception:
            pass

        # --- Merge results ---
        # Build a combined view keyed by node_id (matched via doc metadata)
        merged: Dict[str, HybridResult] = {}

        for doc_id, (doc, vscore) in vector_results.items():
            node_id = doc.metadata.get("node_id")
            if node_id and node_id in graph_results:
                node, gscore = graph_results[node_id]
                merged[node_id] = HybridResult(
                    node=node,
                    document=doc,
                    graph_score=gscore,
                    vector_score=vscore,
                    alpha=self.alpha,
                )
            else:
                # Vector-only result
                node = self._graph.get_node(node_id) if node_id else None
                merged[doc_id] = HybridResult(
                    node=node,
                    document=doc,
                    graph_score=0.0,
                    vector_score=vscore,
                    alpha=self.alpha,
                )

        for node_id, (node, gscore) in graph_results.items():
            if node_id not in merged:
                merged[node_id] = HybridResult(
                    node=node,
                    document=None,
                    graph_score=gscore,
                    vector_score=0.0,
                    alpha=self.alpha,
                )

        # Apply temporal filter
        if since:
            merged = {
                k: v for k, v in merged.items()
                if _result_timestamp(v) >= since
            }

        # Apply entity_type filter
        if entity_type:
            merged = {
                k: v for k, v in merged.items()
                if v.node and v.node.entity_type == entity_type
            }

        sorted_results = sorted(merged.values(), key=lambda r: r.combined_score, reverse=True)
        return sorted_results[:top_k]

    def retrieve_neighbors(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
        depth: int = 1,
    ) -> List[GraphNode]:
        """Walk the graph from ``node_id`` up to ``depth`` hops."""
        visited: set = set()
        frontier = [node_id]
        results: List[GraphNode] = []

        for _ in range(depth):
            next_frontier: List[str] = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                neighbors = self._graph.get_neighbors(nid, relation_type=relation_type)
                for neighbor in neighbors:
                    if neighbor.node_id not in visited:
                        results.append(neighbor)
                        next_frontier.append(neighbor.node_id)
            frontier = next_frontier

        return results


def _result_timestamp(result: HybridResult) -> datetime:
    if result.node:
        return result.node.created_at
    if result.document:
        return result.document.created_at
    return datetime.min.replace(tzinfo=timezone.utc)
