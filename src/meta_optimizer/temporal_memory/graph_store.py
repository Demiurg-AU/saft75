"""
Graph storage backend for the Temporal Knowledge Graph.
Supports NetworkX (in-memory) and Neo4j (production) backends.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class GraphNode:
    """Represents a node in the knowledge graph."""

    def __init__(
        self,
        node_id: str,
        entity_type: str,
        label: str,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> None:
        self.node_id = node_id
        self.entity_type = entity_type
        self.label = label
        self.properties = properties or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entity_type": self.entity_type,
            "label": self.label,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        return cls(
            node_id=data["node_id"],
            entity_type=data["entity_type"],
            label=data["label"],
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class GraphEdge:
    """Represents a directed edge (relationship) in the knowledge graph."""

    def __init__(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
    ) -> None:
        self.edge_id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.weight = weight
        self.properties = properties or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.valid_until = valid_until

    @property
    def is_valid(self) -> bool:
        if self.valid_until is None:
            return True
        return datetime.now(timezone.utc) < self.valid_until

    def invalidate(self) -> None:
        self.valid_until = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdge":
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
        )


class NetworkXGraphStore:
    """
    In-memory graph store backed by NetworkX.
    Suitable for development and single-session use.
    """

    def __init__(self) -> None:
        if not HAS_NETWORKX:
            raise ImportError("networkx is required. Install with: pip install networkx")
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}

    # --- Node operations ---

    def add_node(self, node: GraphNode) -> str:
        self._nodes[node.node_id] = node
        self._graph.add_node(
            node.node_id,
            entity_type=node.entity_type,
            label=node.label,
            **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in node.properties.items()},
        )
        return node.node_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def find_nodes(
        self,
        entity_type: Optional[str] = None,
        label_contains: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[GraphNode]:
        results = list(self._nodes.values())
        if entity_type:
            results = [n for n in results if n.entity_type == entity_type]
        if label_contains:
            lower = label_contains.lower()
            results = [n for n in results if lower in n.label.lower()]
        if since:
            results = [n for n in results if n.created_at >= since]
        return results

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        node = self._nodes.get(node_id)
        if not node:
            return False
        node.properties.update(properties)
        node.updated_at = datetime.now(timezone.utc)
        return True

    # --- Edge operations ---

    def add_edge(self, edge: GraphEdge) -> str:
        self._edges[edge.edge_id] = edge
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_id=edge.edge_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
        )
        return edge.edge_id

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self._edges.get(edge_id)

    def get_edges_between(self, source_id: str, target_id: str) -> List[GraphEdge]:
        return [
            e for e in self._edges.values()
            if e.source_id == source_id and e.target_id == target_id
        ]

    def invalidate_edge(self, edge_id: str) -> bool:
        edge = self._edges.get(edge_id)
        if not edge:
            return False
        edge.invalidate()
        return True

    def get_neighbors(self, node_id: str, relation_type: Optional[str] = None) -> List[GraphNode]:
        if node_id not in self._graph:
            return []
        neighbor_ids = list(self._graph.successors(node_id))
        neighbors = [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]
        if relation_type:
            valid_edges = {
                e.target_id for e in self._edges.values()
                if e.source_id == node_id and e.relation_type == relation_type and e.is_valid
            }
            neighbors = [n for n in neighbors if n.node_id in valid_edges]
        return neighbors

    # --- Statistics ---

    def get_statistics(self) -> Dict[str, Any]:
        valid_edges = [e for e in self._edges.values() if e.is_valid]
        entity_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            entity_counts[node.entity_type] = entity_counts.get(node.entity_type, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "valid_edges": len(valid_edges),
            "entity_type_distribution": entity_counts,
        }

    # --- Persistence ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        for node_data in data.get("nodes", []):
            node = GraphNode.from_dict(node_data)
            self.add_node(node)
        for edge_data in data.get("edges", []):
            edge = GraphEdge.from_dict(edge_data)
            self.add_edge(edge)


def create_node(
    entity_type: str,
    label: str,
    properties: Optional[Dict[str, Any]] = None,
) -> GraphNode:
    """Helper to create a GraphNode with a generated UUID."""
    return GraphNode(
        node_id=str(uuid.uuid4()),
        entity_type=entity_type,
        label=label,
        properties=properties,
    )


def create_edge(
    source_id: str,
    target_id: str,
    relation_type: str,
    weight: float = 1.0,
    properties: Optional[Dict[str, Any]] = None,
) -> GraphEdge:
    """Helper to create a GraphEdge with a generated UUID."""
    return GraphEdge(
        edge_id=str(uuid.uuid4()),
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=weight,
        properties=properties,
    )
