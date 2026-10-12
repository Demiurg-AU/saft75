"""
Time-aware query helpers for the Temporal Knowledge Graph.
Provides natural language-style temporal queries such as:
- what_changed_since(days_ago=7)
- when_was_last(topic)
- get_history(entity_id)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .graph_store import GraphNode, NetworkXGraphStore


class TemporalQuery:
    """
    Provides temporal query capabilities over a graph store.

    All timestamps are stored and compared in UTC.
    """

    def __init__(self, graph_store: NetworkXGraphStore) -> None:
        self._graph = graph_store

    def what_changed_since(
        self,
        days_ago: int = 7,
        entity_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all nodes created or updated since ``days_ago`` days ago.

        Returns a list of change records sorted by updated_at descending.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days_ago)
        nodes = self._graph.find_nodes(entity_type=entity_type, since=since)
        records = []
        for node in nodes:
            records.append({
                "node_id": node.node_id,
                "entity_type": node.entity_type,
                "label": node.label,
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
                "days_ago": (datetime.now(timezone.utc) - node.updated_at).days,
                "properties": node.properties,
            })
        records.sort(key=lambda r: r["updated_at"], reverse=True)
        return records

    def when_was_last(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Find the most recent node whose label contains ``topic``.

        Returns a dict with timestamp info, or None if not found.
        """
        nodes = self._graph.find_nodes(label_contains=topic)
        if not nodes:
            return None
        most_recent = max(nodes, key=lambda n: n.updated_at)
        delta = datetime.now(timezone.utc) - most_recent.updated_at
        return {
            "node_id": most_recent.node_id,
            "label": most_recent.label,
            "entity_type": most_recent.entity_type,
            "last_updated": most_recent.updated_at.isoformat(),
            "days_ago": delta.days,
            "hours_ago": int(delta.total_seconds() // 3600),
            "properties": most_recent.properties,
        }

    def get_history(
        self,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Return the most recently created/updated nodes, optionally
        filtered by ``entity_type``.
        """
        nodes = self._graph.find_nodes(entity_type=entity_type)
        nodes.sort(key=lambda n: n.updated_at, reverse=True)
        return [
            {
                "node_id": n.node_id,
                "entity_type": n.entity_type,
                "label": n.label,
                "created_at": n.created_at.isoformat(),
                "updated_at": n.updated_at.isoformat(),
            }
            for n in nodes[:limit]
        ]

    def get_timeline(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        entity_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all nodes created within an optional time window,
        sorted chronologically.
        """
        nodes = self._graph.find_nodes(entity_type=entity_type, since=start)
        if end:
            nodes = [n for n in nodes if n.created_at <= end]
        nodes.sort(key=lambda n: n.created_at)
        return [
            {
                "node_id": n.node_id,
                "entity_type": n.entity_type,
                "label": n.label,
                "created_at": n.created_at.isoformat(),
            }
            for n in nodes
        ]

    def find_related_in_timeframe(
        self,
        topic: str,
        days_ago: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Find nodes related to ``topic`` that were created or updated
        within the last ``days_ago`` days.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days_ago)
        nodes = self._graph.find_nodes(label_contains=topic, since=since)
        return [
            {
                "node_id": n.node_id,
                "entity_type": n.entity_type,
                "label": n.label,
                "created_at": n.created_at.isoformat(),
                "updated_at": n.updated_at.isoformat(),
                "relevance": "high" if topic.lower() in n.label.lower() else "medium",
            }
            for n in nodes
        ]
