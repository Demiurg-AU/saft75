"""
State persistence for the META-OPTIMIZER workflow engine.

Stores workflow states as JSON files on disk, enabling pause/resume
and crash recovery for long-running optimization workflows.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class WorkflowState:
    """Immutable snapshot of a workflow's state at a given point in time."""

    def __init__(
        self,
        workflow_id: str,
        state_name: str,
        data: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> None:
        self.workflow_id = workflow_id
        self.state_name = state_name
        self.data = data or {}
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "state_name": self.state_name,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkflowState":
        return cls(
            workflow_id=d["workflow_id"],
            state_name=d["state_name"],
            data=d.get("data", {}),
            created_at=datetime.fromisoformat(d["created_at"]),
        )


class StateManager:
    """
    Manages workflow state persistence.

    Supports both in-memory (ephemeral) and file-based (persistent) storage.
    """

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        """
        Args:
            storage_dir: Directory for on-disk state storage.
                         If None, states are kept in-memory only.
        """
        self._storage_dir = storage_dir
        self._states: Dict[str, List[WorkflowState]] = {}
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            self._load_all()

    # ------------------------------------------------------------------
    # State CRUD
    # ------------------------------------------------------------------

    def save_state(self, state: WorkflowState) -> None:
        """Append ``state`` to the history for its workflow."""
        wid = state.workflow_id
        if wid not in self._states:
            self._states[wid] = []
        self._states[wid].append(state)
        if self._storage_dir:
            self._persist(wid)

    def get_latest_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Return the most recently saved state for ``workflow_id``."""
        history = self._states.get(workflow_id, [])
        return history[-1] if history else None

    def get_state_history(self, workflow_id: str) -> List[WorkflowState]:
        """Return the full state history for ``workflow_id``."""
        return list(self._states.get(workflow_id, []))

    def list_workflows(self) -> List[str]:
        """Return all known workflow IDs."""
        return list(self._states.keys())

    def delete_workflow(self, workflow_id: str) -> bool:
        """Remove all persisted state for ``workflow_id``."""
        if workflow_id not in self._states:
            return False
        del self._states[workflow_id]
        if self._storage_dir:
            path = self._state_path(workflow_id)
            if os.path.exists(path):
                os.remove(path)
        return True

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _state_path(self, workflow_id: str) -> str:
        assert self._storage_dir is not None
        safe_id = workflow_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._storage_dir, f"{safe_id}.json")

    def _persist(self, workflow_id: str) -> None:
        path = self._state_path(workflow_id)
        data = [s.to_dict() for s in self._states[workflow_id]]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_all(self) -> None:
        assert self._storage_dir is not None
        for filename in os.listdir(self._storage_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self._storage_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    records = json.load(fh)
                for record in records:
                    state = WorkflowState.from_dict(record)
                    wid = state.workflow_id
                    if wid not in self._states:
                        self._states[wid] = []
                    self._states[wid].append(state)
            except (json.JSONDecodeError, KeyError):
                pass  # Skip corrupted state files
