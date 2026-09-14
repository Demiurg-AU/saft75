"""
LangGraph-inspired workflow engine for META-OPTIMIZER v7.0.

Implements an explicit state machine with:
- Conditional routing between nodes
- Error recovery and automatic retry
- Parallel strategy execution
- Workflow visualization (ASCII)
- State persistence/checkpointing
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .state_manager import StateManager, WorkflowState
from .error_handler import ErrorHandler, RetryConfig
from .parallel_executor import ParallelExecutor, ParallelTask


# ------------------------------------------------------------------
# Workflow node types
# ------------------------------------------------------------------

class NodeType(str, Enum):
    START = "start"
    PROCESS = "process"
    DECISION = "decision"
    END = "end"
    ERROR = "error"


class WorkflowNode:
    """A single node in the workflow graph."""

    def __init__(
        self,
        name: str,
        node_type: NodeType,
        handler: Optional[Callable] = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.node_type = node_type
        self.handler = handler
        self.description = description
        self.transitions: Dict[str, str] = {}  # condition_key → next_node_name

    def add_transition(self, condition: str, target: str) -> None:
        self.transitions[condition] = target

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.handler:
            return self.handler(context) or context
        return context


# ------------------------------------------------------------------
# Workflow engine
# ------------------------------------------------------------------

class WorkflowEngine:
    """
    Explicit state machine workflow engine.

    Nodes are registered with handlers; transitions are determined
    by string conditions returned by each handler.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        error_handler: Optional[ErrorHandler] = None,
        max_steps: int = 100,
    ) -> None:
        self._nodes: Dict[str, WorkflowNode] = {}
        self._start_node: Optional[str] = None
        self._state_manager = state_manager or StateManager()
        self._error_handler = error_handler or ErrorHandler()
        self._max_steps = max_steps

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        node_type: NodeType = NodeType.PROCESS,
        handler: Optional[Callable] = None,
        description: str = "",
    ) -> "WorkflowEngine":
        self._nodes[name] = WorkflowNode(name, node_type, handler, description)
        if node_type == NodeType.START:
            self._start_node = name
        return self

    def add_edge(self, from_node: str, to_node: str, condition: str = "default") -> "WorkflowEngine":
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes:
            raise ValueError(f"Target node '{to_node}' not found")
        self._nodes[from_node].add_transition(condition, to_node)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the workflow from the start node.

        Args:
            initial_context: Initial data passed to the first node.
            workflow_id: Optional ID for state persistence.

        Returns:
            Final context dict with a ``_workflow`` metadata key.
        """
        wid = workflow_id or str(uuid.uuid4())
        context = dict(initial_context or {})
        context["_workflow"] = {
            "id": wid,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "errors": [],
        }

        if not self._start_node:
            raise RuntimeError("No start node defined. Add a node with NodeType.START.")

        current = self._start_node
        visit_counts: Dict[str, int] = {}
        step_count = 0
        cycle_limit = max(len(self._nodes), 1)

        while current and step_count < self._max_steps:
            node = self._nodes.get(current)
            if not node:
                break

            step_count += 1
            context["_workflow"]["steps"].append({
                "node": current,
                "step": step_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Detect infinite loops: a node revisited more times than there
            # are nodes in the graph is very unlikely to be making progress.
            visit_counts[current] = visit_counts.get(current, 0) + 1
            if visit_counts[current] > cycle_limit and node.node_type != NodeType.END:
                context["_workflow"]["errors"].append(
                    f"Cycle detected at node '{current}' (visited {visit_counts[current]} times)"
                )
                break

            # Persist state checkpoint
            self._state_manager.save_state(
                WorkflowState(workflow_id=wid, state_name=current, data=dict(context))
            )

            # Execute node handler
            try:
                result = self._error_handler.run_with_retry(
                    current,
                    node.execute,
                    context,
                )
                if isinstance(result, dict):
                    context.update(result)
            except Exception as exc:
                context["_workflow"]["errors"].append(f"{current}: {exc}")
                error_node = node.transitions.get("error")
                if error_node:
                    current = error_node
                    continue
                break

            # Determine next node
            if node.node_type == NodeType.END:
                break

            condition = context.pop("_next_condition", "default")
            next_node = node.transitions.get(condition) or node.transitions.get("default")
            current = next_node

        context["_workflow"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        context["_workflow"]["total_steps"] = step_count
        return context

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_workflow(self) -> str:
        """Return a simple ASCII representation of the workflow graph."""
        lines = ["META-OPTIMIZER Workflow Graph", "=" * 40]
        for name, node in self._nodes.items():
            prefix = {
                NodeType.START: "▶",
                NodeType.END: "■",
                NodeType.ERROR: "✕",
                NodeType.DECISION: "◆",
                NodeType.PROCESS: "●",
            }.get(node.node_type, "○")
            desc = f" — {node.description}" if node.description else ""
            lines.append(f"  {prefix} [{name}]{desc}")
            for cond, target in node.transitions.items():
                lines.append(f"      └─[{cond}]──▶ {target}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Pre-built META-OPTIMIZER workflow
# ------------------------------------------------------------------

class MetaOptimizerWorkflow:
    """
    High-level workflow for META-OPTIMIZER optimization tasks.

    Combines the workflow engine with the optimization pipeline and
    TKG memory for a complete end-to-end orchestration system.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_workers: int = 4,
    ) -> None:
        state_mgr = StateManager(storage_dir=storage_dir)
        error_hdlr = ErrorHandler(RetryConfig(max_attempts=3))
        self._engine = WorkflowEngine(
            state_manager=state_mgr,
            error_handler=error_hdlr,
        )
        self._executor = ParallelExecutor(max_workers=max_workers)
        self._build_workflow()

    def _build_workflow(self) -> None:
        """Construct the default META-OPTIMIZER workflow graph."""
        engine = self._engine

        engine.add_node("start", NodeType.START, self._step_init, "Initialize request")
        engine.add_node("analyze", NodeType.PROCESS, self._step_analyze, "Analyze request type")
        engine.add_node("route", NodeType.DECISION, self._step_route, "Route to appropriate handler")
        engine.add_node("optimize_model", NodeType.PROCESS, self._step_optimize, "Run optimization pipeline")
        engine.add_node("store_memory", NodeType.PROCESS, self._step_store, "Store results in TKG")
        engine.add_node("error_recovery", NodeType.ERROR, self._step_recover, "Error recovery")
        engine.add_node("end", NodeType.END, self._step_finalize, "Finalize and return results")

        engine.add_edge("start", "analyze")
        engine.add_edge("analyze", "route")
        engine.add_edge("route", "optimize_model", "optimize_model")
        engine.add_edge("route", "store_memory", "store_memory")
        engine.add_edge("route", "end", "default")
        engine.add_edge("optimize_model", "store_memory")
        engine.add_edge("optimize_model", "error_recovery", "error")
        engine.add_edge("store_memory", "end")
        engine.add_edge("error_recovery", "end")

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _step_init(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["status"] = "running"
        context["results"] = {}
        return context

    def _step_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        request = context.get("request", "")
        request_type = context.get("request_type", "")
        if not request_type:
            lower = request.lower()
            if any(kw in lower for kw in ("optim", "quantiz", "prun", "compress")):
                request_type = "optimize_model"
            elif any(kw in lower for kw in ("remember", "store", "learn", "memorize")):
                request_type = "store_memory"
            else:
                request_type = "generic"
        context["resolved_request_type"] = request_type
        return context

    def _step_route(self, context: Dict[str, Any]) -> Dict[str, Any]:
        rtype = context.get("resolved_request_type", "generic")
        context["_next_condition"] = rtype
        return context

    def _step_optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context.get("model")
        test_input = context.get("test_input")
        strategies = context.get("strategies", ["hybrid_moderate"])
        use_parallel = context.get("use_parallel", False)

        if model is None or test_input is None:
            context["optimization_results"] = {"error": "No model or test_input provided"}
            return context

        try:
            # Import here to avoid circular imports at module level
            from ..amazon_robotics import OptimizationPipeline
            pipeline = OptimizationPipeline()

            if use_parallel and len(strategies) > 1:
                tasks = [
                    ParallelTask(
                        name=s,
                        func=pipeline.optimize,
                        args=(model, test_input),
                        kwargs={"strategy": s},
                    )
                    for s in strategies
                ]
                results = self._executor.run(tasks)
                context["optimization_results"] = {
                    r.name: r.result if r.success else {"error": r.error}
                    for r in results
                }
            else:
                context["optimization_results"] = {
                    s: pipeline.optimize(model, test_input, strategy=s)
                    for s in strategies
                }
        except Exception as exc:
            context["optimization_results"] = {"error": str(exc)}
            context["_next_condition"] = "error"

        return context

    def _step_store(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ..temporal_memory import TemporalKnowledgeGraph, EntityType
            tkg = TemporalKnowledgeGraph()
            opt_results = context.get("optimization_results", {})
            if opt_results:
                tkg.add_knowledge(
                    text=f"Optimization run: {context.get('request', 'unknown')}",
                    entity_type=EntityType.OPTIMIZATION,
                    metadata={"results_summary": str(opt_results)[:500]},
                )
            context["memory_stored"] = True
        except Exception as exc:
            context["memory_stored"] = False
            context["memory_error"] = str(exc)
        return context

    def _step_recover(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["recovered"] = True
        context["status"] = "degraded"
        return context

    def _step_finalize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        errors = context.get("_workflow", {}).get("errors", [])
        context["success"] = len(errors) == 0
        return context

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        request: str,
        request_type: Optional[str] = None,
        model: Any = None,
        test_input: Any = None,
        max_retries: int = 3,
        use_parallel: bool = False,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the META-OPTIMIZER workflow.

        Args:
            request: Natural language description of the task.
            request_type: Override automatic routing ("optimize_model" | "store_memory").
            model: PyTorch model to optimize (required for optimization tasks).
            test_input: Test input tensor.
            max_retries: Maximum retry attempts per step.
            use_parallel: Run multiple strategies in parallel.
            strategies: Optimization strategies to apply.

        Returns:
            Final workflow context dict with results.
        """
        self._engine._error_handler.config.max_attempts = max_retries
        initial = {
            "request": request,
            "request_type": request_type or "",
            "model": model,
            "test_input": test_input,
            "use_parallel": use_parallel,
            "strategies": strategies or ["hybrid_moderate"],
        }
        return self._engine.run(initial_context=initial)

    def visualize_workflow(self) -> str:
        return self._engine.visualize_workflow()
