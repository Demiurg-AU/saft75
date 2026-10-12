"""
Parallel strategy executor for the META-OPTIMIZER workflow engine.

Runs multiple optimization strategies concurrently using Python's
concurrent.futures.ThreadPoolExecutor and returns aggregated results.
"""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ParallelTask:
    """A single task to run in parallel."""
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


@dataclass
class ParallelResult:
    """Result of a single parallel task."""
    name: str
    result: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "result": self.result if not callable(self.result) else str(self.result),
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
            "success": self.success,
        }


class ParallelExecutor:
    """
    Executes multiple tasks concurrently and aggregates results.

    Uses a thread pool by default.  CPU-bound tasks may benefit from
    ``max_workers=1`` to avoid GIL contention with PyTorch.

    Usage::

        executor = ParallelExecutor(max_workers=4, timeout_s=120)
        tasks = [
            ParallelTask("strategy_a", pipeline.optimize, args=(model, inp), kwargs={"strategy": "fast_int8"}),
            ParallelTask("strategy_b", pipeline.optimize, args=(model, inp), kwargs={"strategy": "compact_prune"}),
        ]
        results = executor.run(tasks)
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout_s: float = 300.0,
    ) -> None:
        self.max_workers = max_workers
        self.timeout_s = timeout_s

    def run(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """
        Execute all ``tasks`` in parallel.

        Returns a list of ParallelResult in the same order as ``tasks``.
        """
        if not tasks:
            return []

        results: Dict[str, ParallelResult] = {}
        futures: Dict[concurrent.futures.Future, ParallelTask] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for task in tasks:
                future = pool.submit(self._run_task, task)
                futures[future] = task

            for future in concurrent.futures.as_completed(
                futures.keys(), timeout=self.timeout_s
            ):
                task = futures[future]
                try:
                    pr = future.result(timeout=1.0)
                except Exception as exc:
                    pr = ParallelResult(
                        name=task.name,
                        error=str(exc),
                        success=False,
                    )
                results[task.name] = pr

        # Return in original order
        return [results.get(t.name, ParallelResult(name=t.name, success=False, error="Not executed")) for t in tasks]

    def run_best_of(
        self,
        tasks: List[ParallelTask],
        score_fn: Callable[[Any], float],
    ) -> Tuple[Optional[ParallelResult], List[ParallelResult]]:
        """
        Run all tasks in parallel and return the one with the highest score.

        Args:
            tasks: Tasks to run.
            score_fn: Function that accepts a task result dict and returns a float score.

        Returns:
            (best_result, all_results) tuple.
        """
        all_results = self.run(tasks)
        best: Optional[ParallelResult] = None
        best_score = float("-inf")
        for pr in all_results:
            if not pr.success or pr.result is None:
                continue
            try:
                score = score_fn(pr.result)
                if score > best_score:
                    best_score = score
                    best = pr
            except Exception:
                pass
        return best, all_results

    @staticmethod
    def _run_task(task: ParallelTask) -> ParallelResult:
        start = time.perf_counter()
        try:
            result = task.func(*task.args, **task.kwargs)
            return ParallelResult(
                name=task.name,
                result=result,
                duration_s=time.perf_counter() - start,
            )
        except Exception as exc:
            return ParallelResult(
                name=task.name,
                error=str(exc),
                duration_s=time.perf_counter() - start,
                success=False,
            )
