"""
Error recovery strategies for the META-OPTIMIZER workflow engine.

Provides automatic retry logic, exponential back-off, and graceful
degradation with configurable fallback strategies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RetryConfig:
    """Configuration for retry/back-off behaviour."""
    max_attempts: int = 3
    initial_delay_s: float = 1.0
    backoff_factor: float = 2.0
    max_delay_s: float = 30.0
    retryable_exceptions: tuple = (Exception,)


@dataclass
class ErrorRecord:
    """Records a single error event during workflow execution."""
    step: str
    attempt: int
    error_type: str
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "attempt": self.attempt,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class ErrorHandler:
    """
    Handles errors during workflow execution with configurable retry
    logic and fallback strategies.

    Usage::

        handler = ErrorHandler(RetryConfig(max_attempts=3))
        result = handler.run_with_retry("quantize", my_func, arg1, arg2)
    """

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig()
        self._errors: List[ErrorRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_with_retry(
        self,
        step_name: str,
        func: Callable,
        *args: Any,
        fallback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute ``func(*args, **kwargs)`` with automatic retry on failure.

        Args:
            step_name: Human-readable name for logging/error records.
            func: Callable to execute.
            *args: Positional arguments for ``func``.
            fallback: Optional callable invoked if all retries fail.
            **kwargs: Keyword arguments for ``func``.

        Returns:
            Result of ``func`` or ``fallback`` if provided.

        Raises:
            The last caught exception if no fallback is configured.
        """
        last_exc: Optional[Exception] = None
        delay = self.config.initial_delay_s

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.config.retryable_exceptions as exc:
                last_exc = exc
                record = ErrorRecord(
                    step=step_name,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                self._errors.append(record)

                if attempt < self.config.max_attempts:
                    time.sleep(min(delay, self.config.max_delay_s))
                    delay *= self.config.backoff_factor

        # All attempts exhausted
        if fallback is not None:
            try:
                return fallback(*args, **kwargs)
            except Exception as fb_exc:
                record = ErrorRecord(
                    step=f"{step_name}_fallback",
                    attempt=0,
                    error_type=type(fb_exc).__name__,
                    message=str(fb_exc),
                )
                self._errors.append(record)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"All {self.config.max_attempts} attempts failed for step '{step_name}'")

    def run_safe(
        self,
        step_name: str,
        func: Callable,
        *args: Any,
        default: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute ``func`` and return ``default`` on any exception.
        Never raises.
        """
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self._errors.append(ErrorRecord(
                step=step_name,
                attempt=1,
                error_type=type(exc).__name__,
                message=str(exc),
            ))
            return default

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_errors(self) -> List[Dict[str, Any]]:
        """Return all recorded errors as a list of dicts."""
        return [e.to_dict() for e in self._errors]

    def clear_errors(self) -> None:
        """Reset the error log."""
        self._errors.clear()

    def has_errors(self) -> bool:
        return len(self._errors) > 0

    def error_summary(self) -> Dict[str, Any]:
        """Return a brief summary of errors grouped by step."""
        by_step: Dict[str, int] = {}
        for e in self._errors:
            by_step[e.step] = by_step.get(e.step, 0) + 1
        return {
            "total_errors": len(self._errors),
            "errors_by_step": by_step,
        }
