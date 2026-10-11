"""
Structured neural network pruning via real channel/filter removal.

Implements:
- Structured L2-norm channel pruning for ``nn.Sequential`` chains of
  ``nn.Linear``/``nn.Conv2d`` — rebuilds each layer with genuinely fewer
  output channels/features (and trims the next layer's input dimension
  to match), rather than masking weights to zero in place.

Unstructured pruning (``torch.nn.utils.prune``'s usual approach: zero out
individual weights, keep the tensor's shape unchanged) is intentionally
not implemented. Verified empirically (see ``tests/test_smoke.py``): on
CPU, converting a masked dense tensor to ``torch.sparse_coo``/
``sparse_csr`` only uses less memory than the dense tensor once sparsity
exceeds roughly 80% (COO) or 70% (CSR) — the per-nonzero index overhead
dominates below that. None of this library's predefined strategies prune
anywhere near that threshold (0.2/0.3/0.5), so there is no CPU-beneficial
way to implement "real" unstructured pruning for the amounts this library
actually offers; ``_unstructured_prune`` raises ``NotImplementedError``
rather than silently reporting a compression number that storage doesn't
back up.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    _PRUNABLE_TYPES = (nn.Linear, nn.Conv2d)
    _TRANSPARENT_TYPES = (
        nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU,
        nn.Sigmoid, nn.Tanh, nn.Dropout, nn.Identity, nn.Flatten,
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    )


@dataclass
class PruningConfig:
    """Configuration for a pruning pass."""
    method: str = "structured_l2"      # structured_l2 | unstructured_l1 | unstructured_random
    amount: float = 0.3                # Fraction of output channels/connections to prune (0-1)


@dataclass
class PruningResult:
    """Holds the pruned model and associated metrics."""
    original_model: Any
    pruned_model: Any
    method: str
    amount: float
    original_params: int
    remaining_params: int
    original_latency_ms: float
    pruned_latency_ms: float
    original_size_mb: float
    pruned_size_mb: float
    success: bool = True
    error: Optional[str] = None

    @property
    def sparsity(self) -> float:
        if self.original_params == 0:
            return 0.0
        return 1.0 - self.remaining_params / self.original_params

    @property
    def latency_improvement(self) -> float:
        if self.pruned_latency_ms == 0:
            return 0.0
        return self.original_latency_ms / self.pruned_latency_ms

    @property
    def size_reduction(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return 1.0 - self.pruned_size_mb / self.original_size_mb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "amount": self.amount,
            "original_params": self.original_params,
            "remaining_params": self.remaining_params,
            "sparsity": round(self.sparsity, 4),
            "original_latency_ms": round(self.original_latency_ms, 3),
            "pruned_latency_ms": round(self.pruned_latency_ms, 3),
            "latency_improvement": round(self.latency_improvement, 2),
            "original_size_mb": round(self.original_size_mb, 3),
            "pruned_size_mb": round(self.pruned_size_mb, 3),
            "size_reduction": round(self.size_reduction, 3),
            "success": self.success,
            "error": self.error,
        }


class RealPruner:
    """
    Structured neural network pruner with real channel removal.

    Only ``nn.Sequential`` chains of ``nn.Linear``/``nn.Conv2d`` (with
    activations, ``Flatten``, pooling, or ``Dropout`` in between) are
    supported — see the module docstring for why, and why unstructured
    pruning is not offered at all.
    """

    def __init__(self, config: Optional[PruningConfig] = None) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")
        self.config = config or PruningConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        method: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> PruningResult:
        """
        Prune ``model`` according to the configured strategy.

        Args:
            model: PyTorch module to prune.
            test_input: Representative input for latency benchmarking.
            method: Override config method.
            amount: Override config amount.

        Returns:
            PruningResult with the pruned model and metrics.
        """
        method = method or self.config.method
        amount = amount if amount is not None else self.config.amount
        original_params = _count_params(model)
        original_size = _model_size_mb(model)
        original_latency = _benchmark_latency(model, test_input)

        try:
            import copy
            m = copy.deepcopy(model)
            m.eval()

            if method == "structured_l2":
                m = self._structured_prune(m, amount)
            elif method in ("unstructured_l1", "unstructured_random"):
                self._unstructured_prune(amount)
            else:
                raise ValueError(f"Unknown pruning method: {method}")

            pruned_params = _count_params(m)
            pruned_size = _model_size_mb(m)
            pruned_latency = _benchmark_latency(m, test_input)

            return PruningResult(
                original_model=model,
                pruned_model=m,
                method=method,
                amount=amount,
                original_params=original_params,
                remaining_params=pruned_params,
                original_latency_ms=original_latency,
                pruned_latency_ms=pruned_latency,
                original_size_mb=original_size,
                pruned_size_mb=pruned_size,
            )
        except Exception as exc:
            return PruningResult(
                original_model=model,
                pruned_model=model,
                method=method,
                amount=amount,
                original_params=original_params,
                remaining_params=original_params,
                original_latency_ms=original_latency,
                pruned_latency_ms=original_latency,
                original_size_mb=original_size,
                pruned_size_mb=original_size,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    def _unstructured_prune(self, amount: float) -> None:
        """Always raises — see module docstring for the CPU-sparsity tradeoff."""
        raise NotImplementedError(
            f"unstructured pruning (amount={amount}) has no CPU-beneficial "
            "implementation: converting a masked dense tensor to a sparse "
            "format only uses less memory than the dense tensor above "
            "~70-80% sparsity, and no predefined strategy in this library "
            "prunes that aggressively. Use method='structured_l2' instead."
        )

    def _structured_prune(self, model: "nn.Module", amount: float) -> "nn.Module":
        """
        Remove the ``amount`` fraction of output channels/features from
        each non-final ``nn.Linear``/``nn.Conv2d`` layer in ``model``,
        rebuilding it (and trimming the next prunable layer's input
        dimension to match) rather than masking weights to zero.

        The final prunable layer's output dimension is never touched —
        it defines the model's output contract. Any child module that
        isn't a prunable or recognized transparent type (notably any
        normalization layer, which has per-channel parameters that would
        also need trimming) raises ``NotImplementedError`` rather than
        silently skipping it.
        """
        if not isinstance(model, nn.Sequential):
            raise NotImplementedError(
                f"structured pruning requires nn.Sequential; got {type(model).__name__}"
            )

        children = list(model)
        for child in children:
            if not isinstance(child, _PRUNABLE_TYPES + _TRANSPARENT_TYPES):
                raise NotImplementedError(
                    "structured pruning requires nn.Sequential of "
                    f"{[t.__name__ for t in _PRUNABLE_TYPES]} (with only "
                    f"{[t.__name__ for t in _TRANSPARENT_TYPES]} in between); "
                    f"got unsupported layer type {type(child).__name__}"
                )

        prunable_indices = [i for i, c in enumerate(children) if isinstance(c, _PRUNABLE_TYPES)]
        if not prunable_indices:
            return model

        carry_keep_idx: Optional["torch.Tensor"] = None
        carry_block_size = 1

        for pos, idx in enumerate(prunable_indices):
            layer = children[idx]
            is_last = pos == len(prunable_indices) - 1

            if carry_keep_idx is None:
                in_keep = None
            elif carry_block_size == 1:
                in_keep = carry_keep_idx
            else:
                in_keep = torch.cat([
                    torch.arange(c * carry_block_size, (c + 1) * carry_block_size)
                    for c in carry_keep_idx.tolist()
                ])

            if isinstance(layer, nn.Linear):
                new_layer, out_keep = _rebuild_linear(layer, amount, in_keep, is_last)
                carry_block_size = 1
            else:
                if layer.groups != 1:
                    raise NotImplementedError(
                        "structured pruning does not support grouped/depthwise "
                        f"Conv2d (groups={layer.groups})"
                    )
                new_layer, out_keep = _rebuild_conv2d(layer, amount, in_keep, is_last)
                next_idx = prunable_indices[pos + 1] if not is_last else None
                if next_idx is not None and isinstance(children[next_idx], nn.Linear):
                    next_in_features = children[next_idx].in_features
                    if next_in_features % layer.out_channels != 0:
                        raise NotImplementedError(
                            "cannot map Conv2d output channels onto the next "
                            f"Linear's in_features={next_in_features} (not a "
                            f"multiple of out_channels={layer.out_channels}); "
                            "unsupported reshape between Conv2d and Linear"
                        )
                    carry_block_size = next_in_features // layer.out_channels
                else:
                    carry_block_size = 1

            model[idx] = new_layer
            carry_keep_idx = out_keep

        return model


def _rebuild_linear(layer: "nn.Linear", amount: float, in_keep, is_last: bool):
    W = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None
    if in_keep is not None:
        W = W[:, in_keep]
    out_keep = _select_keep(W.reshape(W.shape[0], -1), amount, is_last)
    new_layer = nn.Linear(W.shape[1], len(out_keep), bias=b is not None)
    new_layer.weight.data = W[out_keep].clone()
    if b is not None:
        new_layer.bias.data = b[out_keep].clone()
    return new_layer, out_keep


def _rebuild_conv2d(layer: "nn.Conv2d", amount: float, in_keep, is_last: bool):
    W = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None
    if in_keep is not None:
        W = W[:, in_keep]
    out_keep = _select_keep(W.reshape(W.shape[0], -1), amount, is_last)
    new_layer = nn.Conv2d(
        W.shape[1], len(out_keep),
        kernel_size=layer.kernel_size, stride=layer.stride,
        padding=layer.padding, dilation=layer.dilation,
        groups=1, bias=b is not None,
    )
    new_layer.weight.data = W[out_keep].clone()
    if b is not None:
        new_layer.bias.data = b[out_keep].clone()
    return new_layer, out_keep


def _select_keep(weight_2d: "torch.Tensor", amount: float, is_last: bool) -> "torch.Tensor":
    """Return sorted indices of output rows to keep, by L2 norm."""
    n_out = weight_2d.shape[0]
    if is_last:
        return torch.arange(n_out)
    norms = weight_2d.norm(p=2, dim=1)
    n_remove = int(n_out * amount)
    n_keep = max(1, n_out - n_remove)
    return norms.argsort(descending=True)[:n_keep].sort().values


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _count_params(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


def _model_size_mb(model: "nn.Module") -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def _benchmark_latency(
    model: "nn.Module",
    test_input: "torch.Tensor",
    warmup: int = 5,
    runs: int = 20,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(test_input)
        start = time.perf_counter()
        for _ in range(runs):
            model(test_input)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0
