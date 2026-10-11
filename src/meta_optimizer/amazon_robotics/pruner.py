"""
Structured neural network pruning via real channel/filter removal.

Implements:
- Structured L2-norm channel pruning that rebuilds each prunable layer with
  genuinely fewer output channels/features (and trims the next layer's
  input dimension to match), rather than masking weights to zero in place
  (``torch.nn.utils.prune`` only zeroes weights; it never changes tensor
  shape, so size/latency never move regardless of amount).

Pruning is applied per "island": a ``nn.Sequential`` chain (found anywhere
in the model's module tree, possibly nested, e.g. inside a custom block
like torchvision's ``InvertedResidual``) whose flattened content consists
only of ``nn.Linear``/``nn.Conv2d``, ``nn.BatchNorm1d``/``nn.BatchNorm2d``
(co-trimmed in lockstep with the conv/linear immediately before them), a
depthwise ``nn.Conv2d`` (``groups == in_channels == out_channels`` — a
pass-through layer whose own channel count is dictated by whatever
upstream decided, never independently chosen), and plain activation/
pooling/dropout/flatten layers. Each island is pruned independently and
never at its own boundary (first layer's input, last layer's output are
untouched) — islands are never merged across a non-Sequential boundary
(e.g. a residual add hidden in a custom ``forward()``, as in MobileNetV2's
``InvertedResidual``), so a block's own input/output channel count, and
anything a residual connection depends on, is never touched; only a
block's *internal* channels (e.g. the expanded hidden dimension inside an
inverted residual) are ever pruned. A model with no supported island
anywhere raises ``NotImplementedError`` rather than silently doing
nothing.

Unstructured pruning (``torch.nn.utils.prune``'s usual approach: zero out
individual weights, keep the tensor's shape unchanged) is not implemented.
Verified empirically (see ``tests/test_smoke.py``): on CPU, converting a
masked dense tensor to ``torch.sparse_coo``/``sparse_csr`` only uses less
memory than the dense tensor once sparsity exceeds roughly 80% (COO) or
70% (CSR) — the per-nonzero index overhead dominates below that. None of
this library's predefined strategies prune anywhere near that threshold,
so there is no CPU-beneficial way to implement unstructured pruning for
the amounts this library actually offers.
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
    _BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
    _TRANSPARENT_TYPES = (
        nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU,
        nn.Sigmoid, nn.Tanh, nn.Dropout, nn.Identity, nn.Flatten,
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    )


@dataclass
class PruningConfig:
    """Configuration for a pruning pass."""
    method: str = "structured_l2"      # structured_l2 only — see module docstring
    amount: float = 0.3                # Fraction of output channels/features to prune (0-1)


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

    See the module docstring for exactly what's supported (``nn.Sequential``
    "islands" of Linear/Conv2d/BatchNorm/depthwise-conv/activations, found
    anywhere in the model tree) and why unstructured pruning isn't offered.
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
    # Internal implementation
    # ------------------------------------------------------------------

    def _structured_prune(self, model: "nn.Module", amount: float) -> "nn.Module":
        """
        Find every prunable "island" (see module docstring) in ``model``
        and physically rebuild it with ``amount`` fewer output channels on
        each non-final prunable layer. Raises ``NotImplementedError`` if
        ``model`` contains no supported island at all.
        """
        if isinstance(model, nn.Sequential) and _is_prunable_island(model):
            return nn.Sequential(*_prune_flat_chain(_flatten_sequential(model), amount))

        found = _prune_tree(model, amount)
        if found == 0:
            raise NotImplementedError(
                f"structured pruning found no supported nn.Sequential chain of "
                f"{[t.__name__ for t in _PRUNABLE_TYPES]}/BatchNorm/activation "
                f"layers anywhere in {type(model).__name__} — nothing to prune"
            )
        return model


# ------------------------------------------------------------------
# Island discovery and in-place tree rewriting
# ------------------------------------------------------------------

def _flatten_sequential(seq: "nn.Sequential") -> List["nn.Module"]:
    """Recursively resolve nested nn.Sequential children into a flat list."""
    flat: List["nn.Module"] = []
    for child in seq:
        if isinstance(child, nn.Sequential):
            flat.extend(_flatten_sequential(child))
        else:
            flat.append(child)
    return flat


def _is_supported_leaf(m: "nn.Module") -> bool:
    return isinstance(m, _PRUNABLE_TYPES + _BN_TYPES + _TRANSPARENT_TYPES)


def _is_prunable_island(seq: "nn.Sequential") -> bool:
    flat = _flatten_sequential(seq)
    return len(flat) > 0 and all(_is_supported_leaf(m) for m in flat)


def _prune_tree(module: "nn.Module", amount: float) -> int:
    """
    Recursively find qualifying ``nn.Sequential`` islands among ``module``'s
    descendants and replace each in place with its pruned, flattened
    rebuild. Does not recurse further into an island once found (it's
    processed as one self-contained unit). Returns how many islands were
    found and pruned.
    """
    found = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential) and _is_prunable_island(child):
            new_flat = _prune_flat_chain(_flatten_sequential(child), amount)
            setattr(module, name, nn.Sequential(*new_flat))
            found += 1
        else:
            found += _prune_tree(child, amount)
    return found


# ------------------------------------------------------------------
# Per-chain pruning
# ------------------------------------------------------------------

def _prune_flat_chain(flat_layers: List["nn.Module"], amount: float) -> List["nn.Module"]:
    """
    Prune a single flat chain of layers in place (conceptually — returns a
    new list). "Decider" positions (plain ``nn.Linear`` or ``nn.Conv2d``
    with ``groups == 1``) each choose, via L2 norm, which ``amount``
    fraction of their own output channels to drop — except the chain's
    last decider, whose output is the chain's own output contract and is
    never touched. A depthwise ``nn.Conv2d`` never makes its own decision;
    it just carries forward whatever channel subset the previous decider
    chose. A BatchNorm immediately downstream of a trimmed layer is
    re-sliced to match, using that same layer's own (pre-rebuild) output
    index space.
    """
    decider_positions = [
        i for i, layer in enumerate(flat_layers)
        if isinstance(layer, nn.Linear) or (isinstance(layer, nn.Conv2d) and layer.groups == 1)
    ]
    if not decider_positions:
        return flat_layers
    last_decider = decider_positions[-1]

    new_layers = list(flat_layers)
    carry_keep: Optional["torch.Tensor"] = None
    carry_block_size = 1

    for i, layer in enumerate(flat_layers):
        if isinstance(layer, nn.Linear):
            is_last = i == last_decider
            in_keep = _expand_keep(carry_keep, carry_block_size)
            new_layer, out_keep = _rebuild_linear(layer, amount, in_keep, is_last)
            new_layers[i] = new_layer
            carry_keep, carry_block_size = out_keep, 1

        elif isinstance(layer, nn.Conv2d) and layer.groups == 1:
            is_last = i == last_decider
            new_layer, out_keep = _rebuild_conv2d(layer, amount, carry_keep, is_last)
            new_layers[i] = new_layer
            carry_keep = out_keep
            next_decider = next((d for d in decider_positions if d > i), None)
            if next_decider is not None and isinstance(flat_layers[next_decider], nn.Linear):
                next_in_features = flat_layers[next_decider].in_features
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

        elif isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels == layer.out_channels:
            if carry_keep is not None:
                new_layers[i] = _rebuild_depthwise(layer, carry_keep)
            # else: nothing upstream decided a subset — leave untouched,
            # carry_keep stays None.

        elif isinstance(layer, nn.Conv2d):
            raise NotImplementedError(
                f"structured pruning does not support grouped Conv2d with "
                f"groups={layer.groups} that is neither groups=1 nor fully "
                "depthwise (groups == in_channels == out_channels)"
            )

        elif isinstance(layer, _BN_TYPES):
            if carry_keep is not None:
                new_layers[i] = _rebuild_batchnorm(layer, carry_keep)
            # carry_keep is unchanged either way — BN never changes channel count.

        elif isinstance(layer, _TRANSPARENT_TYPES):
            pass

        else:
            raise NotImplementedError(
                f"unsupported layer type in prunable chain: {type(layer).__name__}"
            )

    return new_layers


def _expand_keep(carry_keep: Optional["torch.Tensor"], block_size: int) -> Optional["torch.Tensor"]:
    if carry_keep is None:
        return None
    if block_size == 1:
        return carry_keep
    return torch.cat([
        torch.arange(c * block_size, (c + 1) * block_size)
        for c in carry_keep.tolist()
    ])


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


def _rebuild_depthwise(layer: "nn.Conv2d", in_keep: "torch.Tensor") -> "nn.Conv2d":
    """
    A depthwise conv (``groups == in_channels == out_channels``) maps
    channel i to channel i — it never chooses its own channel subset, it
    just keeps whichever channels ``in_keep`` (the upstream decider's own
    output index space) selected.
    """
    W = layer.weight.data[in_keep]
    b = layer.bias.data[in_keep] if layer.bias is not None else None
    new_layer = nn.Conv2d(
        len(in_keep), len(in_keep),
        kernel_size=layer.kernel_size, stride=layer.stride,
        padding=layer.padding, dilation=layer.dilation,
        groups=len(in_keep), bias=b is not None,
    )
    new_layer.weight.data = W.clone()
    if b is not None:
        new_layer.bias.data = b.clone()
    return new_layer


def _rebuild_batchnorm(bn: "nn.Module", keep: "torch.Tensor") -> "nn.Module":
    """Re-slice a BatchNorm's per-channel parameters/buffers by ``keep``."""
    cls = type(bn)
    new_bn = cls(
        len(keep), eps=bn.eps, momentum=bn.momentum,
        affine=bn.affine, track_running_stats=bn.track_running_stats,
    )
    if bn.affine:
        new_bn.weight.data = bn.weight.data[keep].clone()
        new_bn.bias.data = bn.bias.data[keep].clone()
    if bn.track_running_stats and bn.running_mean is not None:
        new_bn.running_mean.data = bn.running_mean.data[keep].clone()
        new_bn.running_var.data = bn.running_var.data[keep].clone()
        new_bn.num_batches_tracked.data = bn.num_batches_tracked.data.clone()
    return new_bn


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
