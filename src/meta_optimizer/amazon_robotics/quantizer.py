"""
PyTorch INT8 and INT4 quantization for production edge deployment.

Implements:
- Dynamic INT8 quantization (no calibration data needed) — via torchao's
  ``quantize_()`` API, since ``torch.ao.quantization.quantize_dynamic`` is
  deprecated upstream and slated for removal.
- Static INT8 quantization (with calibration) — via torchao's PT2E flow:
  ``torch.export`` captures the graph, ``X86InductorQuantizer`` annotates
  it, ``prepare_pt2e``/``convert_pt2e`` do the calibrated conversion.
  Only ``backend="fbgemm"`` (x86) is supported: torchao's ARM/XNNPACK
  quantizer only ships under its private ``torchao.testing`` namespace
  in this version, not as a stable API, so ``backend="qnnpack"`` raises
  rather than silently depending on a testing-only module. Converting
  leaves the original float32 weight next to its quantized replacement
  (``torch.export`` preserves the whole original submodule attribute,
  even though the quantized graph never reads it) — ``_drop_unused_attrs``
  removes that dead weight so size metrics reflect the real quantized
  model rather than double-counting it.
- INT4 weight-only quantization (via manual rounding) — kept as a
  CPU-only simulation. torchao's real INT4 tensor formats
  (``Int4WeightOnlyConfig``) all require either a CUDA GPU or the
  GPU-only ``mslk`` package on this platform/version (verified against
  torchao 0.18.0 on x86_64 Linux CPU: ``plain`` and ``preshuffled``
  raise "Requires mslk >= 1.0.0", ``plain_int32`` raises
  "does not support device 'cpu' yet", and ``tile_packed_to_4d``
  requires a CUDA driver). There is no portable real INT4 kernel path
  for this library's CPU-first CI, so this method stays as-is.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.quantization as tq
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
        get_default_x86_inductor_quantization_config,
    )
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False


@dataclass
class QuantizationConfig:
    """Configuration for a quantization pass."""
    method: str = "dynamic_int8"      # dynamic_int8 | static_int8 | int4_weight
    dtype: str = "int8"               # int8 | int4
    per_channel: bool = True
    calibration_batches: int = 10
    backend: str = "fbgemm"           # fbgemm (x86) | qnnpack (ARM)


@dataclass
class QuantizationResult:
    """Holds the quantized model and associated metrics."""
    original_model: Any
    quantized_model: Any
    method: str
    original_size_mb: float
    quantized_size_mb: float
    original_latency_ms: float
    quantized_latency_ms: float
    accuracy_delta: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def size_reduction(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return 1.0 - self.quantized_size_mb / self.original_size_mb

    @property
    def latency_improvement(self) -> float:
        if self.quantized_latency_ms == 0:
            return 0.0
        return self.original_latency_ms / self.quantized_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "original_size_mb": round(self.original_size_mb, 3),
            "quantized_size_mb": round(self.quantized_size_mb, 3),
            "size_reduction": round(self.size_reduction, 3),
            "original_latency_ms": round(self.original_latency_ms, 3),
            "quantized_latency_ms": round(self.quantized_latency_ms, 3),
            "latency_improvement": round(self.latency_improvement, 2),
            "accuracy_delta": round(self.accuracy_delta, 4),
            "success": self.success,
            "error": self.error,
        }


class RealQuantizer:
    """
    Production-grade PyTorch quantization.

    Supports dynamic INT8 (fastest to apply), static INT8 (best accuracy/
    speed tradeoff), and INT4 weight-only (smallest model size).
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")
        self.config = config or QuantizationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        method: Optional[str] = None,
    ) -> QuantizationResult:
        """
        Quantize ``model`` using the specified method.

        Args:
            model: PyTorch module to quantize.
            test_input: Representative input tensor for latency benchmarking.
            method: Override for self.config.method.

        Returns:
            QuantizationResult with metrics and quantized model.
        """
        method = method or self.config.method
        original_size = _model_size_mb(model)
        original_latency = _benchmark_latency(model, test_input)

        try:
            if method == "dynamic_int8":
                q_model = self._dynamic_int8(model)
            elif method == "static_int8":
                q_model = self._static_int8(model, test_input)
            elif method == "int4_weight":
                q_model = self._int4_weight_only(model)
            else:
                raise ValueError(f"Unknown quantization method: {method}")

            q_size = _model_size_mb(q_model)
            q_latency = _benchmark_latency(q_model, test_input)

            return QuantizationResult(
                original_model=model,
                quantized_model=q_model,
                method=method,
                original_size_mb=original_size,
                quantized_size_mb=q_size,
                original_latency_ms=original_latency,
                quantized_latency_ms=q_latency,
            )
        except Exception as exc:
            return QuantizationResult(
                original_model=model,
                quantized_model=model,
                method=method,
                original_size_mb=original_size,
                quantized_size_mb=original_size,
                original_latency_ms=original_latency,
                quantized_latency_ms=original_latency,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal quantization implementations
    # ------------------------------------------------------------------

    def _dynamic_int8(self, model: "nn.Module") -> "nn.Module":
        """
        Dynamic INT8 quantization via torchao — per-channel int8 weights,
        with activations quantized dynamically at inference. No
        calibration data required. Targets ``nn.Linear`` layers only;
        unlike the deprecated ``torch.ao.quantization.quantize_dynamic``
        this does not also cover ``nn.LSTM``/``nn.GRU``.
        """
        if not HAS_TORCHAO:
            raise ImportError("torchao is required. Install with: pip install torchao")
        import copy
        m = copy.deepcopy(model)
        m.eval()
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        return m

    def _static_int8(
        self,
        model: "nn.Module",
        calibration_input: "torch.Tensor",
    ) -> "nn.Module":
        """
        Static INT8 quantization via torchao's PT2E flow — both weights
        and activations are quantized using scale factors calibrated
        from representative data.
        """
        if not HAS_TORCHAO:
            raise ImportError("torchao is required. Install with: pip install torchao")
        if self.config.backend != "fbgemm":
            raise NotImplementedError(
                f"backend={self.config.backend!r} has no stable torchao PT2E "
                "quantizer in this version; only 'fbgemm' (X86InductorQuantizer) "
                "is supported"
            )

        import copy
        m = copy.deepcopy(model)
        m.eval()

        exported = torch.export.export(m, (calibration_input,))
        graph_module = exported.module()

        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config())
        prepared = prepare_pt2e(graph_module, quantizer)

        # Calibration pass
        with torch.no_grad():
            for _ in range(self.config.calibration_batches):
                prepared(calibration_input)

        converted = convert_pt2e(prepared)
        _drop_unused_attrs(converted)
        # Exported GraphModules reject .train()/.eval() by default, but
        # _benchmark_latency (and any caller chaining this result into
        # another step) calls .eval() on whatever model it's handed.
        allow_exported_model_train_eval(converted)
        return converted

    def _int4_weight_only(self, model: "nn.Module") -> "nn.Module":
        """
        INT4 weight-only quantization via manual rounding to 4-bit
        precision. Activations remain in float32 at inference time.
        """
        import copy
        m = copy.deepcopy(model)
        m.eval()

        with torch.no_grad():
            for module in m.modules():
                if isinstance(module, nn.Linear) and module.weight is not None:
                    w = module.weight.data.float()
                    w_min, w_max = w.min(), w.max()
                    scale = (w_max - w_min) / 15.0  # 4-bit: 0..15
                    if scale > 0:
                        w_q = torch.clamp(torch.round((w - w_min) / scale), 0, 15)
                        module.weight.data = (w_q * scale + w_min).to(module.weight.dtype)
        return m


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _drop_unused_attrs(graph_module: "torch.fx.GraphModule") -> None:
    """
    Remove parameters/buffers that carry no ``get_attr`` node in
    ``graph_module``'s traced graph.

    ``torch.export`` preserves the full original submodule hierarchy as
    plain attributes, independent of what the traced graph actually
    reads. PT2E's converted graph never reads the original float32
    weight (it reads a separately-stored quantized replacement instead),
    so that weight sits there as dead memory unless removed — otherwise
    a "quantized" model would measure *larger* than the original.
    """
    referenced = {n.target for n in graph_module.graph.nodes if n.op == "get_attr"}
    dead = [name for name, _ in graph_module.named_parameters() if name not in referenced]
    dead += [name for name, _ in graph_module.named_buffers() if name not in referenced]
    for dotted_name in dead:
        *path, leaf = dotted_name.split(".")
        owner = graph_module
        for part in path:
            owner = getattr(owner, part)
        delattr(owner, leaf)


def _tensor_storage_bytes(t: "torch.Tensor") -> int:
    """
    Return the real storage size of a tensor in bytes.

    Tensor-subclass parameters (e.g. torchao's ``Int8Tensor``) report
    their *logical* dtype/shape via ``element_size()``/``nelement()`` —
    for a float32-logical int8-packed weight that silently reports the
    unquantized float32 size. ``__tensor_flatten__`` unwraps such a
    subclass to its real packed inner tensors (quantized data, scale,
    zero point, ...), recursively, so compression is measured honestly.
    """
    if hasattr(t, "__tensor_flatten__"):
        attrs, _ = t.__tensor_flatten__()
        return sum(_tensor_storage_bytes(getattr(t, name)) for name in attrs)
    return t.nelement() * t.element_size()


def _model_size_mb(model: "nn.Module") -> float:
    """Return the in-memory parameter size of ``model`` in megabytes."""
    total_bytes = sum(_tensor_storage_bytes(p.data) for p in model.parameters())
    total_bytes += sum(_tensor_storage_bytes(b) for b in model.buffers())
    return total_bytes / (1024 ** 2)


def _benchmark_latency(
    model: "nn.Module",
    test_input: "torch.Tensor",
    warmup: int = 5,
    runs: int = 20,
) -> float:
    """Return mean inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(test_input)
        start = time.perf_counter()
        for _ in range(runs):
            model(test_input)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0
