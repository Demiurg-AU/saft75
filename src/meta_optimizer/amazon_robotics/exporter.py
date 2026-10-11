"""
ONNX and TensorRT model export utilities.

Provides:
- ONNX export with opset selection and shape validation
- ONNX Runtime benchmarking
- TensorRT engine building (optional, requires tensorrt package)

Size measurement note: ``torch.onnx.export``'s dynamo-based exporter
(torch>=2.x) writes large initializers to a separate ``<path>.data``
file once they exceed a size threshold, rather than inlining them in
the ``.onnx`` protobuf. Measuring only ``os.path.getsize(output_path)``
silently undercounts the real artifact size whenever that happens —
verified empirically: for a quantized MobileNetV2 variant, the ``.onnx``
graph file alone was *smaller* than FP32's own graph file by a naive
read, while the true total (graph + data) was the honestly compressed
one. ``_onnx_artifact_size`` below always sums both files.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    import numpy as np
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@dataclass
class ExportConfig:
    """Configuration for model export."""
    opset_version: int = 17
    dynamic_axes: Optional[Dict[str, Any]] = None
    simplify: bool = False          # Requires onnx-simplifier
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None


@dataclass
class ExportResult:
    """Result of a model export operation."""
    format: str                     # "onnx" | "tensorrt"
    output_path: str
    original_size_mb: float
    exported_size_mb: float
    original_latency_ms: float
    exported_latency_ms: float
    success: bool = True
    error: Optional[str] = None
    graph_bytes: Optional[int] = None
    data_bytes: Optional[int] = None
    external_data: bool = False
    graph_sha256: Optional[str] = None
    data_sha256: Optional[str] = None
    max_abs_vs_torch: Optional[float] = None

    @property
    def size_reduction(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return 1.0 - self.exported_size_mb / self.original_size_mb

    @property
    def latency_improvement(self) -> float:
        if self.exported_latency_ms == 0:
            return 0.0
        return self.original_latency_ms / self.exported_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "output_path": self.output_path,
            "original_size_mb": round(self.original_size_mb, 3),
            "exported_size_mb": round(self.exported_size_mb, 3),
            "size_reduction": round(self.size_reduction, 3),
            "original_latency_ms": round(self.original_latency_ms, 3),
            "exported_latency_ms": round(self.exported_latency_ms, 3),
            "latency_improvement": round(self.latency_improvement, 2),
            "success": self.success,
            "error": self.error,
            "graph_bytes": self.graph_bytes,
            "data_bytes": self.data_bytes,
            "external_data": self.external_data,
            "graph_sha256": self.graph_sha256,
            "data_sha256": self.data_sha256,
            "max_abs_vs_torch": self.max_abs_vs_torch,
        }


class ModelExporter:
    """
    Export PyTorch models to ONNX or TensorRT formats for
    production edge deployment.
    """

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")
        self.config = config or ExportConfig()

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        output_path: str,
    ) -> ExportResult:
        """
        Export ``model`` to ONNX format.

        Args:
            model: Trained PyTorch module.
            test_input: Example input tensor (defines input shape).
            output_path: Destination .onnx file path.

        Returns:
            ExportResult with size/latency metrics.
        """
        if not HAS_ONNX:
            return ExportResult(
                format="onnx",
                output_path=output_path,
                original_size_mb=0.0,
                exported_size_mb=0.0,
                original_latency_ms=0.0,
                exported_latency_ms=0.0,
                success=False,
                error="onnx package not installed. Run: pip install onnx",
            )

        original_size = _torch_model_size_mb(model)
        original_latency = _torch_benchmark(model, test_input)

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            model.eval()

            input_names = self.config.input_names or ["input"]
            output_names = self.config.output_names or ["output"]
            dynamic_axes = self.config.dynamic_axes or {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    test_input,
                    output_path,
                    opset_version=self.config.opset_version,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                )

            # Validate exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)

            total_bytes, graph_bytes, data_bytes, external_data, graph_sha256, data_sha256 = (
                _onnx_artifact_size(output_path)
            )
            exported_size = total_bytes / (1024 ** 2)

            max_abs_vs_torch = None
            if HAS_ORT:
                exported_latency = _ort_benchmark(output_path, test_input)
                with torch.no_grad():
                    torch_out = model(test_input)
                ort_out = _ort_output(output_path, test_input)
                max_abs_vs_torch = float((torch.as_tensor(ort_out) - torch_out).abs().max())
            else:
                exported_latency = original_latency

            return ExportResult(
                format="onnx",
                output_path=output_path,
                original_size_mb=original_size,
                exported_size_mb=exported_size,
                original_latency_ms=original_latency,
                exported_latency_ms=exported_latency,
                graph_bytes=graph_bytes,
                data_bytes=data_bytes,
                external_data=external_data,
                graph_sha256=graph_sha256,
                data_sha256=data_sha256,
                max_abs_vs_torch=max_abs_vs_torch,
            )

        except Exception as exc:
            return ExportResult(
                format="onnx",
                output_path=output_path,
                original_size_mb=original_size,
                exported_size_mb=0.0,
                original_latency_ms=original_latency,
                exported_latency_ms=0.0,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # TensorRT export (optional)
    # ------------------------------------------------------------------

    def export_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        fp16: bool = True,
        int8: bool = False,
    ) -> ExportResult:
        """
        Build a TensorRT engine from an ONNX model.

        Requires the ``tensorrt`` package and a compatible NVIDIA GPU.
        """
        try:
            import tensorrt as trt  # type: ignore[import]
        except ImportError:
            return ExportResult(
                format="tensorrt",
                output_path=output_path,
                original_size_mb=0.0,
                exported_size_mb=0.0,
                original_latency_ms=0.0,
                exported_latency_ms=0.0,
                success=False,
                error="tensorrt package not installed.",
            )

        onnx_size = os.path.getsize(onnx_path) / (1024 ** 2) if os.path.exists(onnx_path) else 0.0

        try:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    errors = [parser.get_error(i) for i in range(parser.num_errors)]
                    raise RuntimeError(f"TRT parse errors: {errors}")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)

            engine = builder.build_serialized_network(network, config)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(engine)

            trt_size = os.path.getsize(output_path) / (1024 ** 2)
            return ExportResult(
                format="tensorrt",
                output_path=output_path,
                original_size_mb=onnx_size,
                exported_size_mb=trt_size,
                original_latency_ms=0.0,
                exported_latency_ms=0.0,
            )
        except Exception as exc:
            return ExportResult(
                format="tensorrt",
                output_path=output_path,
                original_size_mb=onnx_size,
                exported_size_mb=0.0,
                original_latency_ms=0.0,
                exported_latency_ms=0.0,
                success=False,
                error=str(exc),
            )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _torch_model_size_mb(model: "nn.Module") -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def _torch_benchmark(
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
    return ((time.perf_counter() - start) / runs) * 1000.0


def _ort_benchmark(
    onnx_path: str,
    test_input: "torch.Tensor",
    warmup: int = 5,
    runs: int = 20,
) -> float:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    np_input = test_input.numpy()
    for _ in range(warmup):
        sess.run(None, {input_name: np_input})
    start = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {input_name: np_input})
    return ((time.perf_counter() - start) / runs) * 1000.0


def _ort_output(onnx_path: str, test_input: "torch.Tensor"):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: test_input.numpy()})[0]


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _onnx_artifact_size(output_path: str) -> Tuple[int, int, int, bool, str, Optional[str]]:
    """
    Returns (total_bytes, graph_bytes, data_bytes, external_data,
    graph_sha256, data_sha256) — ``total_bytes`` is the real artifact
    size (graph + external data, if any); see module docstring for why
    the graph file alone undercounts.
    """
    graph_bytes = os.path.getsize(output_path)
    graph_sha256 = _sha256_file(output_path)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        data_bytes = os.path.getsize(data_path)
        data_sha256 = _sha256_file(data_path)
        external_data = True
    else:
        data_bytes = 0
        data_sha256 = None
        external_data = False
    return graph_bytes + data_bytes, graph_bytes, data_bytes, external_data, graph_sha256, data_sha256
