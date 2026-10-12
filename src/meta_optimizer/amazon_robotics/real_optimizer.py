"""
Integrated optimization pipeline combining quantization, pruning, and export.

Provides predefined strategies and a unified interface for model optimization.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .quantizer import QuantizationConfig, RealQuantizer
from .pruner import PruningConfig, RealPruner
from .exporter import ExportConfig, ModelExporter
from .fine_tuning import FineTuneConfig, fine_tune


# ------------------------------------------------------------------
# Strategy definitions
# ------------------------------------------------------------------

@dataclass
class OptimizationStrategy:
    """Full description of a single optimization strategy."""
    name: str
    use_quantization: bool = True
    quantization_type: str = "dynamic_int8"
    use_pruning: bool = False
    pruning_amount: float = 0.0
    pruning_method: str = "structured_l2"
    fine_tune: bool = False            # run fine_tune() on the pruned model before quantization
    export_onnx: bool = False
    target_latency_ms: float = 10.0
    target_size_mb: float = 100.0


PREDEFINED_STRATEGIES: Dict[str, OptimizationStrategy] = {
    "fast_int8": OptimizationStrategy(
        name="fast_int8",
        use_quantization=True,
        quantization_type="dynamic_int8",
        use_pruning=False,
    ),
    "compact_prune": OptimizationStrategy(
        name="compact_prune",
        use_quantization=False,
        use_pruning=True,
        pruning_amount=0.30,
        pruning_method="structured_l2",
    ),
    "hybrid_moderate": OptimizationStrategy(
        name="hybrid_moderate",
        use_quantization=True,
        quantization_type="int8_weight_only_conv",
        use_pruning=True,
        pruning_amount=0.30,
        pruning_method="structured_l2",
        fine_tune=True,
    ),
    "aggressive": OptimizationStrategy(
        name="aggressive",
        use_quantization=True,
        quantization_type="int8_weight_only_conv",
        use_pruning=True,
        pruning_amount=0.50,
        pruning_method="structured_l2",
        fine_tune=True,
    ),
    "fast_int8_conv_s1": OptimizationStrategy(
        name="fast_int8_conv_s1",
        use_quantization=True,
        quantization_type="int8_weight_only_conv",
        use_pruning=False,
    ),
    "fast_int8_conv_s2": OptimizationStrategy(
        name="fast_int8_conv_s2",
        use_quantization=True,
        quantization_type="static_int8",
        use_pruning=False,
    ),
}


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Consolidated result from running the full optimization pipeline."""
    strategy_name: str
    quantization_result: Optional[Dict[str, Any]] = None
    pruning_result: Optional[Dict[str, Any]] = None
    fine_tune_result: Optional[Dict[str, Any]] = None
    export_result: Optional[Dict[str, Any]] = None
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    overall_success: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "quantization_result": self.quantization_result,
            "pruning_result": self.pruning_result,
            "fine_tune_result": self.fine_tune_result,
            "export_result": self.export_result,
            "final_metrics": self.final_metrics,
            "overall_success": self.overall_success,
            "errors": self.errors,
        }


class OptimizationPipeline:
    """
    End-to-end model optimization pipeline.

    Applies a combination of quantization, pruning, and optional ONNX
    export according to a named or custom ``OptimizationStrategy``.

    Usage::

        pipeline = OptimizationPipeline()
        result = pipeline.optimize(
            model=my_model,
            test_input=torch.randn(1, 100),
            strategy="hybrid_moderate",
        )
        print(result["final_metrics"]["latency_improvement"])
    """

    def __init__(self) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")

    def optimize(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        strategy: str = "hybrid_moderate",
        custom_strategy: Optional[OptimizationStrategy] = None,
        onnx_output_dir: Optional[str] = None,
        fine_tune_config: Optional[FineTuneConfig] = None,
    ) -> Dict[str, Any]:
        """
        Optimize ``model`` using the specified strategy.

        Args:
            model: PyTorch module to optimize.
            test_input: Representative input tensor.
            strategy: Name of a predefined strategy.
            custom_strategy: Override with a custom OptimizationStrategy.
            onnx_output_dir: Directory for ONNX export (None = skip export).
            fine_tune_config: Imagenette2-160 archive path/SHA-256 and
                fine-tuning hyperparameters, used only when the strategy
                declares ``fine_tune=True`` (``hybrid_moderate``,
                ``aggressive``). ``None`` (the default) means no archive is
                configured — ``fine_tune()`` then reports
                ``status="NOT_AVAILABLE"`` rather than running, which per
                A11-11 is recorded as a pipeline error (see Step 1.5 below),
                not silently treated as success.

        Returns:
            PipelineResult dict with metrics and optimized model reference.
        """
        strat = custom_strategy or PREDEFINED_STRATEGIES.get(strategy)
        if strat is None:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(PREDEFINED_STRATEGIES.keys())}"
            )

        result = PipelineResult(strategy_name=strat.name)
        current_model = model

        # --- Step 1: Pruning (before quantization for best results) ---
        if strat.use_pruning and strat.pruning_amount > 0:
            pruner = RealPruner(
                PruningConfig(
                    method=strat.pruning_method,
                    amount=strat.pruning_amount,
                )
            )
            pr = pruner.prune(current_model, test_input)
            result.pruning_result = pr.to_dict()
            if pr.success:
                current_model = pr.pruned_model
            else:
                result.errors.append(f"Pruning failed: {pr.error}")

        # --- Step 1.5: Fine-tuning (mandatory for strategies that declare
        # it — hybrid_moderate/aggressive — run on the pruned float model,
        # before quantization) ---
        if strat.fine_tune:
            ft = fine_tune(current_model, fine_tune_config or FineTuneConfig())
            result.fine_tune_result = ft.to_dict()
            if ft.status != "VERIFIED":
                result.errors.append(f"Fine-tuning {ft.status}: {ft.error}")

        # --- Step 2: Quantization ---
        if strat.use_quantization:
            quantizer = RealQuantizer(
                QuantizationConfig(method=strat.quantization_type)
            )
            qr = quantizer.quantize(current_model, test_input)
            result.quantization_result = qr.to_dict()
            if qr.success:
                current_model = qr.quantized_model
            else:
                result.errors.append(f"Quantization failed: {qr.error}")

        # --- Step 3: ONNX export (optional) ---
        if strat.export_onnx or onnx_output_dir:
            export_dir = onnx_output_dir or tempfile.mkdtemp()
            onnx_path = os.path.join(export_dir, f"{strat.name}.onnx")
            exporter = ModelExporter(ExportConfig())
            er = exporter.export_onnx(current_model, test_input, onnx_path)
            result.export_result = er.to_dict()
            if not er.success:
                result.errors.append(f"ONNX export failed: {er.error}")

        # --- Aggregate final metrics ---
        result.final_metrics = self._compute_final_metrics(strat, model, current_model, test_input)
        result.overall_success = len(result.errors) == 0

        return result.to_dict()

    def _compute_final_metrics(
        self,
        strat: OptimizationStrategy,
        original_model: "nn.Module",
        final_model: "nn.Module",
        test_input: "torch.Tensor",
    ) -> Dict[str, Any]:
        """Compute aggregate metrics by comparing original to final model."""
        from .quantizer import _model_size_mb, _benchmark_latency

        original_size = _model_size_mb(original_model)
        final_size = _model_size_mb(final_model)
        original_latency = _benchmark_latency(original_model, test_input)
        final_latency = _benchmark_latency(final_model, test_input)

        size_reduction = (1.0 - final_size / original_size) if original_size > 0 else 0.0
        latency_improvement = (original_latency / final_latency) if final_latency > 0 else 1.0

        return {
            "original_size_mb": round(original_size, 3),
            "final_size_mb": round(final_size, 3),
            "size_reduction": round(size_reduction, 3),
            "original_latency_ms": round(original_latency, 3),
            "final_latency_ms": round(final_latency, 3),
            "latency_improvement": round(latency_improvement, 2),
            "meets_latency_target": final_latency <= strat.target_latency_ms,
            "meets_size_target": final_size <= strat.target_size_mb,
        }

    def compare_strategies(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple strategies and return a comparison table.

        Args:
            model: Model to optimize.
            test_input: Test input tensor.
            strategies: List of strategy names (default: all predefined).

        Returns:
            Dict mapping strategy name → PipelineResult dict.
        """
        strategy_names = strategies or list(PREDEFINED_STRATEGIES.keys())
        comparison: Dict[str, Any] = {}
        for name in strategy_names:
            try:
                comparison[name] = self.optimize(model, test_input, strategy=name)
            except Exception as exc:
                comparison[name] = {"error": str(exc), "overall_success": False}
        return comparison
