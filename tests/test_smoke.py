"""Smoke tests for the README Quick Start examples."""

import pytest


def test_import_package():
    import meta_optimizer  # noqa: F401
    import meta_optimizer.temporal_memory  # noqa: F401
    import meta_optimizer.orchestration  # noqa: F401


def test_tkg_readme_testing_example():
    from meta_optimizer.temporal_memory import EntityType, TemporalKnowledgeGraph

    tkg = TemporalKnowledgeGraph()
    tkg.add_knowledge("Test entry", EntityType.GENERIC)
    assert isinstance(tkg.get_statistics(), dict)


def test_tkg_readme_quick_start():
    from meta_optimizer.temporal_memory import EntityType, TemporalKnowledgeGraph

    tkg = TemporalKnowledgeGraph()
    tkg.add_knowledge(
        text="Alexander optimized MobileNet with INT8 quantization achieving 4.2x speedup",
        entity_type=EntityType.OPTIMIZATION,
        metadata={"model": "MobileNet", "method": "INT8", "speedup": 4.2},
    )
    assert tkg.search("optimizations for MobileNet") is not None
    tkg.what_changed_since(days_ago=7)
    tkg.when_was_last("model optimization")


def test_quantizer_readme_testing_example():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import QuantizationConfig, RealQuantizer

    quantizer = RealQuantizer(QuantizationConfig(method="dynamic_int8"))
    model = torch.nn.Linear(10, 5)
    test_input = torch.randn(1, 10)
    assert isinstance(quantizer.quantize(model, test_input).to_dict(), dict)


def test_structured_prune_linear_chain():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    pruner = RealPruner(PruningConfig(method="structured_l2", amount=0.3))
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10))
    test_input = torch.randn(4, 100)
    orig_out = model(test_input)

    result = pruner.prune(model, test_input)
    assert result.success, result.error
    # The first (non-final) Linear's out_features should have shrunk ~30%.
    assert 0.2 < result.size_reduction < 0.4, result.size_reduction

    pruned_out = result.pruned_model(test_input)
    # The final layer's output shape is the model's contract — it must not change.
    assert pruned_out.shape == orig_out.shape


def test_structured_prune_conv_chain():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    pruner = RealPruner(PruningConfig(method="structured_l2", amount=0.25))
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
    )
    test_input = torch.randn(2, 3, 8, 8)
    orig_out = model(test_input)

    result = pruner.prune(model, test_input)
    assert result.success, result.error
    assert 0.15 < result.size_reduction < 0.35, result.size_reduction

    pruned_out = result.pruned_model(test_input)
    assert pruned_out.shape == orig_out.shape


def test_structured_prune_conv_flatten_linear():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    pruner = RealPruner(PruningConfig(method="structured_l2", amount=0.25))
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 8 * 8, 10),
    )
    test_input = torch.randn(2, 3, 8, 8)
    orig_out = model(test_input)

    result = pruner.prune(model, test_input)
    assert result.success, result.error
    assert 0.15 < result.size_reduction < 0.35, result.size_reduction

    # Confirms the channel -> flatten-index block mapping lines up correctly;
    # a wrong mapping would mix channels and produce garbage, not a crash.
    pruned_out = result.pruned_model(test_input)
    assert pruned_out.shape == orig_out.shape


def test_structured_prune_rejects_non_sequential():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)

        def forward(self, x):
            return x + self.fc(x)

    pruner = RealPruner(PruningConfig(method="structured_l2", amount=0.3))
    result = pruner.prune(Residual(), torch.randn(1, 10))
    assert not result.success
    assert "nn.Sequential" in result.error

    # A BatchNorm inside an otherwise-valid Sequential must also be rejected —
    # its per-channel parameters would need trimming too, which this
    # implementation does not attempt.
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.BatchNorm1d(50), torch.nn.Linear(50, 10))
    result = pruner.prune(model, torch.randn(4, 100))
    assert not result.success
    assert "BatchNorm1d" in result.error


def test_unstructured_prune_raises():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    pruner = RealPruner(PruningConfig(method="unstructured_l1", amount=0.3))
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.Linear(50, 10))
    result = pruner.prune(model, torch.randn(4, 100))
    assert not result.success
    assert "sparsity" in result.error


def test_quantizer_static_int8():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    from meta_optimizer.amazon_robotics import QuantizationConfig, RealQuantizer

    quantizer = RealQuantizer(QuantizationConfig(method="static_int8", calibration_batches=2))
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    test_input = torch.randn(1, 100)
    result = quantizer.quantize(model, test_input)
    assert result.success, result.error
    # ~71% expected (int8/float32 storage ratio plus small scale/zero-point
    # overhead). A narrow band, not just `> 0`, so a regression that makes
    # this fall back to measuring unquantized size (0%) or that silently
    # keeps the dead pre-quantization weight around (negative) both fail
    # loudly instead of slipping through on a loose bound.
    assert 0.6 < result.size_reduction < 0.8, result.size_reduction
    # The converted graph module must still run on fresh input afterwards.
    assert result.quantized_model(torch.randn(1, 100)).shape == (1, 10)


def test_pipeline_readme_quick_start():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import OptimizationPipeline

    pipeline = OptimizationPipeline()
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    test_input = torch.randn(1, 100)
    results = pipeline.optimize(model=model, test_input=test_input, strategy="hybrid_moderate")
    assert "final_metrics" in results
    assert "latency_improvement" in results["final_metrics"]
    assert "size_reduction" in results["final_metrics"]
    assert "overall_success" in results


def test_workflow_visualize():
    pytest.importorskip("torch")
    from meta_optimizer.orchestration import MetaOptimizerWorkflow

    assert MetaOptimizerWorkflow().visualize_workflow()


def test_workflow_readme_quick_start():
    torch = pytest.importorskip("torch")
    from meta_optimizer.orchestration import MetaOptimizerWorkflow

    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    test_input = torch.randn(1, 100)
    result = MetaOptimizerWorkflow().run(
        request="Optimize model for edge deployment",
        request_type="optimize_model",
        model=model,
        test_input=test_input,
        max_retries=3,
        use_parallel=True,
        strategies=["fast_int8", "compact_prune", "hybrid_moderate"],
    )
    assert "success" in result
    assert "optimization_results" in result
