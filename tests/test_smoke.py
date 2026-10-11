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


def test_tkg_what_changed_since_catches_updates():
    from datetime import datetime, timedelta, timezone
    from meta_optimizer.temporal_memory import EntityType, TemporalKnowledgeGraph

    tkg = TemporalKnowledgeGraph()
    node_id = tkg.add_knowledge("Old fact created long ago", EntityType.GENERIC)

    node = tkg._graph.get_node(node_id)
    node.created_at = datetime.now(timezone.utc) - timedelta(days=100)
    node.updated_at = datetime.now(timezone.utc) - timedelta(days=100)

    tkg.update_knowledge(node_id, {"status": "just updated today"})

    changed = tkg.what_changed_since(days_ago=7)
    assert any(r["node_id"] == node_id for r in changed)

    related = tkg._temporal.find_related_in_timeframe("Old fact", days_ago=7)
    assert any(r["node_id"] == node_id for r in related)


def test_tkg_get_timeline_still_filters_by_created_at():
    from datetime import datetime, timedelta, timezone
    from meta_optimizer.temporal_memory import EntityType, TemporalKnowledgeGraph

    tkg = TemporalKnowledgeGraph()
    node_id = tkg.add_knowledge("Old fact created long ago", EntityType.GENERIC)

    node = tkg._graph.get_node(node_id)
    node.created_at = datetime.now(timezone.utc) - timedelta(days=100)
    node.updated_at = datetime.now(timezone.utc) - timedelta(days=100)

    # Updating it recently must NOT make it appear in a timeline window
    # that only covers the last 7 days — get_timeline's contract is
    # "created within", unlike what_changed_since's "created or updated".
    tkg.update_knowledge(node_id, {"status": "just updated today"})

    timeline = tkg.get_timeline(start=datetime.now(timezone.utc) - timedelta(days=7))
    assert not any(r["node_id"] == node_id for r in timeline)


def test_quantizer_readme_testing_example():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import QuantizationConfig, RealQuantizer

    quantizer = RealQuantizer(QuantizationConfig(method="dynamic_int8"))
    model = torch.nn.Linear(10, 5)
    test_input = torch.randn(1, 10)
    assert isinstance(quantizer.quantize(model, test_input).to_dict(), dict)


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
