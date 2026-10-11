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


def test_sd_recipe_roundtrip_mobilenet_v2():
    """
    Step 7 #2: sd+recipe round-trip on S1 (in-process) quantization,
    verified on MobileNetV2 — not the toy Linear model — per the step-7
    rule that size/compression-adjacent checks use MobileNetV2.

    Covers three S1 transforms: dynamic (Int8DynamicActivationInt8WeightConfig,
    no structural change), weight-only (Int8WeightOnlyConfig, no structural
    change), and swap+weight-only (swap_conv2d_1x1_to_linear first, a real
    structural change — the meta skeleton needs the same structural
    transform applied before load_state_dict can match shapes).
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import mobilenet_v2
    from torchao.quantization import (
        quantize_,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
    )
    from torchao.quantization.quant_api import swap_conv2d_1x1_to_linear
    from meta_optimizer.amazon_robotics import save_sd_recipe, load_sd_recipe

    torch.manual_seed(0)
    base = mobilenet_v2(weights=None).eval()
    x = torch.randn(1, 3, 160, 160)

    def dynamic(m):
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        return m

    def weight_only(m):
        quantize_(m, Int8WeightOnlyConfig())
        return m

    def swap_weight_only(m):
        swap_conv2d_1x1_to_linear(m)
        quantize_(m, Int8WeightOnlyConfig())
        return m

    for transform_fn, needs_structural_reapply in (
        (dynamic, False),
        (weight_only, False),
        (swap_weight_only, True),
    ):
        import copy
        quantized = transform_fn(copy.deepcopy(base)).eval()
        with torch.no_grad():
            expected_out = quantized(x)

        sd_recipe = save_sd_recipe(quantized, transform_fn, strategy_name=transform_fn.__name__)
        assert sd_recipe.recipe["area"] == "S1"
        assert "torch" in sd_recipe.recipe["versions"]

        def build_meta_skeleton(needs_structural_reapply=needs_structural_reapply, transform_fn=transform_fn):
            with torch.device("meta"):
                m = mobilenet_v2(weights=None)
            return transform_fn(m) if needs_structural_reapply else m

        reloaded = load_sd_recipe(
            sd_recipe.state_dict_bytes, sd_recipe.recipe, build_meta_skeleton, transform_fn,
        ).eval()
        with torch.no_grad():
            reloaded_out = reloaded(x)
        assert torch.equal(expected_out, reloaded_out), transform_fn.__name__

    # Tamper check: a different transform_fn's source must be rejected,
    # not silently loaded onto mismatched tensors.
    quantized = dynamic(copy.deepcopy(base)).eval()
    sd_recipe = save_sd_recipe(quantized, dynamic, strategy_name="dynamic")
    with pytest.raises(ValueError):
        load_sd_recipe(
            sd_recipe.state_dict_bytes, sd_recipe.recipe,
            lambda: mobilenet_v2(weights=None), weight_only,
        )


def test_sd_recipe_not_available_for_pt2e():
    """sd+recipe is explicitly NOT_AVAILABLE for PT2E/static_int8 models —
    see serialization.py's module docstring for why. Must raise loudly,
    not silently produce an unloadable artifact."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    from meta_optimizer.amazon_robotics import QuantizationConfig, RealQuantizer, save_sd_recipe

    quantizer = RealQuantizer(QuantizationConfig(method="static_int8", calibration_batches=2))
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10))
    test_input = torch.randn(1, 100)
    result = quantizer.quantize(model, test_input)
    assert result.success, result.error

    with pytest.raises(NotImplementedError):
        save_sd_recipe(result.quantized_model, lambda m: m, strategy_name="static_int8")


def test_sd_recipe_rejects_version_mismatch_unless_strict_false():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    from meta_optimizer.amazon_robotics import QuantizationConfig, RealQuantizer, save_sd_recipe, load_sd_recipe

    quantizer = RealQuantizer(QuantizationConfig(method="dynamic_int8"))
    model = torch.nn.Linear(10, 5)
    result = quantizer.quantize(model, torch.randn(1, 10))
    assert result.success, result.error

    def transform(m):
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        return m

    sd_recipe = save_sd_recipe(result.quantized_model, transform, strategy_name="dynamic_int8")
    tampered = dict(sd_recipe.recipe)
    tampered["versions"] = {**tampered["versions"], "torch": "0.0.0-not-installed"}

    def build_meta():
        with torch.device("meta"):
            return torch.nn.Linear(10, 5)

    with pytest.raises(ValueError, match="versions"):
        load_sd_recipe(sd_recipe.state_dict_bytes, tampered, build_meta, transform)

    # strict=False must load anyway despite the version mismatch.
    reloaded = load_sd_recipe(sd_recipe.state_dict_bytes, tampered, build_meta, transform, strict=False)
    assert reloaded is not None


def test_sd_recipe_rejects_corrupted_state_dict():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    from meta_optimizer.amazon_robotics import QuantizationConfig, RealQuantizer, save_sd_recipe, load_sd_recipe

    quantizer = RealQuantizer(QuantizationConfig(method="dynamic_int8"))
    model = torch.nn.Linear(10, 5)
    result = quantizer.quantize(model, torch.randn(1, 10))
    assert result.success, result.error

    def transform(m):
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        return m

    sd_recipe = save_sd_recipe(result.quantized_model, transform, strategy_name="dynamic_int8")
    corrupted_bytes = sd_recipe.state_dict_bytes[:-1] + bytes([sd_recipe.state_dict_bytes[-1] ^ 0xFF])

    def build_meta():
        with torch.device("meta"):
            return torch.nn.Linear(10, 5)

    # Corruption must be rejected even with strict=False — it's a data
    # integrity check, not a version-compatibility leniency.
    with pytest.raises(ValueError, match="state_dict"):
        load_sd_recipe(corrupted_bytes, sd_recipe.recipe, build_meta, transform, strict=False)


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


def test_meets_latency_target_checks_actual_target():
    """Regression test for issue #7: meets_latency_target was always True
    because it compared against an empty dict instead of the strategy's
    own target_latency_ms/target_size_mb."""
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import OptimizationPipeline, OptimizationStrategy

    pipeline = OptimizationPipeline()
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
    )
    test_input = torch.randn(1, 100)

    impossible = OptimizationStrategy(
        name="impossible_target",
        use_quantization=False,
        use_pruning=False,
        target_latency_ms=1e-9,
        target_size_mb=1e-9,
    )
    result = pipeline.optimize(model=model, test_input=test_input, custom_strategy=impossible)
    assert result["final_metrics"]["meets_latency_target"] is False
    assert result["final_metrics"]["meets_size_target"] is False

    easy = OptimizationStrategy(
        name="easy_target",
        use_quantization=False,
        use_pruning=False,
        target_latency_ms=1e9,
        target_size_mb=1e9,
    )
    result = pipeline.optimize(model=model, test_input=test_input, custom_strategy=easy)
    assert result["final_metrics"]["meets_latency_target"] is True
    assert result["final_metrics"]["meets_size_target"] is True


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
