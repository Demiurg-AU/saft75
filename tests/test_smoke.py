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


def test_s2_onnx_export_variants_c_and_e(tmp_path):
    """
    Step 7 #3: S2 (ONNX) export for variants C (swap_conv2d_1x1_to_linear +
    Int8WeightOnlyConfig) and E (PT2E static, X86InductorQuantizer).

    Size is measured as .onnx + .onnx.data combined — see exporter.py's
    module docstring for why the graph file alone undercounts. max_abs
    vs torch is recorded as-is, with no pass/fail threshold: E is known
    (from the XMG witness run) to diverge by ~0.32 in this comparison,
    which is a real, open question for E1 to resolve, not something to
    paper over here.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    torchvision = pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from torchvision.models import mobilenet_v2
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    from torchao.quantization.quant_api import swap_conv2d_1x1_to_linear
    from meta_optimizer.amazon_robotics import ModelExporter, ExportConfig
    import copy

    torch.manual_seed(0)
    base = mobilenet_v2(weights=None).eval()
    x = torch.randn(1, 3, 160, 160)

    def build_c():
        m = copy.deepcopy(base)
        swap_conv2d_1x1_to_linear(m)
        quantize_(m, Int8WeightOnlyConfig())
        return m.eval()

    def build_e():
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer, get_default_x86_inductor_quantization_config,
        )
        from torchao.quantization.pt2e import allow_exported_model_train_eval

        m = copy.deepcopy(base)
        gm = torch.export.export(m, (x,)).module()
        q = X86InductorQuantizer()
        q.set_global(get_default_x86_inductor_quantization_config())
        prepared = prepare_pt2e(gm, q)
        for _ in range(8):
            prepared(torch.randn(1, 3, 160, 160))
        converted = convert_pt2e(prepared)
        allow_exported_model_train_eval(converted)
        # Known pitfall (found earlier this session): .eval()'s return
        # value is None for a PT2E-patched model — never chain it.
        converted.eval()
        return converted

    exporter = ModelExporter(ExportConfig())
    for variant_name, build_fn in (("C", build_c), ("E", build_e)):
        model = build_fn()
        output_path = str(tmp_path / f"s2_{variant_name}.onnx")
        result = exporter.export_onnx(model, x, output_path)
        assert result.success, result.error

        assert result.graph_bytes is not None and result.graph_bytes > 0
        assert result.external_data is True, variant_name
        assert result.data_bytes is not None and result.data_bytes > 0
        assert result.graph_sha256 is not None
        assert result.data_sha256 is not None
        assert result.exported_size_mb == pytest.approx(
            (result.graph_bytes + result.data_bytes) / (1024 ** 2)
        )
        # Recorded as-is — no threshold asserted here (see docstring).
        assert result.max_abs_vs_torch is not None, variant_name


def test_conv_coverage_strategies_via_pipeline_mobilenet_v2(tmp_path):
    """
    Step 7 #4: conv coverage — ``fast_int8_conv_s1`` (swap_conv2d_1x1_to_linear
    + Int8WeightOnlyConfig, S1/sd+recipe-compatible — variant C from PR #2/#3's
    exploratory tests) and ``fast_int8_conv_s2`` (PT2E static, S2/ONNX-only —
    variant E), exercised through ``OptimizationPipeline.optimize()`` with a
    named strategy, not a hand-rolled re-implementation of the transform, on
    MobileNetV2 per the step-7 rule that size/compression-adjacent checks use
    MobileNetV2, not a toy model. ``fast_int8`` (dynamic_int8) is unchanged
    by this PR — these are two new, separate strategy names.

    Acceptance per contract amendment A11-6 (compression is only measured
    between artifacts of the *same format* as the FP32 baseline — never
    FP32-in-memory vs a quantized file, or vice versa):

    - s1 >= 3.4x: sd+recipe artifact bytes (the serialized state_dict file,
      via ``save_sd_recipe`` — same ``torch.save`` format on both sides) for
      FP32 vs the quantized model. The quantized model itself comes from
      ``RealQuantizer`` configured with the strategy's own declared
      ``quantization_type`` (the exact class/method ``optimize()`` calls
      internally) — not a hand-written torchao call — and
      ``quantizer._int8_weight_only_conv`` (the real method that produced
      it) is passed as ``save_sd_recipe``'s ``transform_fn``, so the
      recorded provenance hash matches the actual production code, not a
      duplicate reimplementation of it.
    - s2 >= 3.3x: ONNX artifact bytes (.onnx + .onnx.data combined — see
      exporter.py's module docstring) for FP32 vs the quantized model, both
      exported through the same ``ModelExporter.export_onnx`` call path —
      the FP32 side via a second call on the original model, the quantized
      side via ``optimize()``'s own ``onnx_output_dir`` export.

    ``ExportResult`` no longer carries a ``size_reduction`` property (see
    exporter.py's module docstring for why: it used to compare an ONNX file
    to the in-memory size of whatever model was passed into ``export_onnx``,
    which by the time the pipeline calls it is already the quantized model,
    not FP32 — a cross-format, misleading ratio under A11-6).

    max_abs vs torch is recorded as present, with no pass/fail bound —
    MobileNetV2 with random (untrained) weights produces near-float32-
    noise-floor output magnitude, making a precision threshold on this
    comparison numerically meaningless (established earlier this session
    against the XMG archive).
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    torchvision = pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from torchvision.models import mobilenet_v2
    from meta_optimizer.amazon_robotics import (
        OptimizationPipeline,
        PREDEFINED_STRATEGIES,
        RealQuantizer,
        QuantizationConfig,
        ModelExporter,
        ExportConfig,
        save_sd_recipe,
    )

    assert PREDEFINED_STRATEGIES["fast_int8"].quantization_type == "dynamic_int8"

    torch.manual_seed(0)
    model = mobilenet_v2(weights=None).eval()
    test_input = torch.randn(1, 3, 160, 160)
    pipeline = OptimizationPipeline()

    # --- s1: acceptance gate via optimize() with the named strategy ---
    result_s1 = pipeline.optimize(model=model, test_input=test_input, strategy="fast_int8_conv_s1")
    assert result_s1["overall_success"], result_s1["errors"]
    assert result_s1["quantization_result"]["success"], result_s1["quantization_result"]["error"]

    # --- s1: A11-6 measurement — sd+recipe file bytes, FP32 vs quantized,
    # same torch.save format on both sides. ---
    strat_s1 = PREDEFINED_STRATEGIES["fast_int8_conv_s1"]
    quantizer_s1 = RealQuantizer(QuantizationConfig(method=strat_s1.quantization_type))
    quantized_s1 = quantizer_s1.quantize(model, test_input).quantized_model

    fp32_sd = save_sd_recipe(model, lambda m: m, strategy_name="fp32_baseline")
    quant_sd = save_sd_recipe(quantized_s1, quantizer_s1._int8_weight_only_conv, strategy_name="fast_int8_conv_s1")
    ratio_s1 = len(fp32_sd.state_dict_bytes) / len(quant_sd.state_dict_bytes)
    assert ratio_s1 >= 3.4, ratio_s1

    # --- s2: acceptance gate via optimize() with the named strategy,
    # exporting the quantized model to ONNX through its own onnx_output_dir. ---
    result_s2 = pipeline.optimize(
        model=model, test_input=test_input, strategy="fast_int8_conv_s2",
        onnx_output_dir=str(tmp_path),
    )
    assert result_s2["overall_success"], result_s2["errors"]
    assert result_s2["quantization_result"]["success"], result_s2["quantization_result"]["error"]
    export_result = result_s2["export_result"]
    assert export_result["success"], export_result["error"]
    assert export_result["external_data"] is True
    # Recorded as-is — no threshold asserted here (see docstring).
    assert export_result["max_abs_vs_torch"] is not None

    # --- s2: A11-6 measurement — ONNX artifact bytes, FP32 vs quantized,
    # both exported through the same ModelExporter.export_onnx call path. ---
    exporter = ModelExporter(ExportConfig())
    fp32_export = exporter.export_onnx(model, test_input, str(tmp_path / "fp32_baseline.onnx"))
    assert fp32_export.success, fp32_export.error
    fp32_onnx_bytes = fp32_export.graph_bytes + (fp32_export.data_bytes or 0)
    quant_onnx_bytes = export_result["graph_bytes"] + export_result["data_bytes"]
    ratio_s2 = fp32_onnx_bytes / quant_onnx_bytes
    assert ratio_s2 >= 3.3, ratio_s2


def test_structured_prune_linear_chain():
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    pruner = RealPruner(PruningConfig(method="structured_l2", amount=0.3))
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10))
    test_input = torch.randn(4, 100)
    orig_out = model(test_input)

    result = pruner.prune(model, test_input)
    assert result.success, result.error
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


def test_structured_prune_batchnorm_supported():
    """
    BatchNorm1d/2d is now a supported, actively-trimmed layer (needed for
    any real CNN, which always interleaves Conv2d with BatchNorm2d) — not
    rejected, unlike the pre-step-7 implementation. Its own per-channel
    weight/bias/running_mean/running_var get re-sliced to match whatever
    the preceding Linear/Conv2d pruned, verified here by checking the
    model still runs and the output shape is unchanged.
    """
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50), torch.nn.BatchNorm1d(50), torch.nn.Linear(50, 10),
    )
    test_input = torch.randn(4, 100)
    model.eval()
    orig_out = model(test_input)

    pruner = RealPruner(PruningConfig(method="structured_l2", amount=0.3))
    result = pruner.prune(model, test_input)
    assert result.success, result.error
    assert 0.2 < result.size_reduction < 0.4, result.size_reduction

    pruned_model = result.pruned_model.eval()
    pruned_out = pruned_model(test_input)
    assert pruned_out.shape == orig_out.shape


def test_structured_prune_rejects_no_island():
    """A model with no supported nn.Sequential island anywhere (not even
    nested) raises NotImplementedError rather than silently doing nothing."""
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


def test_unstructured_prune_method_raises():
    """Unstructured pruning was removed entirely (issue #6) — any method
    string other than structured_l2 is rejected as unknown, not silently
    accepted and then a no-op."""
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import PruningConfig, RealPruner

    pruner = RealPruner(PruningConfig(method="unstructured_l1", amount=0.3))
    result = pruner.prune(torch.nn.Sequential(torch.nn.Linear(10, 5)), torch.randn(1, 10))
    assert not result.success
    assert "Unknown pruning method" in result.error


def test_physical_pruning_and_hybrid_moderate_mobilenet_v2():
    """
    Step 7 #5: physical channel pruning — real structured removal (not
    masking), generalized beyond a flat nn.Sequential to nested islands
    (see pruner.py's module docstring) so it actually works on MobileNetV2's
    InvertedResidual blocks (BatchNorm2d throughout, depthwise convs).

    Acceptance, both via OptimizationPipeline.optimize() with the named
    hybrid_moderate strategy (pruning + int8_weight_only_conv — NOT
    dynamic_int8, which can't reach the combined 4x target; fast_int8 and
    compact_prune are unchanged):

    - >= 13.5% of parameters physically removed (pruning_result's own
      remaining_params/original_params, real counts — not a mask's
      nominal "sparsity" over zeroed-but-still-stored weights).
    - hybrid_moderate's sd+recipe artifact (state_dict file bytes) is
      >= 4x smaller than the FP32 baseline's, same torch.save format on
      both sides (A11-6). The quantized+pruned model for this measurement
      comes from RealPruner/RealQuantizer configured with the strategy's
      own declared pruning_method/quantization_type — the exact classes
      optimize() calls internally — not a reimplementation of the
      transform; quantizer._int8_weight_only_conv (the real method that
      produced it) is passed as save_sd_recipe's transform_fn.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import mobilenet_v2
    from meta_optimizer.amazon_robotics import (
        OptimizationPipeline, PREDEFINED_STRATEGIES, RealPruner, PruningConfig,
        RealQuantizer, QuantizationConfig, save_sd_recipe,
    )

    torch.manual_seed(0)
    model = mobilenet_v2(weights=None).eval()
    test_input = torch.randn(1, 3, 160, 160)
    pipeline = OptimizationPipeline()

    result = pipeline.optimize(model=model, test_input=test_input, strategy="hybrid_moderate")
    # NOT overall_success: hybrid_moderate requires fine-tuning (A11-11,
    # see test_fine_tune_required_for_hybrid_strategies_a11_11) and no
    # archive is configured here, so overall_success is correctly False —
    # pruning and quantization succeeding on their own is what this test
    # actually checks.
    pruning_result = result["pruning_result"]
    assert pruning_result["success"], pruning_result["error"]
    quantization_result = result["quantization_result"]
    assert quantization_result["success"], quantization_result["error"]

    original_params = sum(p.numel() for p in model.parameters())
    physical_reduction = 1 - pruning_result["remaining_params"] / original_params
    assert physical_reduction >= 0.135, physical_reduction

    strat = PREDEFINED_STRATEGIES["hybrid_moderate"]
    pruner = RealPruner(PruningConfig(method=strat.pruning_method, amount=strat.pruning_amount))
    pr = pruner.prune(model, test_input)
    assert pr.success, pr.error
    quantizer = RealQuantizer(QuantizationConfig(method=strat.quantization_type))
    qr = quantizer.quantize(pr.pruned_model, test_input)
    assert qr.success, qr.error

    fp32_sd = save_sd_recipe(model, lambda m: m, strategy_name="fp32_baseline")
    quant_sd = save_sd_recipe(qr.quantized_model, quantizer._int8_weight_only_conv, strategy_name="hybrid_moderate")
    ratio = len(fp32_sd.state_dict_bytes) / len(quant_sd.state_dict_bytes)
    assert ratio >= 4.0, ratio


def test_fine_tune_required_for_hybrid_strategies_a11_11():
    """
    A11-11: hybrid_moderate/aggressive declare fine_tune=True, so
    optimize() must call fine_tune() on the pruned model before
    quantization — not treat it as a side extra the caller opts into
    separately. When that mandatory step comes back NOT_AVAILABLE or
    ERROR (no archive configured here — this sandbox has none, verified),
    it must be recorded in result["errors"] and overall_success must be
    False — a required step that didn't happen is not a silent pass.

    fast_int8_conv_s1/s2 and compact_prune don't declare fine_tune at all
    (no pruning+quantization combination, or no pruning), so they must be
    completely unaffected — no fine_tune_result, no related error.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    from meta_optimizer.amazon_robotics import OptimizationPipeline, PREDEFINED_STRATEGIES

    assert PREDEFINED_STRATEGIES["hybrid_moderate"].fine_tune is True
    assert PREDEFINED_STRATEGIES["aggressive"].fine_tune is True
    assert PREDEFINED_STRATEGIES["fast_int8"].fine_tune is False
    assert PREDEFINED_STRATEGIES["compact_prune"].fine_tune is False
    assert PREDEFINED_STRATEGIES["fast_int8_conv_s1"].fine_tune is False
    assert PREDEFINED_STRATEGIES["fast_int8_conv_s2"].fine_tune is False

    pipeline = OptimizationPipeline()
    model = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.ReLU(), torch.nn.Linear(50, 10))
    test_input = torch.randn(4, 100)

    result = pipeline.optimize(model=model, test_input=test_input, strategy="hybrid_moderate")
    assert result["fine_tune_result"] is not None
    assert result["fine_tune_result"]["status"] == "NOT_AVAILABLE"
    assert any("Fine-tuning NOT_AVAILABLE" in e for e in result["errors"]), result["errors"]
    assert result["overall_success"] is False
    # Pruning/quantization themselves still ran and succeeded independently.
    assert result["pruning_result"]["success"]
    assert result["quantization_result"]["success"]

    result = pipeline.optimize(model=model, test_input=test_input, strategy="fast_int8_conv_s1")
    assert result["fine_tune_result"] is None
    assert result["overall_success"] is True, result["errors"]


def test_fine_tune_not_available_without_archive():
    """
    Step 7 #5: fine-tuning's Imagenette2-160 archive is an optional data
    dependency (the same archive E1's val-split check uses) — its absence
    must be an explicit NOT_AVAILABLE status, never a silent pass and never
    an exception that takes down a caller that doesn't need it. This
    sandbox has no such archive (verified: none exists anywhere on this
    machine), so this test exercises exactly that path — not the "happy
    path" with a real archive, which remains CLAIMED/untested here.
    """
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics import FineTuneConfig, fine_tune

    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    teacher = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    result = fine_tune(model, teacher, FineTuneConfig(archive_path="/nonexistent/imagenette2-160.tgz"))
    assert result.status == "NOT_AVAILABLE"
    assert result.accuracy_before is None
    assert result.accuracy_after is None

    # No archive_path configured at all must behave the same way, not crash.
    result = fine_tune(model, teacher, FineTuneConfig())
    assert result.status == "NOT_AVAILABLE"


def test_build_imagenet_label_map_translates_synsets_to_imagenet_indices():
    """
    fine_tune() must never replace the model's classifier head — Imagenette's
    own 10 folder labels are translated to their real ImageNet-1000 index
    instead, so loss/accuracy stay over the full 1000-class output (the
    same output space E1's own val check uses). This is a direct unit
    check on that translation table, independent of the full archive
    pipeline.
    """
    pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics.fine_tuning import _build_imagenet_label_map

    classes = [
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
    ]
    label_map = _build_imagenet_label_map(classes)
    assert label_map.tolist() == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    with pytest.raises(ValueError, match="unrecognized Imagenette synset"):
        _build_imagenet_label_map(["n01440764", "not-a-real-synset"])


def test_fine_tune_preserves_classifier_head_and_imagenet_labels(tmp_path):
    """
    End-to-end happy path on a small, locally-generated fake
    Imagenette2-160-shaped archive (real synset folder names, tiny
    synthetic images) — not NOT_AVAILABLE, the actual training/eval code
    path. Verifies: (1) the classifier head is untouched (still 1000
    classes) after fine_tune(), and (2) training/eval complete without a
    label-index error in the accuracy computation (the only place labels
    are used at all — see test_distillation_loss_ignores_labels for the
    training loss itself) — a fake dataset is enough to prove the
    plumbing, the real accuracy numbers remain CLAIMED until a real
    archive is used on XMG.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import tarfile
    from PIL import Image
    from meta_optimizer.amazon_robotics import FineTuneConfig, fine_tune

    synsets = ["n01440764", "n02102040", "n02979186"]
    root = tmp_path / "imagenette2-160"
    for synset in synsets:
        d = root / "train" / synset
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (160, 160), color=(i * 10, i * 20, i * 30)).save(d / f"{synset}_{i}.JPEG")

    archive_path = tmp_path / "imagenette2-160.tgz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(root, arcname="imagenette2-160")
    sha256 = __import__("hashlib").sha256(archive_path.read_bytes()).hexdigest()

    # A cheap 1000-class classifier head — fine_tune() only cares that it's
    # an nn.Module accepting [B,3,224,224] and producing [B,1000] logits.
    def build_model():
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, stride=4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 1000),
        )

    model = build_model()
    teacher = build_model()

    result = fine_tune(model, teacher, FineTuneConfig(
        archive_path=str(archive_path), expected_sha256=sha256,
        epochs=1, holdout_fraction=0.34, batch_size=4,
    ))
    assert result.status == "VERIFIED", result.error
    assert model[-1].out_features == 1000
    assert result.accuracy_before is not None
    assert result.accuracy_after is not None


def test_distillation_loss_ignores_labels():
    """
    _distillation_loss's signature is (student_logits, teacher_logits,
    temperature) — there is no parameter through which a label, hard or
    otherwise, could reach it. Correctness checked against a manual
    KL(softmax(teacher) || softmax(student)) computation at T=1.
    """
    torch = pytest.importorskip("torch")
    from meta_optimizer.amazon_robotics.fine_tuning import _distillation_loss
    import inspect

    params = list(inspect.signature(_distillation_loss).parameters)
    assert params == ["student_logits", "teacher_logits", "temperature"]

    torch.manual_seed(0)
    student_logits = torch.randn(8, 1000)
    teacher_logits = torch.randn(8, 1000)

    loss = _distillation_loss(student_logits, teacher_logits, temperature=1.0)

    teacher_probs = torch.softmax(teacher_logits, dim=1)
    manual = (teacher_probs * (torch.log(teacher_probs) - torch.log_softmax(student_logits, dim=1))).sum(dim=1).mean()
    assert torch.allclose(loss, manual, atol=1e-5), (loss.item(), manual.item())


def test_fine_tune_training_result_unaffected_by_garbage_labels(tmp_path, monkeypatch):
    """
    The training loss is pure distillation (student vs. frozen teacher
    logits) — Imagenette's synset->ImageNet-index labels are used only
    for accuracy_before/after, never for anything the optimizer's
    gradient depends on. Proven here by swapping in a deliberately WRONG
    synset->index mapping for a second run and checking the trained
    weights come out bit-identical to the first (same archive, same seed,
    same initial weights) — if labels leaked into the loss anywhere, a
    wrong mapping would change the loss values and the weights would
    diverge. accuracy_before/after legitimately differ between the two
    runs (different "correct" index for the same images) — only the
    trained weights are asserted equal.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import copy
    import tarfile
    from PIL import Image
    from meta_optimizer.amazon_robotics import FineTuneConfig, fine_tune
    import meta_optimizer.amazon_robotics.fine_tuning as fine_tuning_module

    synsets = ["n01440764", "n02102040", "n02979186"]
    root = tmp_path / "imagenette2-160"
    for synset in synsets:
        d = root / "train" / synset
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (160, 160), color=(i * 10, i * 20, i * 30)).save(d / f"{synset}_{i}.JPEG")

    archive_path = tmp_path / "imagenette2-160.tgz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(root, arcname="imagenette2-160")
    sha256 = __import__("hashlib").sha256(archive_path.read_bytes()).hexdigest()

    torch.manual_seed(0)
    model_a = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, stride=4), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1000),
    )
    model_b = copy.deepcopy(model_a)
    teacher = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, stride=4), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1000),
    )
    config = FineTuneConfig(
        archive_path=str(archive_path), expected_sha256=sha256,
        epochs=1, holdout_fraction=0.34, batch_size=4, seed=0,
    )

    result_a = fine_tune(model_a, copy.deepcopy(teacher), config)
    assert result_a.status == "VERIFIED", result_a.error

    garbage_map = {synset: 999 - i for i, synset in enumerate(synsets)}
    monkeypatch.setattr(fine_tuning_module, "_IMAGENETTE_SYNSET_TO_IMAGENET_IDX", garbage_map)
    result_b = fine_tune(model_b, copy.deepcopy(teacher), config)
    assert result_b.status == "VERIFIED", result_b.error

    for (name_a, p_a), (name_b, p_b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), f"{name_a} diverged — labels leaked into the training loss"


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
