# META-OPTIMIZER v7.0

**AI System for Temporal Memory, Model Optimization and Workflow Orchestration**

## 🚀 What's New in v7.0

### Three Major Upgrades (Based on 2025 Best Practices)

#### #1: Temporal Knowledge Graph Memory
- Target: accuracy boost from temporal memory (reference point: Mem0 benchmark, not reproduced here)
- Graph + Vector hybrid storage
- Temporal tracking (when facts were learned)
- Cross-session synthesis
- Relationship modeling
- Edge invalidation

**Replaces:** Simple JSON storage → NetworkX graph

#### #2: Real Model Optimization Pipeline
- PyTorch INT8 quantization and simulated INT4 weight rounding
- **REAL pruning** (structured/unstructured)
- Target: 3-5x inference speedup (benchmark pending)
- Target: 4-10x memory reduction (benchmark pending)
- ONNX/TensorRT export
- Designed for edge deployment (INT8 + ONNX export)

**Replaces:** Placeholder optimizations → Actual implementations

#### #3: LangGraph-Inspired Orchestration
- Explicit state machines
- Conditional routing
- Error recovery mechanisms
- Parallel strategy execution
- Visual workflow debugging

**Replaces:** Implicit chains → Explicit workflows

---

## 📁 Architecture

```
src/
└── meta_optimizer/
    ├── __init__.py
    ├── temporal_memory/          # TKG Memory System
    │   ├── tkg_engine.py         # Main interface
    │   ├── graph_store.py        # Graph storage (NetworkX)
    │   ├── vector_store.py       # Embeddings (ChromaDB/in-memory)
    │   ├── hybrid_retriever.py   # Graph + Vector queries
    │   └── temporal_query.py     # Time-aware queries
    ├── amazon_robotics/          # Model Optimization
    │   ├── quantizer.py          # PyTorch INT8 / simulated INT4
    │   ├── pruner.py             # Structured/unstructured pruning
    │   ├── exporter.py           # ONNX/TensorRT
    │   └── real_optimizer.py     # Integrated pipeline
    └── orchestration/            # Workflow Engine
        ├── workflow_engine.py    # State machine + MetaOptimizerWorkflow
        ├── state_manager.py      # State persistence
        ├── error_handler.py      # Recovery strategies
        └── parallel_executor.py  # Multi-strategy execution
```

Installed as the `meta_optimizer` package using a `src` layout and `pyproject.toml`.

---

## 🎯 Quick Start

### 1. Install

```bash
pip install -e .                    # core: temporal_memory + orchestration (networkx only)
pip install -e ".[optimization]"    # + PyTorch quantization / pruning
pip install -e ".[onnx]"            # + ONNX export and onnxruntime validation
pip install -e ".[chromadb]"        # enables vector_backend="chromadb"; default remains in-memory
pip install -e ".[all]"             # everything above + pytest
```

TensorRT export requires NVIDIA's platform-specific `tensorrt` package; install it separately.


### 2. Use Temporal Knowledge Graph

```python
from meta_optimizer.temporal_memory import TemporalKnowledgeGraph, EntityType

# Initialize TKG
tkg = TemporalKnowledgeGraph()

# Add knowledge
tkg.add_knowledge(
    text="Alexander optimized MobileNet with INT8 quantization achieving 4.2x speedup",
    entity_type=EntityType.OPTIMIZATION,
    metadata={
        "model": "MobileNet",
        "method": "INT8",
        "speedup": 4.2
    }
)

# Search knowledge
results = tkg.search("optimizations for MobileNet")

# Temporal queries
recent = tkg.what_changed_since(days_ago=7)
last_opt = tkg.when_was_last("model optimization")
```

### 3. Use Real Optimization Pipeline

```python
from meta_optimizer.amazon_robotics import OptimizationPipeline
import torch

# Initialize pipeline
pipeline = OptimizationPipeline()

# Create test model
model = torch.nn.Sequential(
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 10)
)

# Dummy input
test_input = torch.randn(1, 100)

# Run optimization
results = pipeline.optimize(
    model=model,
    test_input=test_input,
    strategy="hybrid_moderate"  # or "fast_int8", "compact_prune", "aggressive"
)

# Check results
print(f"Speedup: {results['final_metrics']['latency_improvement']}")
print(f"Compression: {results['final_metrics']['size_reduction']}")
print(f"Success: {results['overall_success']}")
```

### 4. Use Workflow Orchestration

```python
from meta_optimizer.orchestration import MetaOptimizerWorkflow

# Initialize workflow
workflow = MetaOptimizerWorkflow()

# Run complete pipeline
result = workflow.run(
    request="Optimize model for edge deployment",
    request_type="optimize_model",
    model=model,
    test_input=test_input,
    max_retries=3,
    use_parallel=True,
    strategies=["fast_int8", "compact_prune", "hybrid_moderate"]
)

# Check result
print(result['success'])
print(result['optimization_results'])
```

---

## 📊 Expected Performance

### TKG Memory
- Target: improved multi-turn accuracy (benchmark pending)
- Target: 90% latency reduction vs full-context re-processing (benchmark pending)
- **Cross-session synthesis** — "what changed since last week?"
- **Temporal queries** — "when did we last optimize X?"

### Real Optimization
- Target: 3-5x inference speedup (benchmark pending)
- Target: 4-10x memory reduction (benchmark pending)
- Target: <3% accuracy drop (validation harness pending)
- **<10ms latency target** (edge deployment)
- **<100MB size target** (edge deployment)

### Workflow Orchestration
- **Explicit error handling** — no silent failures
- **Automatic retry** — 3 attempts with different strategies
- **Parallel execution** — test multiple strategies simultaneously
- **Graceful degradation** — fallback to baseline on failure

---

## 🔧 Configuration

### TKG Storage Backend

```python
# Option 1: In-memory (development)
tkg = TemporalKnowledgeGraph(backend="networkx")

# Option 2: Persistent storage
tkg = TemporalKnowledgeGraph(
    backend="networkx",
    storage_path="/path/to/.claude/data/tkg"
)
```

### Optimization Strategies

```python
# Predefined strategies
strategies = {
    "fast_int8": "Quick INT8 quantization",
    "compact_prune": "30% structured pruning",
    "hybrid_moderate": "INT8 + 20% pruning",
    "aggressive": "INT8 + 50% pruning"
}

# Custom strategy
from meta_optimizer.amazon_robotics import OptimizationStrategy

custom = OptimizationStrategy(
    name="Custom Strategy",
    use_quantization=True,
    quantization_type="dynamic_int8",
    use_pruning=True,
    pruning_amount=0.4,
    target_latency_ms=5.0
)
```

---

## 🧪 Testing

```bash
pip install -e ".[onnx,test]"
pytest
```

`tests/test_smoke.py` executes the Quick Start examples above. Tests that need
PyTorch skip automatically when only the core package is installed.

---

## 📈 Migration from v6.0

### Old JSON → New TKG

```python
tkg = TemporalKnowledgeGraph()

# Migrate from old learning_data.json
migration_stats = tkg.migrate_from_json(
    json_path="/home/claude/.claude/tools/learning_data.json"
)

print(f"Migrated {migration_stats['sessions']} sessions")
print(f"Migrated {migration_stats['lessons']} lessons")
```

---

## 🎯 Production Deployment Checklist

- [ ] Install all dependencies
- [ ] Configure persistent storage path for TKG
- [ ] Test optimization pipeline with real models
- [ ] Set up workflow monitoring
- [ ] Configure error alerting
- [ ] Backup TKG data regularly

---

## 📚 References

### Industry Research (2025)

1. **Mem0**: "+26% accuracy vs. OpenAI Memory on the LOCOMO benchmark" — https://mem0.ai/research (arXiv:2504.19413)
2. **Edge AI Survey**: PRISMA-aligned systematic review of DL deployment on embedded hardware — https://www.mdpi.com/2079-9292/14/24/4877
3. **QAP (quantization-aware pruning)**: Hawks et al. (2021), *Frontiers in AI* — ~50x reduction in bit-operations (BOPs), not storage size — https://doi.org/10.3389/frai.2021.676564
4. **AI Agent Framework Guide**: Langflow, "The Complete Guide to Choosing an AI Agent Framework in 2025" — https://www.langflow.org/blog/the-complete-guide-to-choosing-an-ai-agent-framework-in-2025
5. **Temporal KG**: Zep — "A Temporal Knowledge Graph Architecture for Agent Memory" — https://arxiv.org/abs/2501.13956

---

**Version:** 7.0.0
**Status:** Alpha — packaging and CI smoke tests in place, benchmarks pending
