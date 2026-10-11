# CLAUDE.md

This file provides guidance to AI assistants (such as Claude) working in this repository.

## Repository Overview

**META-OPTIMIZER v7.0** is a Python toolkit of three loosely-coupled subsystems, delivered
as an installable package under `src/meta_optimizer/`:

1. **Temporal Knowledge Graph (TKG) memory** — graph + vector hybrid storage for
   cross-session knowledge, with temporal queries ("what changed since last week?").
2. **Real model optimization pipeline** — actual PyTorch quantization (dynamic INT8 via
   `torchao.quantization.quantize_()`; static/calibrated INT8 via the legacy
   `torch.ao.quantization` eager-mode API, deprecated upstream but not yet migrated to
   torchao's PT2E flow; INT4 is simulated by rounding weights to 16 levels and storing
   them back as float, not true INT4 storage/kernels — see `quantizer.py::_int4_weight_only`
   for why: torchao's real INT4 tensor formats all require a CUDA GPU or the GPU-only
   `mslk` package, so there's no portable INT4 kernel path for this CPU-first library),
   pruning (structured/unstructured), and ONNX/TensorRT export, not placeholders.
3. **Workflow orchestration** — a small LangGraph-inspired explicit state machine with
   conditional routing, retry/error recovery, and parallel strategy execution.

There is no CLI, server, or entry point — this is a library intended to be imported by
another project. See `README.md` for the full usage guide and quick-start examples.

## Project Structure

```
src/
└── meta_optimizer/
    ├── __init__.py
    ├── temporal_memory/          # TKG Memory System
    │   ├── tkg_engine.py         # TemporalKnowledgeGraph — main public interface
    │   ├── graph_store.py        # NetworkXGraphStore, GraphNode, GraphEdge
    │   ├── vector_store.py       # VectorDocument, create_vector_store
    │   ├── hybrid_retriever.py   # HybridRetriever
    │   └── temporal_query.py     # TemporalQuery
    ├── amazon_robotics/          # Model Optimization
    │   ├── quantizer.py          # RealQuantizer — INT8 / simulated INT4
    │   ├── pruner.py             # RealPruner
    │   ├── exporter.py           # ModelExporter — ONNX/TensorRT
    │   └── real_optimizer.py     # OptimizationPipeline
    └── orchestration/            # Workflow Engine
        ├── workflow_engine.py    # WorkflowEngine and MetaOptimizerWorkflow
        ├── state_manager.py      # Checkpointing
        ├── error_handler.py      # Retry and error records
        └── parallel_executor.py  # Concurrent strategy execution
```

The project is installed as the `meta_optimizer` package. Import public symbols from
subpackage exports, for example `from meta_optimizer.temporal_memory import TemporalKnowledgeGraph`.


## Language, Runtime, and Dependencies

- **Language**: Python 3.10+.
- **Packaging**: `pyproject.toml` with a `src/meta_optimizer` package layout.
- **Core dependency**: `networkx`.
- **Optional extras**:
  - `optimization`: PyTorch, torchao, and NumPy for quantization and pruning.
  - `onnx`: optimization dependencies plus ONNX and ONNX Runtime.
  - `chromadb`: ChromaDB and sentence-transformers. The default vector backend remains
    in-memory; explicitly requesting ChromaDB without this extra raises `ImportError`.
  - TensorRT is platform-specific and must be installed separately.
- Install with `pip install -e .`; use extras such as `pip install -e ".[onnx,test]"`
  for development and smoke testing.


## Code Conventions

Follow the patterns already established in `src/meta_optimizer/`:

- `from __future__ import annotations` at the top of every module.
- Full type hints on public functions/methods (`Optional`, `Dict[str, Any]`, etc.).
- Config and result types are `@dataclass`es, but only **result** dataclasses carry a
  `to_dict()` method for serialization (see `QuantizationResult` in `quantizer.py`,
  `PruningResult` in `pruner.py`, `ExportResult` in `exporter.py`, `PipelineResult` in
  `real_optimizer.py`). Config dataclasses (`QuantizationConfig`, `PruningConfig`,
  `ExportConfig`, `OptimizationStrategy`) are plain field containers with no `to_dict()` —
  follow this config/result split for new dataclasses rather than plain dicts.
- Enums for fixed categories (`EntityType`, `NodeType`) subclass `str, Enum` so values
  serialize cleanly to JSON and compare equal to plain strings.
- Google-style docstrings (`Args:` / `Returns:`) on public methods; module-level
  docstrings describe the file's role in the subsystem.
- `# ---- Section ----` comment dividers group related methods within a class (see
  `tkg_engine.py`).
- Persistence is explicit and JSON-based (`_save()`/`_load()` writing to a
  `storage_path`), not a database — keep new persistence consistent with this unless a
  real backend is deliberately introduced.
- Public API is curated via each subpackage's `__init__.py` `__all__` list — add new
  public symbols there when you add them to a module.

General principles regardless of stack:
- Prefer clarity over cleverness
- Keep functions small and focused on a single responsibility
- Avoid over-engineering — build only what is needed now
- Do not add error handling for scenarios that cannot occur
- Do not add comments unless the logic is non-obvious

## Testing

The automated smoke suite is in `tests/test_smoke.py` and exercises the README Quick
Start paths. GitHub Actions installs the package in a clean environment, verifies the
core imports, installs the ONNX/test extras, and runs `pytest`.

Run locally with:

```bash
pip install -e ".[onnx,test]"
pytest
```


## Git Workflow

### Branch naming
- AI-assisted work: `claude/<short-description>-<session-id>`
- Features: `feature/<description>`
- Bug fixes: `fix/<description>`
- This repo has no long-lived `main`/`master` — check `git remote show origin` for the
  current default branch (HEAD) before branching, since it may itself be a
  `claude/...`-named branch left over from prior AI-assisted work.

### Commit messages
Write clear, imperative-style commit messages:
```
Add authentication middleware
Fix null pointer in user service
Refactor database connection pool
```

### Pushing changes
Always push with tracking:
```bash
git push -u origin <branch-name>
```

### Pull requests
- Keep PRs focused on a single concern
- Include a clear description of what changed and why

## Security

- Never commit secrets, API keys, or credentials
- `.gitignore` already excludes Python build artifacts, virtual envs, IDE files, and
  optimizer output artifacts (`*.onnx`, `*.trt`, `*.engine`) — extend it rather than
  committing generated/exported models
- Validate all user input at system boundaries

## AI Assistant Guidelines

When working in this repository, AI assistants should:

1. **Read before editing** — always read a file before modifying it
2. **Stay in scope** — only change what was requested; avoid unsolicited refactors
3. **Keep changes minimal** — prefer editing existing files over creating new ones
4. **Confirm before destructive actions** — force pushes, file deletions, branch resets require explicit user approval
5. **Use the designated branch** — all work goes to the branch specified in the task context
6. **Match existing conventions** — see Code Conventions above (dataclasses, type hints,
   `__all__`-curated exports, docstring style) rather than introducing new patterns
7. **Update this file** — when new conventions, tools, dependencies, or structure are established, update the relevant section of this file

## Updating This File

As the project grows, keep this file in sync with reality:
- Update `pyproject.toml` and this file when dependencies or extras change.
- Keep README Quick Start examples covered by `tests/test_smoke.py`.
- Record material CI/CD changes.
- Record new subsystems added alongside `temporal_memory`, `amazon_robotics`, and `orchestration`.
