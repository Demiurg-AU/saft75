"""
S1 (in-process) artifact serialization: state_dict + recipe.

Format: the model's ``state_dict()`` saved via ``torch.save`` (reloaded
with ``weights_only=True``), plus a recipe recording what would be
needed to reproduce the transform that produced it — pinned
torch/torchvision/torchao versions, a SHA-256 of the transform
function's own source, and a SHA-256 of the state_dict bytes
themselves. ``load_sd_recipe`` verifies all three before loading:
state_dict integrity (always), the pinned versions against what's
currently installed (unless ``strict=False``), and the transform
source hash — so reloading with corrupted bytes, a different library
version, or a different transform than the one that produced the
weights fails loudly instead of silently loading mismatched tensors.

This format applies to S1 (in-process) quantization that keeps the
original module structure's ``state_dict()`` shape — torchao's
``quantize_()`` family (dynamic, weight-only, with or without a
structural transform like ``swap_conv2d_1x1_to_linear`` first).

It does NOT apply to PT2E/``static_int8`` (see ``quantizer.py``'s
``_static_int8`` docstring): going through ``torch.export`` produces
internal names like ``_frozen_param0`` that don't correspond to any
``nn.Module``'s ``state_dict()`` shape at all, regardless of cleanup.
That model's only persistence path is re-export + ``torch.export.save``,
which is an S2/ONNX concern, not S1 — ``save_sd_recipe`` on such a model
raises ``NotImplementedError`` rather than silently producing an
artifact nothing can load.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision
except ImportError:
    torchvision = None

try:
    import torchao
except ImportError:
    torchao = None


def _versions() -> Dict[str, str]:
    versions = {"torch": torch.__version__}
    if torchvision is not None:
        versions["torchvision"] = torchvision.__version__
    if torchao is not None:
        versions["torchao"] = torchao.__version__
    return versions


def _transform_source_sha256(transform_fn: Callable[["nn.Module"], "nn.Module"]) -> str:
    return hashlib.sha256(inspect.getsource(transform_fn).encode()).hexdigest()


@dataclass
class SdRecipe:
    """A saved state_dict plus the recipe needed to verify/reproduce its transform."""
    state_dict_bytes: bytes
    recipe: Dict[str, Any]

    def save(self, sd_path: str, recipe_path: str) -> None:
        with open(sd_path, "wb") as fh:
            fh.write(self.state_dict_bytes)
        with open(recipe_path, "w", encoding="utf-8") as fh:
            json.dump(self.recipe, fh, indent=2)

    @property
    def state_dict_sha256(self) -> str:
        return hashlib.sha256(self.state_dict_bytes).hexdigest()


def save_sd_recipe(
    model: "nn.Module",
    transform_fn: Callable[["nn.Module"], "nn.Module"],
    strategy_name: str,
) -> SdRecipe:
    """
    Save an already-transformed (quantized) model as state_dict + recipe.

    Args:
        model: the transformed model to save.
        transform_fn: the function that produced ``model`` from a float
            model — its source is hashed into the recipe; ``load_sd_recipe``
            verifies this hash before loading.
        strategy_name: recorded in the recipe for provenance.

    Raises:
        NotImplementedError: if ``model`` is a ``torch.export``/PT2E
            ``GraphModule`` — this format doesn't apply (see module docstring).
    """
    if isinstance(model, torch.fx.GraphModule):
        raise NotImplementedError(
            "sd+recipe is not defined for PT2E/torch.export models "
            "(GraphModule state_dict keys don't correspond to any "
            "nn.Module's structure); the format for this model is ONNX"
        )

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    sd_bytes = buf.getvalue()
    recipe = {
        "area": "S1",
        "strategy_name": strategy_name,
        "versions": _versions(),
        "transform_source_sha256": _transform_source_sha256(transform_fn),
        "state_dict_sha256": hashlib.sha256(sd_bytes).hexdigest(),
    }
    return SdRecipe(state_dict_bytes=sd_bytes, recipe=recipe)


def load_sd_recipe(
    sd_bytes: bytes,
    recipe: Dict[str, Any],
    build_meta_skeleton: Callable[[], "nn.Module"],
    transform_fn: Callable[["nn.Module"], "nn.Module"],
    strict: bool = True,
) -> "nn.Module":
    """
    Rebuild a quantized model from a saved sd+recipe artifact.

    Args:
        sd_bytes: bytes previously produced by ``SdRecipe.state_dict_bytes``.
        recipe: the matching recipe dict.
        build_meta_skeleton: produces a module on ``torch.device("meta")``
            whose parameter names/shapes already match ``sd_bytes`` — for a
            transform that only changes weight dtype/subclass (e.g. plain
            weight-only quantization), this is just the original float
            architecture; for one that also changes module structure (e.g.
            ``swap_conv2d_1x1_to_linear``), the caller must apply that
            structural change to the meta skeleton too. This utility only
            handles the save/verify/load mechanics, not transform-specific
            structural knowledge.
        transform_fn: the original transform function, used only to verify
            its source hash matches the recipe's recorded hash.
        strict: when True (default), the recipe's recorded
            torch/torchvision/torchao versions must match the versions
            installed now, or this raises — a quantized artifact loaded
            under different library versions is not guaranteed to mean the
            same thing. Pass ``strict=False`` to load anyway (the caller is
            then asserting they've checked compatibility some other way).

    Raises:
        ValueError: if ``transform_fn``'s source doesn't match the recipe's
            hash (not the transform that produced the saved weights); if
            ``sd_bytes``'s SHA-256 doesn't match the recipe's recorded hash
            (corrupted or substituted state_dict — always checked,
            regardless of ``strict``); or, when ``strict=True``, if the
            recorded versions don't match the currently installed ones.
    """
    got_sd_hash = hashlib.sha256(sd_bytes).hexdigest()
    if got_sd_hash != recipe["state_dict_sha256"]:
        raise ValueError(
            f"state_dict SHA-256 {got_sd_hash!r} does not match the "
            f"recipe's recorded hash {recipe['state_dict_sha256']!r} — "
            "the state_dict bytes are corrupted or do not match this recipe"
        )

    if strict:
        current = _versions()
        recorded = recipe["versions"]
        mismatched = {
            k: (recorded.get(k), current.get(k))
            for k in recorded
            if recorded.get(k) != current.get(k)
        }
        if mismatched:
            raise ValueError(
                f"recipe versions do not match the current environment: {mismatched} "
                "(recorded, current) — pass strict=False to load anyway"
            )

    got_hash = _transform_source_sha256(transform_fn)
    if got_hash != recipe["transform_source_sha256"]:
        raise ValueError(
            f"transform_fn source hash {got_hash!r} does not match the "
            f"recipe's recorded hash {recipe['transform_source_sha256']!r} — "
            "this is not the same transform that produced the saved weights"
        )

    skeleton = build_meta_skeleton()
    state_dict = torch.load(io.BytesIO(sd_bytes), weights_only=True)
    skeleton.load_state_dict(state_dict, assign=True)
    return skeleton
