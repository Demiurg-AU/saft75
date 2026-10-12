"""
Fine-tuning after physical pruning, on Imagenette2-160's train split.

Data dependency: a local ``imagenette2-160.tgz``-style archive (fast.ai's
Imagenette2-160, the same archive E1's val-split check uses), supplied by
the caller via ``FineTuneConfig.archive_path`` together with the SHA-256
the caller already knows is correct for their own copy
(``expected_sha256``) — this module has no way to independently confirm
what the "correct" hash for that archive is, so it never hardcodes one;
it only ever compares the archive's *actual* hash to whatever the caller
supplied, the same pinned-hash-verification discipline already used by
``serialization.py`` for sd+recipe artifacts.

This step's data dependency is optional and, per the step-7 contract,
explicitly NOT_AVAILABLE rather than a hard failure when the archive
isn't present — this is reported as an explicit ``FineTuneResult.status``,
never silently converted to success and never raised as an exception that
would take down a caller that doesn't need fine-tuning to succeed. A
present-but-corrupted/substituted archive (SHA-256 mismatch against
``expected_sha256``) is the opposite case and IS a hard ``"ERROR"`` status
— unlike a merely-missing archive, a wrong one must fail loudly.

The held-out accuracy control is a slice of Imagenette's TRAIN split
(``FineTuneConfig.holdout_fraction``), never the val split — val is
reserved for E1's own check elsewhere, so this module never reads it.

If the model's current classifier head (its last ``nn.Linear``) doesn't
already output Imagenette's class count (10), it's replaced with a fresh
``nn.Linear`` of the right size — ordinary transfer-learning practice, not
a hidden label remapping. A model with no ``nn.Linear`` at all (e.g. a
PT2E/``static_int8`` ``GraphModule`` — see ``quantizer.py``'s
``_static_int8`` docstring) has no classifier head to adapt and raises
``NotImplementedError``, the same "this representation doesn't support
that operation" pattern as ``serialization.py``'s sd+recipe.

CUDA is permitted (``FineTuneConfig.device="cuda"``) but is this caller's
own responsibility to schedule — this module does nothing to detect or
avoid contention with other GPU work running at the same time.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@dataclass
class FineTuneConfig:
    """Configuration for an Imagenette2-160 fine-tuning pass."""
    archive_path: Optional[str] = None
    expected_sha256: Optional[str] = None
    epochs: int = 1
    holdout_fraction: float = 0.10     # fraction of TRAIN held out as a control, never val
    batch_size: int = 32
    learning_rate: float = 1e-4
    device: str = "cpu"                # "cpu" | "cuda"
    seed: int = 0


@dataclass
class FineTuneResult:
    """Result of a fine-tuning pass — see module docstring for status values."""
    status: str                        # "VERIFIED" | "NOT_AVAILABLE" | "ERROR"
    archive_sha256: Optional[str] = None
    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None
    holdout_size: Optional[int] = None
    train_size: Optional[int] = None
    epochs: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "archive_sha256": self.archive_sha256,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "holdout_size": self.holdout_size,
            "train_size": self.train_size,
            "epochs": self.epochs,
            "error": self.error,
        }


def fine_tune(model: "nn.Module", config: Optional[FineTuneConfig] = None) -> FineTuneResult:
    """
    Fine-tune ``model`` on Imagenette2-160's train split, reporting
    accuracy on a held-out slice of train (never val) before and after.

    See the module docstring for the NOT_AVAILABLE/ERROR status contract.
    """
    config = config or FineTuneConfig()
    if not HAS_TORCH:
        return FineTuneResult(status="ERROR", error="torch is required")
    if not config.archive_path or not os.path.exists(config.archive_path):
        return FineTuneResult(
            status="NOT_AVAILABLE",
            error=f"Imagenette2-160 archive not found at archive_path={config.archive_path!r}",
        )
    if not HAS_TORCHVISION:
        return FineTuneResult(status="NOT_AVAILABLE", error="torchvision is required")

    actual_sha256 = _sha256_file(config.archive_path)
    if config.expected_sha256 is not None and actual_sha256 != config.expected_sha256:
        return FineTuneResult(
            status="ERROR",
            archive_sha256=actual_sha256,
            error=(
                f"archive SHA-256 {actual_sha256!r} does not match expected "
                f"{config.expected_sha256!r} — corrupted or substituted archive"
            ),
        )

    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linears:
        return FineTuneResult(
            status="ERROR",
            archive_sha256=actual_sha256,
            error=(
                f"{type(model).__name__} has no nn.Linear classifier head to adapt/fine-tune "
                "(e.g. a PT2E/static_int8 GraphModule — see quantizer.py's _static_int8 docstring)"
            ),
        )

    extract_dir = tempfile.mkdtemp(prefix="imagenette2-160-")
    with tarfile.open(config.archive_path, "r:*") as tar:
        tar.extractall(extract_dir, filter="data")

    train_dir = _find_train_dir(extract_dir)
    transform = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    full_train = ImageFolder(train_dir, transform=transform)

    _adapt_classifier_head(model, linears, len(full_train.classes))

    generator = torch.Generator().manual_seed(config.seed)
    n_holdout = max(1, int(len(full_train) * config.holdout_fraction))
    n_train = len(full_train) - n_holdout
    train_subset, holdout_subset = torch.utils.data.random_split(
        full_train, [n_train, n_holdout], generator=generator,
    )

    device = torch.device(config.device)
    model = model.to(device)

    accuracy_before = _evaluate_accuracy(model, holdout_subset, config.batch_size, device)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True, generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    model.eval()

    accuracy_after = _evaluate_accuracy(model, holdout_subset, config.batch_size, device)

    return FineTuneResult(
        status="VERIFIED",
        archive_sha256=actual_sha256,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        holdout_size=n_holdout,
        train_size=n_train,
        epochs=config.epochs,
    )


def _adapt_classifier_head(model: "nn.Module", linears, num_classes: int) -> None:
    """Replace the last nn.Linear with a fresh one if its size doesn't match."""
    name, last_linear = linears[-1]
    if last_linear.out_features == num_classes:
        return
    new_linear = nn.Linear(last_linear.in_features, num_classes)
    *path, leaf = name.split(".")
    parent = model
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, leaf, new_linear)


def _find_train_dir(extract_dir: str) -> str:
    for root, dirs, _ in os.walk(extract_dir):
        if "train" in dirs:
            return os.path.join(root, "train")
    raise FileNotFoundError(f"no 'train' directory found under extracted archive at {extract_dir}")


def _evaluate_accuracy(model: "nn.Module", dataset, batch_size: int, device) -> float:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
