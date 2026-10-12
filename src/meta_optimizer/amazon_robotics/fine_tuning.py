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

The model's classifier head is NEVER replaced — it stays the original
1000-class ImageNet head throughout.

Training uses knowledge distillation, not hard-label cross-entropy: the
loss is ``KL(softmax(teacher_logits/T) || softmax(student_logits/T))``
(``_distillation_loss``) over the full 1000-class output, T=1 by default,
with the pre-pruning FP32 model as the frozen teacher (passed explicitly
via ``fine_tune``'s ``teacher_model`` argument — ``eval()``'d, run under
``torch.no_grad()``, never updated). Hard labels never appear in this
loss at all — Imagenette's own 10 folder labels (``ImageFolder`` assigns
them local indices 0-9, alphabetically by the synset folder name),
translated to their real ImageNet-1000 index via
``_IMAGENETTE_SYNSET_TO_IMAGENET_IDX``, are used *only* for
``accuracy_before``/``accuracy_after`` on the held-out slice, never for
anything the optimizer's gradient depends on. This also means training
needs no notion of which 10 classes Imagenette covers at all — only
evaluating the before/after accuracy numbers does.

Preprocessing is ``MobileNet_V2_Weights.IMAGENET1K_V1.transforms()`` —
the same preset E1 uses (224 center-crop, ImageNet normalization) — for
anything being *measured* (the held-out control). Training is the same
preset's own ``resize_size``/``crop_size``/``mean``/``std`` but with a
random crop and a random horizontal flip substituted for the preset's
deterministic resize+center-crop, since training benefits from those two
augmentations and nothing else is introduced on top of the same
normalization.

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
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torchvision.models import MobileNet_V2_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# Imagenette2-160's 10 synset folder names -> their real index in the
# standard 1000-class ILSVRC ordering (caller-supplied, not independently
# verifiable in this sandbox — see module docstring's archive-hash note
# for the same "no way to confirm independently" situation).
_IMAGENETTE_SYNSET_TO_IMAGENET_IDX: Dict[str, int] = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


@dataclass
class FineTuneConfig:
    """Configuration for an Imagenette2-160 fine-tuning pass."""
    archive_path: Optional[str] = None
    expected_sha256: Optional[str] = None
    epochs: int = 1
    holdout_fraction: float = 0.10     # fraction of TRAIN held out as a control, never val
    batch_size: int = 32
    learning_rate: float = 1e-4
    temperature: float = 1.0           # distillation temperature T
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


def fine_tune(
    model: "nn.Module",
    teacher_model: "nn.Module",
    config: Optional[FineTuneConfig] = None,
) -> FineTuneResult:
    """
    Fine-tune ``model`` via distillation from ``teacher_model`` on
    Imagenette2-160's train split, reporting accuracy on a held-out slice
    of train (never val) before and after.

    Args:
        model: the student — typically the pruned model, about to be
            quantized.
        teacher_model: the frozen reference the student is distilled
            towards — the pre-pruning FP32 model. Never updated; always
            run in ``eval()`` under ``torch.no_grad()``.
        config: see ``FineTuneConfig``.

    See the module docstring for the NOT_AVAILABLE/ERROR status contract
    and the distillation loss (no hard labels).
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

    try:
        extract_dir = tempfile.mkdtemp(prefix="imagenette2-160-")
        with tarfile.open(config.archive_path, "r:*") as tar:
            tar.extractall(extract_dir, filter="data")
        train_dir = _find_train_dir(extract_dir)

        train_transform, eval_transform = _build_transforms()
        # Two ImageFolder instances over the identical directory produce
        # identical `samples`/`classes` ordering — only their transform
        # differs — so splitting by index keeps a holdout sample's image
        # paired with the eval (non-augmented) transform and a train
        # sample's paired with the augmented one.
        meta_dataset = ImageFolder(train_dir)
        label_map = _build_imagenet_label_map(meta_dataset.classes)

        generator = torch.Generator().manual_seed(config.seed)
        n_samples = len(meta_dataset)
        shuffled = torch.randperm(n_samples, generator=generator).tolist()
        n_holdout = max(1, int(n_samples * config.holdout_fraction))
        n_train = n_samples - n_holdout
        train_indices, holdout_indices = shuffled[:n_train], shuffled[n_train:]

        train_dataset = ImageFolder(train_dir, transform=train_transform)
        eval_dataset = ImageFolder(train_dir, transform=eval_transform)
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        holdout_subset = torch.utils.data.Subset(eval_dataset, holdout_indices)

        device = torch.device(config.device)
        model = model.to(device)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        # label_map stays on CPU — labels arrive from the DataLoader on
        # CPU too, so indexing happens before the .to(device) move below.
        # Used only for accuracy_before/after, never for the training loss.

        accuracy_before = _evaluate_accuracy(model, holdout_subset, config.batch_size, device, label_map)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=config.batch_size, shuffle=True, generator=generator,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        model.train()
        for _ in range(config.epochs):
            for images, _labels in train_loader:
                images = images.to(device)
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                optimizer.zero_grad()
                student_logits = model(images)
                loss = _distillation_loss(student_logits, teacher_logits, config.temperature)
                loss.backward()
                optimizer.step()
        model.eval()

        accuracy_after = _evaluate_accuracy(model, holdout_subset, config.batch_size, device, label_map)
    except Exception as exc:
        return FineTuneResult(status="ERROR", archive_sha256=actual_sha256, error=str(exc))

    return FineTuneResult(
        status="VERIFIED",
        archive_sha256=actual_sha256,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        holdout_size=n_holdout,
        train_size=n_train,
        epochs=config.epochs,
    )


def _distillation_loss(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    temperature: float = 1.0,
) -> "torch.Tensor":
    """
    ``KL(softmax(teacher_logits/T) || softmax(student_logits/T))`` over
    the full 1000-class output. Takes only logits and a temperature — no
    labels, hard or otherwise, appear anywhere in its signature or body.
    """
    teacher_probs = torch.softmax(teacher_logits / temperature, dim=1).detach()
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
    return torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def _build_imagenet_label_map(classes: List[str]) -> "torch.Tensor":
    """
    Map ``ImageFolder.classes`` (Imagenette's synset folder names, in the
    local 0..9 index order ``ImageFolder`` assigned them) to their real
    ImageNet-1000 index. Returns a LongTensor ``t`` such that
    ``t[local_label]`` is the correct 1000-class index — raises if any
    folder name isn't one of Imagenette's 10 known synsets, rather than
    silently mapping it to the wrong class.
    """
    try:
        imagenet_indices = [_IMAGENETTE_SYNSET_TO_IMAGENET_IDX[synset] for synset in classes]
    except KeyError as exc:
        raise ValueError(
            f"unrecognized Imagenette synset folder {exc.args[0]!r} — not one of the "
            f"10 known synsets {sorted(_IMAGENETTE_SYNSET_TO_IMAGENET_IDX)}"
        ) from exc
    return torch.tensor(imagenet_indices, dtype=torch.long)


def _build_transforms():
    """
    Returns (train_transform, eval_transform). ``eval_transform`` is
    exactly ``MobileNet_V2_Weights.IMAGENET1K_V1.transforms()`` — the same
    preset E1 uses. ``train_transform`` uses that preset's own
    resize_size/crop_size/mean/std but substitutes a random crop and a
    random horizontal flip for the preset's deterministic resize+center-crop.
    """
    eval_transform = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    train_transform = transforms.Compose([
        transforms.Resize(eval_transform.resize_size[0]),
        transforms.RandomCrop(eval_transform.crop_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std),
    ])
    return train_transform, eval_transform


def _find_train_dir(extract_dir: str) -> str:
    for root, dirs, _ in os.walk(extract_dir):
        if "train" in dirs:
            return os.path.join(root, "train")
    raise FileNotFoundError(f"no 'train' directory found under extracted archive at {extract_dir}")


def _evaluate_accuracy(model: "nn.Module", dataset, batch_size: int, device, label_map: "torch.Tensor") -> float:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), label_map[labels].to(device)
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
