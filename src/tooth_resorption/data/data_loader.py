"""Dataset and DataLoader factories.

Supports two backends:

* ``"synthetic"`` — uses the procedural generator in
  :mod:`tooth_resorption.data.synthetic_data` so the pipeline runs without
  any external data.
* ``"real"`` — loads LabelMe-style JSON annotations from a user-provided
  directory. Each JSON must contain ``imageData`` (base64-encoded JPEG) and
  ``shapes[0].label`` (one of the class names). The real data itself is NOT
  shipped with this repository.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Callable, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from tooth_resorption.config import CLASS_NAMES, NUM_CLASSES, SEED, DataSource
from tooth_resorption.data.preprocessing import build_transforms
from tooth_resorption.data.synthetic_data import generate
from tooth_resorption.logging_utils import get_logger

logger = get_logger(__name__)

Transform = Callable[[Image.Image], torch.Tensor]


def _label_from_string(raw: str) -> int:
    """Best-effort mapping of an annotator-provided label string to a class index.

    Tolerates Turkish accents/case (e.g. ``Temaslı`` / ``temasli``) and the
    English synonym ``resorption``.

    Raises:
        ValueError: If ``raw`` cannot be mapped to any known class.
    """
    low = raw.strip().lower()
    if "temas" in low:
        return 0
    if "ba" in low and ("ms" in low or "gim" in low):
        return 1
    if "rezo" in low or "resor" in low:
        return 2
    raise ValueError(f"Unrecognised class label: {raw!r}")


def _stratified_indices(
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return ``(train_idx, val_idx)`` with per-class proportions preserved.

    Args:
        labels: Per-sample integer class labels.
        val_split: Fraction of the dataset to reserve for validation.
        seed: RNG seed for the shuffle.

    Returns:
        Two disjoint lists of integer indices into ``labels``.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    labels_arr = np.asarray(labels)
    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_split))) if len(cls_idx) > 1 else 0
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _seed_worker(_worker_id: int) -> None:
    """DataLoader worker init: forwards the base seed to NumPy + Python random."""
    import random as _random

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    _random.seed(seed)


class SyntheticToothDataset(Dataset[tuple[torch.Tensor, int]]):
    """In-memory dataset of procedurally generated dental-X-ray-like images."""

    def __init__(
        self,
        n_per_class: int = 16,
        seed: int = SEED,
        transform: Transform | None = None,
    ) -> None:
        self.transform = transform
        self.images, self.labels = generate(n_per_class=n_per_class, seed=seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]
        tensor = self.transform(img) if self.transform is not None else _to_tensor(img)
        return tensor, int(self.labels[idx])


class RealToothDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load tooth-resorption images from LabelMe-style JSON annotations.

    Each JSON file must contain a base64-encoded JPEG under ``imageData`` and
    at least one shape whose first ``label`` matches one of the class names.
    Corrupt or unlabelled files are silently skipped with a warning.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        RuntimeError: If no valid samples were found under ``root``.
    """

    def __init__(self, root: Path | str, transform: Transform | None = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")
        self.transform = transform
        self.samples: list[tuple[Image.Image, int]] = []
        self.labels: list[int] = []

        n_skipped = 0
        for json_path in sorted(self.root.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                image_b64 = payload.get("imageData")
                shapes = payload.get("shapes") or []
                if not image_b64 or not shapes:
                    n_skipped += 1
                    continue
                raw_label = shapes[0].get("label", "")
                label = _label_from_string(raw_label)
                img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
                self.samples.append((img, label))
                self.labels.append(label)
            except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                n_skipped += 1
                logger.warning("Skipping %s (%s)", json_path.name, exc)

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.root}")
        if n_skipped:
            logger.info(
                "Loaded %d samples from %s (%d skipped)",
                len(self.samples), self.root, n_skipped,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.samples[idx]
        tensor = self.transform(img) if self.transform is not None else _to_tensor(img)
        return tensor, int(label)


class _SubsetWithTransform(Dataset[tuple[torch.Tensor, int]]):
    """Apply a transform on top of a raw-PIL view of the underlying dataset."""

    def __init__(
        self,
        base: SyntheticToothDataset | RealToothDataset,
        indices: list[int],
        transform: Transform | None,
    ) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        if isinstance(self.base, SyntheticToothDataset):
            img = self.base.images[real_idx]
            label = self.base.labels[real_idx]
        else:
            img, label = self.base.samples[real_idx]
        tensor = self.transform(img) if self.transform is not None else _to_tensor(img)
        return tensor, int(label)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Fallback PIL → CHW float tensor when no transform is supplied."""
    from torchvision.transforms.functional import to_tensor

    return cast(torch.Tensor, to_tensor(img))


def build_dataloaders(
    source: DataSource = "synthetic",
    data_path: Path | str | None = None,
    n_per_class: int = 24,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = SEED,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    tuple[str, ...],
]:
    """Build stratified train and validation DataLoaders.

    Args:
        source: Either ``"synthetic"`` or ``"real"``.
        data_path: Required when ``source == "real"``.
        n_per_class: Number of synthetic samples per class.
        batch_size: DataLoader batch size.
        val_split: Fraction of the dataset reserved for validation.
        num_workers: PyTorch DataLoader worker count.
        seed: RNG seed for the train/val split and worker initialisation.

    Returns:
        ``(train_loader, val_loader, class_names)``.

    Raises:
        ValueError: If ``source`` is unknown or ``data_path`` is missing for
            real-data runs.
    """
    train_tf = build_transforms(train=True)
    val_tf = build_transforms(train=False)

    if source == "synthetic":
        base: SyntheticToothDataset | RealToothDataset = SyntheticToothDataset(
            n_per_class=n_per_class, seed=seed, transform=None
        )
    elif source == "real":
        if data_path is None:
            raise ValueError('data_path is required when source="real"')
        base = RealToothDataset(root=data_path, transform=None)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    train_idx, val_idx = _stratified_indices(base.labels, val_split=val_split, seed=seed)
    if not train_idx or not val_idx:
        raise RuntimeError(
            f"Stratified split produced empty partition "
            f"(train={len(train_idx)}, val={len(val_idx)}). "
            "Increase n_per_class or lower val_split."
        )

    train_view = _SubsetWithTransform(base, train_idx, train_tf)
    val_view = _SubsetWithTransform(base, val_idx, val_tf)

    generator = torch.Generator().manual_seed(seed)
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_view,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val_view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
    )
    logger.info(
        "Built dataloaders: source=%s train=%d val=%d batch=%d",
        source, len(train_view), len(val_view), batch_size,
    )
    return train_loader, val_loader, CLASS_NAMES


__all__ = [
    "NUM_CLASSES",
    "RealToothDataset",
    "SyntheticToothDataset",
    "build_dataloaders",
]
