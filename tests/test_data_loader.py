"""Unit tests for :mod:`tooth_resorption.data.data_loader`."""

from __future__ import annotations

import pytest
import torch
from tooth_resorption.config import NUM_CLASSES
from tooth_resorption.data.data_loader import SyntheticToothDataset, build_dataloaders
from tooth_resorption.data.preprocessing import build_transforms


def test_synthetic_dataset_length_and_getitem() -> None:
    ds = SyntheticToothDataset(n_per_class=3, seed=0, transform=build_transforms(train=False))
    assert len(ds) == NUM_CLASSES * 3
    tensor, label = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert 0 <= label < NUM_CLASSES


def test_build_dataloaders_yields_correct_batch_shape(dataloaders) -> None:  # type: ignore[no-untyped-def]
    train_loader, val_loader, class_names = dataloaders
    assert len(class_names) == NUM_CLASSES

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.ndim == 4
    assert batch_x.shape[1:] == (3, 224, 224)
    assert batch_y.ndim == 1
    assert batch_y.shape[0] == batch_x.shape[0]
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype in (torch.int64, torch.int32, torch.int16, torch.long)

    val_x, _ = next(iter(val_loader))
    assert val_x.shape[1:] == (3, 224, 224)


def test_build_dataloaders_stratification() -> None:
    train_loader, val_loader, _ = build_dataloaders(
        source="synthetic", n_per_class=9, batch_size=4, val_split=0.33, seed=7
    )
    val_labels: list[int] = []
    for _, labels in val_loader:
        val_labels.extend(labels.tolist())
    assert set(val_labels) == set(
        range(NUM_CLASSES)
    ), f"Stratified split should keep at least one sample per class in val; got {set(val_labels)}"

    train_size = sum(x.size(0) for x, _ in train_loader)
    val_size = sum(x.size(0) for x, _ in val_loader)
    assert train_size + val_size == NUM_CLASSES * 9


def test_build_dataloaders_rejects_unknown_source() -> None:
    with pytest.raises(ValueError):
        build_dataloaders(source="not-a-source", n_per_class=3, val_split=0.2)  # type: ignore[arg-type]


def test_build_dataloaders_requires_data_path_for_real() -> None:
    with pytest.raises(ValueError):
        build_dataloaders(source="real", data_path=None)
