"""Wisdom tooth root-resorption detection pipeline.

Public re-exports keep import paths short for downstream users and notebooks::

    from tooth_resorption import build_model, build_dataloaders, TrainConfig
"""

from __future__ import annotations

from tooth_resorption.config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
    SEED,
    TrainConfig,
)
from tooth_resorption.data.data_loader import (
    RealToothDataset,
    SyntheticToothDataset,
    build_dataloaders,
)
from tooth_resorption.models.model import (
    AttentionViT,
    BaselineCNN,
    build_model,
    count_parameters,
)

__version__ = "0.2.0"

__all__ = [
    "CLASS_DESCRIPTIONS",
    "CLASS_NAMES",
    "IMAGE_SIZE",
    "NUM_CLASSES",
    "SEED",
    "AttentionViT",
    "BaselineCNN",
    "RealToothDataset",
    "SyntheticToothDataset",
    "TrainConfig",
    "__version__",
    "build_dataloaders",
    "build_model",
    "count_parameters",
]
