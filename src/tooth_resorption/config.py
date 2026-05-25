"""Central configuration for the tooth-resorption-detection pipeline.

All hyper-parameters, paths and class definitions live here so that the rest
of the codebase stays free of magic numbers and can be re-used as a library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42

# --------------------------------------------------------------------------- #
# Filesystem layout                                                           #
# Project root is two levels up from this file:                               #
#   src/tooth_resorption/config.py -> tooth-resorption-detection/             #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
SYNTHETIC_DIR: Final[Path] = DATA_DIR / "synthetic"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = RESULTS_DIR / "figures"
PLOTS_DIR: Final[Path] = RESULTS_DIR / "plots"
REPORTS_DIR: Final[Path] = RESULTS_DIR / "reports"
METRICS_DIR: Final[Path] = RESULTS_DIR / "metrics"
DEPLOYMENT_DIR: Final[Path] = PROJECT_ROOT / "deployment"

# --------------------------------------------------------------------------- #
# Task definition                                                             #
# --------------------------------------------------------------------------- #
CLASS_NAMES: Final[tuple[str, str, str]] = ("temasli", "bagimsiz", "rezorpsiyon")
CLASS_DESCRIPTIONS: Final[tuple[str, str, str]] = (
    "in contact with adjacent tooth",
    "independent of adjacent tooth",
    "external root resorption present",
)
NUM_CLASSES: Final[int] = len(CLASS_NAMES)

# --------------------------------------------------------------------------- #
# Image normalisation                                                         #
# --------------------------------------------------------------------------- #
IMAGE_SIZE: Final[int] = 224
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

ModelName = Literal["baseline_cnn", "vit_tiny", "vit_base"]
DataSource = Literal["synthetic", "real"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Immutable training hyper-parameters.

    Defaults are tuned for a CPU smoke run on synthetic data (~60s end-to-end).
    For real-data training, override ``model_name="vit_base"`` with
    ``pretrained=True`` and bump ``epochs`` to ~50.

    Attributes:
        model_name: Which architecture to instantiate. One of
            ``"baseline_cnn"``, ``"vit_tiny"``, ``"vit_base"``.
        pretrained: Use ImageNet weights for ``vit_*`` models.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate for AdamW.
        weight_decay: AdamW weight decay.
        num_workers: PyTorch DataLoader worker count.
        val_split: Fraction of the dataset held out for validation.
        n_per_class_synthetic: Synthetic samples per class.
        augment: Toggle training-time augmentations.
        amp: Use ``torch.cuda.amp`` mixed-precision training.
        grad_clip: Gradient L2-norm clip value; ``None`` disables clipping.
        early_stopping_patience: Stop after N epochs without val-F1 improvement.
        deterministic: Force PyTorch into deterministic mode (slower).
    """

    model_name: ModelName = "baseline_cnn"
    pretrained: bool = False
    epochs: int = 2
    batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    val_split: float = 0.2
    n_per_class_synthetic: int = 24
    augment: bool = True
    amp: bool = False
    grad_clip: float | None = 1.0
    early_stopping_patience: int = 0
    deterministic: bool = False


__all__ = [
    "CLASS_DESCRIPTIONS",
    "CLASS_NAMES",
    "DATA_DIR",
    "DEPLOYMENT_DIR",
    "FIGURES_DIR",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_SIZE",
    "METRICS_DIR",
    "MODELS_DIR",
    "NUM_CLASSES",
    "PLOTS_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "SEED",
    "SYNTHETIC_DIR",
    "DataSource",
    "ModelName",
    "TrainConfig",
]
