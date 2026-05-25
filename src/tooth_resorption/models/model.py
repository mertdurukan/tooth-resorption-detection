"""Model definitions.

Ships two architectures — the head-to-head comparison reported in the README:

* :class:`BaselineCNN`: a compact ~650K-parameter convolutional classifier
  matching the original baseline used at the start of the MSc work.
* :class:`AttentionViT`: a Vision Transformer (``vit_base_patch16_224`` by
  default) loaded via ``timm``. This is the "improved" model whose results
  are reported in the README.

The full attention/transformer zoo from the original MSc work (Swin-Tiny /
Small / Base, ResNet50-CBAM / SE, EfficientNet-Attention, DenseNet-Attention,
and the ViT/Swin + attention variants) lives in
:mod:`tooth_resorption.models.architectures`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from tooth_resorption.config import NUM_CLASSES, ModelName


class BaselineCNN(nn.Module):
    """Small convolutional baseline classifier.

    Architecture mirrors the original 2D CNN used at the start of the MSc
    work: four conv-pool blocks followed by a dense head. Roughly 650K
    parameters; trains in seconds on CPU.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.features(x)
        logits: torch.Tensor = self.classifier(features)
        return logits


class AttentionViT(nn.Module):
    """Vision Transformer classifier.

    Wraps a ``timm`` ViT backbone (default ``vit_base_patch16_224``) and
    replaces its head with a ``num_classes`` linear projection. ``timm`` is
    imported lazily so the rest of the package remains importable without it.

    Args:
        model_name: A ``timm`` model identifier.
        num_classes: Number of output logits.
        pretrained: Whether to load ImageNet-21k weights (requires network).
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.backbone(x)
        return logits

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Toggle gradient computation on every backbone parameter except the head.

        Used for the two-stage fine-tuning strategy described in the README:
        freeze for the first ~10 epochs (head-only warmup), then unfreeze the
        whole stack for the remaining epochs.
        """
        for name, param in self.backbone.named_parameters():
            if "head" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = not freeze


_MODEL_REGISTRY: dict[ModelName, Callable[[bool], nn.Module]] = {
    "baseline_cnn": lambda _pretrained: BaselineCNN(),
    "vit_tiny": lambda pretrained: AttentionViT("vit_tiny_patch16_224", pretrained=pretrained),
    "vit_base": lambda pretrained: AttentionViT("vit_base_patch16_224", pretrained=pretrained),
}


def build_model(name: ModelName, pretrained: bool = False) -> nn.Module:
    """Instantiate one of the shipped architectures.

    Args:
        name: One of ``"baseline_cnn"``, ``"vit_tiny"``, ``"vit_base"``.
        pretrained: Whether to load ImageNet weights (ViT-only; silently
            ignored for the baseline CNN).

    Raises:
        ValueError: If ``name`` is not a registered model.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Choose from {sorted(_MODEL_REGISTRY)}."
        )
    return _MODEL_REGISTRY[name](pretrained)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


__all__ = ["AttentionViT", "BaselineCNN", "build_model", "count_parameters"]
