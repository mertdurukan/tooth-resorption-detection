"""Image preprocessing and augmentation pipelines.

Augmentations match those used in the original MSc experiments (random
horizontal flip, ±15° rotation, small affine translation, mild colour/contrast
jitter) and produce ImageNet-normalised 3-channel tensors of shape
``(3, IMAGE_SIZE, IMAGE_SIZE)``.

Why mild colour jitter on grayscale X-rays? The base images are converted to
RGB so we can re-use ImageNet pretraining. Aggressive colour jitter would
distort the radiological intensities; ``brightness=contrast=0.15`` is the
upper bound that keeps the X-ray plausibly radiological.
"""

from __future__ import annotations

from typing import Callable

from torchvision import transforms

from tooth_resorption.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

Transform = Callable[..., object]


def build_transforms(image_size: int = IMAGE_SIZE, train: bool = True) -> transforms.Compose:
    """Build a torchvision transform pipeline.

    Args:
        image_size: Square output side, in pixels.
        train: If ``True``, apply stochastic augmentations; otherwise only
            resize + ImageNet normalisation.

    Returns:
        A :class:`torchvision.transforms.Compose` mapping ``PIL.Image`` to
        a normalised float tensor.
    """
    ops: list[Transform] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
    return transforms.Compose(ops)


__all__ = ["build_transforms"]
