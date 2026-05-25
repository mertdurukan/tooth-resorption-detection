"""Input-gradient saliency maps for model interpretability.

Given any classifier from :mod:`tooth_resorption.models.architectures`, this
module computes per-pixel sensitivity for a target class and renders a heat
map overlay on top of the original radiograph.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tooth_resorption.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def load_sample_image(json_path: Path | str) -> tuple[Image.Image, str | None]:
    """Return ``(image, raw_label)`` from a LabelMe JSON file."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_data = data["imageData"]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    label: str | None = None
    for shape in data.get("shapes", []):
        label = shape.get("label")
        break
    return image, label


def compute_saliency_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, int]:
    """Return a 2D saliency map and the target class used to compute it.

    Uses the simple "input gradient" method (|d output[target] / d input|),
    taking the channel-wise maximum and rescaling to ``[0, 1]``.
    """
    model.eval()
    image_tensor = image_tensor.clone().requires_grad_(True)
    output = model(image_tensor)
    if target_class is None:
        target_class = int(output.argmax(dim=1).item())
    model.zero_grad()
    output[0, target_class].backward()
    saliency = image_tensor.grad.data.abs()
    saliency = saliency.squeeze().max(dim=0)[0]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.cpu().numpy(), target_class


def create_overlay(
    original_image: Image.Image,
    saliency_map: np.ndarray,
    alpha: float = 0.5,
    size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend the original image with a jet-colormap heatmap.

    Returns:
        ``(overlay, heatmap)`` — both ``np.uint8`` RGB arrays of shape ``(H, W, 3)``.
    """
    heatmap = cm.jet(saliency_map)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    original_np = np.array(original_image.resize((size, size)))
    overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)
    return overlay, heatmap


def default_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Inference-time transform matching the rest of the pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


__all__ = [
    "compute_saliency_map",
    "create_overlay",
    "default_transform",
    "load_sample_image",
]
