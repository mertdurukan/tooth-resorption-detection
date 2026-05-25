"""Procedural synthetic dental-X-ray generator.

The real clinical dataset cannot be redistributed, so the pipeline needs a
self-contained substitute that exercises every code path (data loading,
augmentation, training, evaluation). This module produces grayscale
``IMAGE_SIZE x IMAGE_SIZE`` images that vaguely resemble panoramic dental
X-rays plus per-image class labels.

The images are NOT medically meaningful; they exist purely so the rest of
the pipeline can run end-to-end without leaking patient data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tooth_resorption.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, SEED
from tooth_resorption.logging_utils import get_logger

logger = get_logger(__name__)


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _base_xray(rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    """Build a grayscale background resembling a soft-tissue X-ray field."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    radial = 1 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size * 0.8)
    radial = np.clip(radial, 0.1, 1.0)
    noise = rng.normal(0.0, 0.08, (size, size))
    field = 0.45 * radial + 0.15 + noise
    field = np.clip(field, 0.0, 1.0)
    out: np.ndarray = (field * 255).astype(np.uint8)
    return out


def _draw_teeth(img: Image.Image, rng: np.random.Generator, n_teeth: int = 6) -> None:
    """Sprinkle bright ellipses to mimic teeth in the lower half of the frame."""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for _ in range(n_teeth):
        w = int(rng.integers(18, 32))
        h = int(rng.integers(28, 48))
        x = int(rng.integers(8, width - w - 8))
        y = int(rng.integers(height // 2 - 10, height - h - 10))
        brightness = int(rng.integers(190, 245))
        draw.ellipse((x, y, x + w, y + h), fill=brightness)


def _apply_class_signature(img: Image.Image, label: int, rng: np.random.Generator) -> None:
    """Inject a class-conditional pattern so a model can actually learn.

    - ``temasli`` (0): two ellipses placed in contact (touching).
    - ``bagimsiz`` (1): two ellipses placed clearly apart.
    - ``rezorpsiyon`` (2): an ellipse with a dark erosion blob along its root.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    cy = height - 60
    if label == 0:
        cx = width // 2 - 20
        draw.ellipse((cx, cy, cx + 28, cy + 40), fill=230)
        draw.ellipse((cx + 26, cy, cx + 54, cy + 40), fill=230)
    elif label == 1:
        cx = width // 2 - 60
        draw.ellipse((cx, cy, cx + 28, cy + 40), fill=230)
        draw.ellipse((cx + 70, cy, cx + 98, cy + 40), fill=230)
    else:
        cx = width // 2 - 14
        draw.ellipse((cx, cy, cx + 28, cy + 44), fill=230)
        ex = cx + int(rng.integers(2, 12))
        ey = cy + 28
        draw.ellipse((ex, ey, ex + 14, ey + 14), fill=40)


def generate(
    n_per_class: int = 8,
    seed: int = SEED,
    image_size: int = IMAGE_SIZE,
) -> tuple[list[Image.Image], list[int]]:
    """Generate ``n_per_class`` synthetic samples for every class.

    Args:
        n_per_class: Number of synthetic samples per class.
        seed: RNG seed; the generator is fully deterministic given this value.
        image_size: Square output side, in pixels.

    Returns:
        A tuple ``(images, labels)`` where ``images`` are PIL RGB images of
        size ``image_size x image_size`` and ``labels`` are integers in
        ``[0, NUM_CLASSES)``.
    """
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be >= 1, got {n_per_class}")
    rng = _seeded_rng(seed)
    images: list[Image.Image] = []
    labels: list[int] = []
    for label in range(NUM_CLASSES):
        for _ in range(n_per_class):
            gray = _base_xray(rng, image_size)
            pil = Image.fromarray(gray, mode="L").convert("RGB")
            _draw_teeth(pil, rng, n_teeth=int(rng.integers(4, 8)))
            _apply_class_signature(pil, label, rng)
            pil = pil.filter(ImageFilter.GaussianBlur(radius=0.6))
            images.append(pil)
            labels.append(label)

    paired = list(zip(images, labels, strict=True))
    random.Random(seed).shuffle(paired)
    shuffled_images, shuffled_labels = zip(*paired, strict=True)
    return list(shuffled_images), list(shuffled_labels)


def save_synthetic_dataset(
    out_dir: Path | str,
    n_per_class: int = 8,
    seed: int = SEED,
) -> Path:
    """Persist a synthetic dataset to disk in a class-folder layout.

    Args:
        out_dir: Destination directory. Created if missing.
        n_per_class: Number of synthetic samples per class.
        seed: RNG seed.

    Returns:
        Path to the written ``manifest.json``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images, labels = generate(n_per_class=n_per_class, seed=seed)
    manifest: list[dict[str, object]] = []
    for idx, (img, label) in enumerate(zip(images, labels, strict=True)):
        class_name = CLASS_NAMES[label]
        class_dir = out_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        img_path = class_dir / f"{idx:04d}.png"
        img.save(img_path)
        manifest.append(
            {"path": str(img_path.relative_to(out_path)), "label": int(label)}
        )
    manifest_path = out_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"class_names": list(CLASS_NAMES), "samples": manifest}, f, indent=2)
    logger.info("Saved synthetic manifest at %s (%d samples)", manifest_path, len(manifest))
    return manifest_path


def _cli() -> None:
    path = save_synthetic_dataset("data/synthetic", n_per_class=8)
    logger.info("Saved synthetic manifest at: %s", path)


if __name__ == "__main__":
    _cli()


__all__ = ["generate", "save_synthetic_dataset"]
