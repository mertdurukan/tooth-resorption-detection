"""Generate a saliency map for a single LabelMe sample.

Example::

    python scripts/generate_saliency_map.py \
        --model-type swin_small_mha_head \
        --checkpoint models/swin_small_mha_head_best.pth \
        --json data/raw/labelme_dataset/some_patient.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from tooth_resorption.config import CLASS_NAMES, FIGURES_DIR, IMAGE_SIZE, NUM_CLASSES
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.models.architectures import create_model
from tooth_resorption.visualization.saliency import (
    compute_saliency_map,
    create_overlay,
    default_transform,
    load_sample_image,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an input-gradient saliency map.")
    parser.add_argument("--model-type", default="swin_small_mha_head")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Trained PyTorch checkpoint (.pth)."
    )
    parser.add_argument(
        "--json", type=Path, required=True, help="LabelMe JSON file with the input image."
    )
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "saliency_map_analysis.png")
    parser.add_argument("--overlay-out", type=Path, default=FIGURES_DIR / "saliency_overlay.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = create_model(args.model_type, num_classes=NUM_CLASSES, pretrained=False)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint %s", args.checkpoint)
    else:
        logger.warning("Checkpoint missing; using random weights (demo only).")
    model = model.to(device).eval()

    original_image, raw_label = load_sample_image(args.json)
    logger.info("Loaded sample %s | label=%s", args.json, raw_label)

    image_tensor = default_transform()(original_image).unsqueeze(0).to(device)
    saliency_map, predicted_class = compute_saliency_map(model, image_tensor)
    overlay, _ = create_overlay(original_image, saliency_map, size=IMAGE_SIZE)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[0].set_title("Input image", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(saliency_map, cmap="gray")
    axes[1].set_title("Saliency (gradient magnitude)", fontsize=12)
    axes[1].axis("off")
    axes[2].imshow(saliency_map, cmap="jet")
    axes[2].set_title("Heatmap (hot = important)", fontsize=12)
    axes[2].axis("off")
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay (pred = {CLASS_NAMES[predicted_class]})", fontsize=12)
    axes[3].axis("off")

    plt.suptitle(f"Saliency map analysis — {args.model_type}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", args.out)

    args.overlay_out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.overlay_out)
    logger.info("Wrote %s", args.overlay_out)


if __name__ == "__main__":
    main()
