"""Export the best attention-zoo checkpoint to ONNX + TorchScript.

Designed for downstream deployment (Flask API, mobile, ONNX Runtime).
Writes artefacts under ``deployment/{onnx,torchscript,quantized}/`` along
with ``deployment/model_metadata.json``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from tooth_resorption.config import (
    CLASS_NAMES,
    DEPLOYMENT_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    METRICS_DIR,
    NUM_CLASSES,
)
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.models.architectures import create_model, get_model_info

logger = get_logger(__name__)


def _load_pytorch_checkpoint(model_type: str, checkpoint_path: Path) -> torch.nn.Module:
    model = create_model(model_type, num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=14,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    logger.info("Exported ONNX -> %s", out_path)
    return out_path


def export_torchscript(model: torch.nn.Module, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(out_path))
    logger.info("Exported TorchScript -> %s", out_path)
    return out_path


def export_quantized(model: torch.nn.Module, out_path: Path) -> Path:
    """Dynamic int8 quantization of ``nn.Linear`` layers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), out_path)
    logger.info("Exported quantized state dict -> %s", out_path)
    return out_path


def write_metadata(
    model_type: str,
    checkpoint_path: Path,
    metrics_payload: dict[str, Any] | None,
    out_path: Path,
) -> Path:
    info: dict[str, Any] = {
        "model_info": {
            "name": model_type,
            "version": "1.0.0",
            "export_date": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": str(checkpoint_path),
        },
        "classes": list(CLASS_NAMES),
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "preprocessing": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        },
    }
    if metrics_payload is not None:
        info["performance"] = metrics_payload
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info("Wrote metadata -> %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the best PyTorch model for deployment.")
    parser.add_argument("--model-type", required=True, help="Architecture key (e.g. vit_base_16).")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the *_best.pth checkpoint."
    )
    parser.add_argument(
        "--metrics-json", type=Path, default=METRICS_DIR / "evaluation_results.json"
    )
    parser.add_argument("--deployment-dir", type=Path, default=DEPLOYMENT_DIR)
    parser.add_argument("--skip-quantization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = _load_pytorch_checkpoint(args.model_type, args.checkpoint)
    info = get_model_info(model)
    logger.info("Loaded %s | %s", args.model_type, info)

    export_onnx(model, args.deployment_dir / "onnx" / f"{args.model_type}.onnx")
    export_torchscript(model, args.deployment_dir / "torchscript" / f"{args.model_type}.pt")
    if not args.skip_quantization:
        export_quantized(model, args.deployment_dir / "quantized" / f"{args.model_type}_int8.pt")

    metrics_payload: dict[str, Any] | None = None
    if args.metrics_json.exists():
        try:
            with args.metrics_json.open("r", encoding="utf-8") as f:
                all_metrics = json.load(f)
            match = next((r for r in all_metrics if r["model_type"] == args.model_type), None)
            if match is not None:
                metrics_payload = match
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not parse metrics file: %s", exc)

    write_metadata(
        args.model_type,
        args.checkpoint,
        metrics_payload,
        args.deployment_dir / "model_metadata.json",
    )


if __name__ == "__main__":
    main()
