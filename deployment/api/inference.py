"""Inference helper for the deployed ViT-Base/16 tooth-resorption model.

Loads a TorchScript-exported model from ``deployment/torchscript/`` and
returns a JSON-serialisable prediction. For ONNX serving see
``docs/deployment.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ToothResorptionInference:
    """Production-style wrapper around a TorchScript ViT-Base/16 model."""

    DEFAULT_CLASS_NAMES: tuple[str, ...] = ("temasli", "bagimsiz", "rezorpsiyon")
    DEFAULT_IMAGE_SIZE: int = 224
    IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        class_names: tuple[str, ...] | None = None,
        image_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        size = image_size or self.DEFAULT_IMAGE_SIZE
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        """Read ``image_path`` and convert it into the network's input tensor."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Return a JSON-serialisable prediction for a single image."""
        tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())

        return {
            "predicted_class": self.class_names[predicted_class],
            "predicted_class_index": predicted_class,
            "confidence": confidence,
            "probabilities": {
                class_name: float(probabilities[0, i].item())
                for i, class_name in enumerate(self.class_names)
            },
            "model_name": "vit_base_16",
            "model_performance": {
                "accuracy": 0.8987,
                "macro_f1": 0.8981,
                "macro_auc_ovr": 0.9862,
            },
        }

    def predict_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        """Run :meth:`predict` over an iterable of image paths."""
        results: list[dict[str, Any]] = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result["image_path"] = str(image_path)
            results.append(result)
        return results


def _smoke_test() -> None:
    detector = ToothResorptionInference(
        model_path="deployment/torchscript/vit_base_16.pt",
        device="cpu",
    )
    rng = np.random.default_rng(0)
    dummy = Image.fromarray(rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8))
    tmp_path = Path("deployment/_smoke.jpg")
    dummy.save(tmp_path)
    try:
        print(json.dumps(detector.predict(tmp_path), indent=2))
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    _smoke_test()
