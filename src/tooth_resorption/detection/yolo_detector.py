"""YOLOv11 object-detection wrapper for the tooth-resorption pipeline.

Builds a YOLO-format dataset from LabelMe JSON annotations, trains the
``yolo11n`` (nano) backbone, and runs detection-style evaluation. ASCII-only
filenames are generated for every image so the YOLO toolchain does not trip
on the original Turkish patient names.

``ultralytics`` is imported lazily so the module remains importable without
the YOLO dependency installed.
"""

from __future__ import annotations

import base64
import json
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from tooth_resorption.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
)
from tooth_resorption.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_PATH = RAW_DATA_DIR / "labelme_dataset"
DEFAULT_DATASET_DIR = PROCESSED_DATA_DIR / "yolo_dataset"


def _ascii_slug(name: str) -> str:
    """Strip diacritics and non-ASCII characters from ``name`` for YOLO compatibility."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return folded.replace(" ", "_") or "image"


class YoloToothDetector:
    """End-to-end YOLOv11 pipeline for tooth-resorption detection."""

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        dataset_dir: Path | str = DEFAULT_DATASET_DIR,
        image_size: int = 640,
    ) -> None:
        self.data_path = Path(data_path)
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.class_names = list(CLASS_NAMES)
        self.model: Any = None

    def prepare_dataset(self, val_split: float = 0.2) -> int:
        """Convert LabelMe JSONs to YOLO-format dataset.

        Returns:
            Total number of images that were exported.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path missing: {self.data_path}")

        for split in ("train", "val"):
            for sub in ("images", "labels"):
                (self.dataset_dir / split / sub).mkdir(parents=True, exist_ok=True)

        images_info: list[dict[str, Any]] = []
        for file in sorted(self.data_path.iterdir()):
            if file.suffix.lower() != ".json":
                continue
            try:
                with file.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                image_data = payload.get("imageData")
                if not image_data:
                    continue
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                img_width, img_height = image.size

                image_name = _ascii_slug(file.stem) + ".jpg"

                annotations: list[str] = []
                shapes = payload.get("shapes") or []
                for shape in shapes:
                    if shape.get("shape_type") != "rectangle":
                        continue
                    label = shape.get("label", "")
                    low = label.lower()
                    class_id = 0
                    if "temas" in low:
                        class_id = 0
                    elif "ba" in low and ("ms" in low or "gim" in low):
                        class_id = 1
                    elif "rezo" in low or "resor" in low:
                        class_id = 2

                    (x1, y1), (x2, y2) = shape["points"]
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = abs(x2 - x1) / img_width
                    height = abs(y2 - y1) / img_height
                    annotations.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    )

                images_info.append(
                    {"image_name": image_name, "image": image, "annotations": annotations}
                )
            except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", file.name, exc)
                continue

        rng = np.random.default_rng(SEED)
        indices = rng.permutation(len(images_info))
        val_size = max(1, int(len(images_info) * val_split))
        val_indices = {int(i) for i in indices[:val_size]}

        for idx, info in enumerate(images_info):
            split = "val" if idx in val_indices else "train"
            img_path = self.dataset_dir / split / "images" / info["image_name"]
            info["image"].save(img_path)
            label_path = (
                self.dataset_dir / split / "labels" / info["image_name"].replace(".jpg", ".txt")
            )
            label_path.write_text("\n".join(info["annotations"]) + "\n", encoding="utf-8")

        dataset_yaml = {
            "path": str(self.dataset_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": NUM_CLASSES,
            "names": self.class_names,
        }
        (self.dataset_dir / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml), encoding="utf-8"
        )
        logger.info("Exported %d images to %s", len(images_info), self.dataset_dir)
        return len(images_info)

    def train(
        self,
        epochs: int = 50,
        batch: int = 4,
        project: Path | str = "runs",
        name: str = "tooth_resorption",
        weights: str = "yolo11n.pt",
    ) -> Any:
        """Train ``yolo11n`` on the prepared dataset."""
        from ultralytics import YOLO

        self.model = YOLO(weights)
        results = self.model.train(
            data=str(self.dataset_dir / "dataset.yaml"),
            epochs=epochs,
            imgsz=self.image_size,
            batch=batch,
            patience=10,
            save=True,
            plots=True,
            verbose=True,
            project=str(project),
            name=name,
        )
        logger.info("YOLO training finished")
        return results

    def evaluate(self) -> Any:
        """Run YOLO validation on the held-out split."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded")
        results = self.model.val(
            data=str(self.dataset_dir / "dataset.yaml"),
            save_json=True,
            plots=True,
        )
        logger.info(
            "YOLO eval | mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f",
            results.box.map50,
            results.box.map,
            results.box.mp,
            results.box.mr,
        )
        return results

    def predict(self, image_path: Path | str) -> list[dict[str, Any]]:
        """Run inference on a single image; returns the per-detection list."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded")
        results = self.model.predict(str(image_path), verbose=False)
        detections: list[dict[str, Any]] = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id],
                        "confidence": float(box.conf[0]),
                        "xyxy": [float(x) for x in box.xyxy[0].tolist()],
                    }
                )
        return detections


__all__ = ["DEFAULT_DATASET_DIR", "DEFAULT_DATA_PATH", "YoloToothDetector"]
