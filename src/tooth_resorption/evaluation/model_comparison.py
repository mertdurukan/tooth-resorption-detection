"""Cross-model evaluation utilities.

Unifies PyTorch (attention zoo), TensorFlow (legacy CNN) and YOLO model
evaluation behind a single :class:`UnifiedModelEvaluator` API so each
architecture is benchmarked under identical inputs and metrics.

Migrated and de-Turkified from the legacy ``evaluate_all_models.py`` script.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from tooth_resorption.config import (
    CLASS_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    RAW_DATA_DIR,
)
from tooth_resorption.data.data_loader import RealToothDataset
from tooth_resorption.data.preprocessing import build_transforms
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.models.architectures import create_model, get_model_info

logger = get_logger(__name__)

DEFAULT_DATA_PATH = RAW_DATA_DIR / "labelme_dataset"


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a metrics dict identical to the legacy comparison pipeline."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    precision_pc, recall_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )

    auc_macro = 0.0
    auc_per_class = [0.0, 0.0, 0.0]
    if y_pred_proba is not None:
        try:
            y_true_bin = label_binarize(y_true_arr, classes=list(range(NUM_CLASSES)))
            auc_macro = float(
                roc_auc_score(y_true_bin, y_pred_proba, average="macro", multi_class="ovr")
            )
            for i in range(NUM_CLASSES):
                if len(np.unique(y_true_bin[:, i])) > 1:
                    auc_per_class[i] = float(roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i]))
        except ValueError as exc:
            logger.debug("AUC could not be computed: %s", exc)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(NUM_CLASSES)))
    composite = (
        accuracy + float(precision_macro) + float(recall_macro) + float(f1_macro) + auc_macro
    ) / 5.0

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "auc_macro": auc_macro,
        "composite_score": float(composite),
        "precision_per_class": [float(x) for x in precision_pc],
        "recall_per_class": [float(x) for x in recall_pc],
        "f1_per_class": [float(x) for x in f1_pc],
        "auc_per_class": auc_per_class,
        "confusion_matrix": cm.tolist(),
    }


class UnifiedModelEvaluator:
    """Evaluate any combination of PyTorch / TensorFlow / YOLO classifiers."""

    def __init__(self, data_path: Path | str = DEFAULT_DATA_PATH) -> None:
        self.data_path = Path(data_path)
        self.class_names = list(CLASS_NAMES)
        self.results: list[dict[str, Any]] = []

        logger.info("Loading test dataset from %s", self.data_path)
        self.dataset = RealToothDataset(
            root=self.data_path, transform=build_transforms(train=False)
        )
        if len(self.dataset) == 0:
            raise ValueError(f"Dataset is empty: {self.data_path}")
        self.test_loader: DataLoader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        logger.info("Loaded %d samples", len(self.dataset))

    def evaluate_pytorch_model(
        self, model_path: Path | str, model_type: str
    ) -> dict[str, Any] | None:
        """Evaluate a single attention-zoo PyTorch checkpoint."""
        model_path = Path(model_path)
        logger.info("Evaluating PyTorch model %s from %s", model_type, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model = create_model(model_type, num_classes=NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            info = get_model_info(model)

            y_true: list[int] = []
            y_pred: list[int] = []
            y_pred_proba: list[np.ndarray] = []
            inference_times: list[float] = []

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(device)
                    t0 = time.time()
                    outputs = model(images)
                    inference_times.append((time.time() - t0) * 1000.0)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    y_true.extend(labels.tolist())
                    y_pred.extend(preds.cpu().tolist())
                    y_pred_proba.append(probs.cpu().numpy())

            proba_arr = np.concatenate(y_pred_proba, axis=0)
            metrics = compute_metrics(y_true, y_pred, proba_arr)
            metrics.update(
                {
                    "model_type": model_type,
                    "model_category": "Attention/Transformer",
                    "inference_time_ms": float(np.mean(inference_times)),
                    "total_params": int(info["total_params"]),
                    "model_size_mb": float(info["model_size_mb"]),
                }
            )
            return metrics
        except Exception as exc:
            logger.exception("Failed to evaluate %s: %s", model_type, exc)
            return None

    def evaluate_keras_cnn(
        self, model_path: Path | str = MODELS_DIR / "cnn_baseline.h5"
    ) -> dict[str, Any] | None:
        """Evaluate the legacy TensorFlow CNN baseline."""
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning("CNN model not found at %s", model_path)
            return None
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("TensorFlow not installed; skipping CNN baseline")
            return None

        logger.info("Evaluating CNN baseline from %s", model_path)
        try:
            model = tf.keras.models.load_model(model_path)

            images: list[np.ndarray] = []
            labels: list[int] = []
            for img, label in self.test_loader:
                img_np = img.squeeze().permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                images.append(img_np)
                labels.append(int(label.item()))

            images_arr = np.asarray(images)
            labels_arr = np.asarray(labels)

            inference_times: list[float] = []
            probs_list: list[np.ndarray] = []
            for img in images_arr:
                t0 = time.time()
                pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
                inference_times.append((time.time() - t0) * 1000.0)
                probs_list.append(pred[0])

            proba_arr = np.asarray(probs_list)
            preds = np.argmax(proba_arr, axis=1)
            metrics = compute_metrics(labels_arr, preds, proba_arr)
            metrics.update(
                {
                    "model_type": "cnn_baseline",
                    "model_category": "CNN",
                    "inference_time_ms": float(np.mean(inference_times)),
                    "total_params": int(model.count_params()),
                    "model_size_mb": float(model_path.stat().st_size / (1024 * 1024)),
                }
            )
            return metrics
        except Exception as exc:
            logger.exception("Failed to evaluate CNN baseline: %s", exc)
            return None

    def evaluate_yolo_model(self, weights: Path | str | None = None) -> dict[str, Any] | None:
        """Evaluate the YOLO detection model as a 3-class classifier."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed; skipping YOLO")
            return None

        weights_path: Path | None
        if weights is None:
            candidates = sorted(MODELS_DIR.glob("yolo*best*.pt"))
            weights_path = candidates[-1] if candidates else None
        else:
            weights_path = Path(weights)
        if weights_path is None or not weights_path.exists():
            logger.warning("YOLO weights not found")
            return None

        logger.info("Evaluating YOLO model from %s", weights_path)
        model = YOLO(str(weights_path))

        y_true: list[int] = []
        y_pred: list[int] = []
        y_pred_proba: list[np.ndarray] = []
        inference_times: list[float] = []

        for i in range(len(self.dataset)):
            img, label = self.dataset.samples[i]
            img_np = np.asarray(img)
            t0 = time.time()
            results = model.predict(img_np, verbose=False)
            inference_times.append((time.time() - t0) * 1000.0)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                max_idx = int(boxes.conf.argmax())
                pred_class = int(boxes.cls[max_idx])
                confidence = float(boxes.conf[max_idx])
                probs = np.zeros(NUM_CLASSES)
                probs[pred_class] = confidence
                remainder = (1.0 - confidence) / max(1, NUM_CLASSES - 1)
                probs[probs == 0] = remainder
            else:
                pred_class = 0
                probs = np.array([0.5, 0.25, 0.25])

            y_true.append(int(label))
            y_pred.append(pred_class)
            y_pred_proba.append(probs)

        metrics = compute_metrics(y_true, y_pred, np.asarray(y_pred_proba))
        metrics.update(
            {
                "model_type": "yolov11_detection",
                "model_category": "YOLO",
                "inference_time_ms": float(np.mean(inference_times)),
                "model_size_mb": float(weights_path.stat().st_size / (1024 * 1024)),
            }
        )
        return metrics

    def evaluate_all(self, models_dir: Path = MODELS_DIR) -> list[dict[str, Any]]:
        """Evaluate every PyTorch checkpoint plus the legacy CNN and YOLO models."""
        results: list[dict[str, Any]] = []

        cnn_result = self.evaluate_keras_cnn()
        if cnn_result is not None:
            results.append(cnn_result)

        yolo_result = self.evaluate_yolo_model()
        if yolo_result is not None:
            results.append(yolo_result)

        if models_dir.exists():
            for model_file in sorted(models_dir.glob("*_best.pth")):
                model_type = model_file.stem.replace("_best", "")
                metrics = self.evaluate_pytorch_model(model_file, model_type)
                if metrics is not None:
                    results.append(metrics)

        self.results = results
        return results

    def save_results(self, output_dir: Path | str = METRICS_DIR) -> pd.DataFrame:
        """Persist evaluation results as JSON + CSV and return the comparison table."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "evaluation_results.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Wrote %s", json_path)

        summary_rows = [
            {
                "Model": r["model_type"],
                "Category": r["model_category"],
                "Accuracy": r["accuracy"],
                "Precision": r["precision_macro"],
                "Recall": r["recall_macro"],
                "F1-Score": r["f1_macro"],
                "AUC": r["auc_macro"],
                "CompositeScore": r["composite_score"],
                "InferenceMs": r["inference_time_ms"],
                "SizeMb": r.get("model_size_mb", 0.0),
            }
            for r in self.results
        ]
        df = pd.DataFrame(summary_rows).sort_values("CompositeScore", ascending=False)
        csv_path = output_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)
        return df

    def per_class_report(self, model_type: str, output_dir: Path | str = METRICS_DIR) -> Path:
        """Write a sklearn classification report for ``model_type`` and return its path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        match = next((r for r in self.results if r["model_type"] == model_type), None)
        if match is None:
            raise ValueError(f"Model not in evaluator.results: {model_type!r}")
        y_true: list[int] = []
        y_pred: list[int] = []
        cm = np.asarray(match["confusion_matrix"])
        for true_idx, row in enumerate(cm):
            for pred_idx, count in enumerate(row):
                y_true.extend([true_idx] * int(count))
                y_pred.extend([pred_idx] * int(count))
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, digits=4, zero_division=0
        )
        out = output_dir / f"classification_report_{model_type}.txt"
        out.write_text(report, encoding="utf-8")
        return out


__all__ = [
    "DEFAULT_DATA_PATH",
    "UnifiedModelEvaluator",
    "compute_metrics",
]
