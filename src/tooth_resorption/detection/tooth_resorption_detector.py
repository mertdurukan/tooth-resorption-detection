"""Legacy TensorFlow / Keras CNN classifier for tooth-resorption detection.

This is the original baseline used at the start of the MSc work and is kept
for reproducibility and side-by-side comparison with the modern PyTorch ViT
zoo. New experiments should prefer
:mod:`tooth_resorption.models.architectures`.

TensorFlow is imported lazily so the rest of the package stays importable
on systems that only have PyTorch installed.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from tooth_resorption.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    METRICS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    RAW_DATA_DIR,
    SEED,
)
from tooth_resorption.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_PATH = RAW_DATA_DIR / "labelme_dataset"


@dataclass
class CnnTrainConfig:
    """One point in the CNN grid search."""

    lr: float
    batch_size: int
    optimizer: str
    epochs: int


@dataclass
class CnnTrainResult:
    """Per-config result of the CNN grid search."""

    config: CnnTrainConfig
    val_acc: float
    val_loss: float
    training_time_seconds: float
    model_path: str
    history: dict[str, list[float]] = field(default_factory=dict)


class ToothResorptionDetector:
    """Keras CNN baseline classifier for tooth resorption.

    Wraps the original notebook pipeline behind a class so it can be reused
    from scripts. All Turkish identifiers and ad-hoc print statements have
    been replaced with English equivalents and a structured logger.
    """

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_size = (image_size, image_size)
        self.model: Any = None
        self.class_names = list(CLASS_NAMES)
        self.images: np.ndarray = np.empty((0,))
        self.labels: np.ndarray = np.empty((0,), dtype=np.int64)
        self.patient_info: list[dict[str, Any]] = []

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load every LabelMe JSON under ``self.data_path``.

        Returns:
            ``(images, labels)`` as NumPy arrays. ``images`` is normalised
            to ``[0, 1]`` and shaped ``(N, H, W, 3)``.
        """
        logger.info("Loading data from %s", self.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        images: list[np.ndarray] = []
        labels: list[int] = []
        patient_info: list[dict[str, Any]] = []

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
                image = Image.open(BytesIO(image_bytes)).convert("RGB").resize(self.image_size)
                image_array = np.asarray(image, dtype=np.float32) / 255.0

                label = 0
                shape_label = ""
                shapes = payload.get("shapes") or []
                if shapes:
                    shape_label = shapes[0].get("label", "")
                    low = shape_label.lower()
                    if "temas" in low:
                        label = 0
                    elif "ba" in low and ("ms" in low or "gim" in low):
                        label = 1
                    elif "rezo" in low or "resor" in low:
                        label = 2

                images.append(image_array)
                labels.append(label)
                patient_info.append({"file": file.name, "label": shape_label})
            except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", file.name, exc)
                continue

        self.images = np.asarray(images, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.patient_info = patient_info
        logger.info("Loaded %d images", len(images))
        return self.images, self.labels

    def build_model(self) -> Any:
        """Build the original CNN architecture and return the Keras model."""
        from tensorflow.keras import Input, Sequential, layers

        model = Sequential(
            [
                Input(shape=(*self.image_size, 3)),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.MaxPooling2D(2, 2),
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
        self.model = model
        logger.info("Built Keras CNN with %d parameters", model.count_params())
        return model

    def _train_with_config(
        self,
        config: CnnTrainConfig,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        save_dir: Path,
    ) -> CnnTrainResult:
        from tensorflow.keras import optimizers
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

        model = self.build_model()

        if config.optimizer == "Adam":
            opt = optimizers.Adam(learning_rate=config.lr)
        else:
            opt = optimizers.SGD(learning_rate=config.lr, momentum=0.9)

        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"cnn_temp_{config.lr}_{config.batch_size}.h5"

        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, verbose=0),
        ]

        start_time = time.time()
        history = model.fit(
            x_train,
            y_train,
            epochs=config.epochs,
            validation_data=(x_val, y_val),
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        training_time = time.time() - start_time

        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        self.model = model
        return CnnTrainResult(
            config=config,
            val_acc=float(val_acc),
            val_loss=float(val_loss),
            training_time_seconds=float(training_time),
            model_path=str(model_path),
            history={k: list(map(float, v)) for k, v in history.history.items()},
        )

    def grid_search_train(
        self,
        epochs: int = 50,
        save_dir: Path = MODELS_DIR,
        logs_dir: Path | None = None,
    ) -> CnnTrainResult:
        """Run the original 3-config grid search and return the best result."""
        if self.images.size == 0:
            self.load_data()
        if self.images.size == 0:
            raise RuntimeError("No data loaded; cannot train")

        x_train, x_val, y_train, y_val = train_test_split(
            self.images,
            self.labels,
            test_size=0.2,
            stratify=self.labels,
            random_state=SEED,
        )
        logger.info("Train=%d Val=%d", len(x_train), len(x_val))

        configs = [
            CnnTrainConfig(lr=0.001, batch_size=8, optimizer="Adam", epochs=epochs),
            CnnTrainConfig(lr=0.0005, batch_size=4, optimizer="Adam", epochs=epochs),
            CnnTrainConfig(lr=0.001, batch_size=8, optimizer="SGD", epochs=epochs),
        ]

        all_results: list[CnnTrainResult] = []
        best_result: CnnTrainResult | None = None
        for i, config in enumerate(configs, 1):
            logger.info("[%d/%d] %s", i, len(configs), config)
            result = self._train_with_config(config, x_train, y_train, x_val, y_val, save_dir)
            all_results.append(result)
            if best_result is None or result.val_acc > best_result.val_acc:
                best_result = result

        assert best_result is not None
        logger.info("Best config: %s (val_acc=%.4f)", best_result.config, best_result.val_acc)

        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "cnn_baseline.h5"
        self.model.save(final_path)
        logger.info("Saved best model -> %s", final_path)

        logs_dir = logs_dir or (METRICS_DIR / "training_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "model_type": "cnn_baseline",
            "best_config": best_result.config.__dict__,
            "best_val_acc": best_result.val_acc,
            "best_val_loss": best_result.val_loss,
            "training_time_seconds": best_result.training_time_seconds,
            "all_configs": [
                {
                    "config": r.config.__dict__,
                    "val_acc": r.val_acc,
                    "val_loss": r.val_loss,
                }
                for r in all_results
            ],
            "timestamp": datetime.now().isoformat(),
        }
        with (logs_dir / "cnn_training_results.json").open("w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)
        logger.info("Wrote training log -> %s", logs_dir / "cnn_training_results.json")
        return best_result

    def evaluate_model(self, test_split: float = 0.2) -> dict[str, Any]:
        """Evaluate the trained model on a stratified split."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        _x_train, x_test, _y_train, y_test = train_test_split(
            self.images,
            self.labels,
            test_size=test_split,
            stratify=self.labels,
            random_state=SEED,
        )
        y_pred_proba = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        report = classification_report(
            y_test, y_pred, target_names=self.class_names, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)

        per_class_auc: dict[str, float | None] = {}
        for i, class_name in enumerate(self.class_names):
            y_test_binary = (y_test == i).astype(int)
            y_pred_binary = y_pred_proba[:, i]
            if len(np.unique(y_test_binary)) > 1:
                per_class_auc[class_name] = float(roc_auc_score(y_test_binary, y_pred_binary))
            else:
                per_class_auc[class_name] = None

        accuracy = float(np.mean(y_pred == y_test))
        return {
            "accuracy": accuracy,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_proba": y_pred_proba.tolist(),
            "confusion_matrix": cm.tolist(),
            "per_class_auc": per_class_auc,
            "classification_report": report,
        }

    def predict_sample(self, image_path: Path | str) -> dict[str, Any]:
        """Run inference on a single image file."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        batch = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(batch)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return {
            "class": self.class_names[predicted_class],
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: float(prob) for i, prob in enumerate(prediction[0])
            },
        }


__all__ = ["CnnTrainConfig", "CnnTrainResult", "ToothResorptionDetector"]
