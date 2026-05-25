"""Grid-search training driver for the attention / transformer model zoo.

Migrated from the legacy ``train_attention_transformers.py`` script and
re-organised to fit the :mod:`tooth_resorption` package. Trains every model
in :func:`tooth_resorption.models.architectures.create_model` end-to-end
with a small per-model hyper-parameter sweep, two-stage fine-tuning (10-epoch
head-only warm-up, then unfreeze), early stopping and per-model JSON logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, random_split

from tooth_resorption.config import (
    CLASS_NAMES,
    METRICS_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    RAW_DATA_DIR,
    SEED,
)
from tooth_resorption.data.data_loader import RealToothDataset, _label_from_string
from tooth_resorption.data.preprocessing import build_transforms
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.models.architectures import create_model, get_model_info

logger = get_logger(__name__)

DEFAULT_DATA_PATH = RAW_DATA_DIR / "labelme_dataset"


@dataclass(frozen=True, slots=True)
class HyperParams:
    """Per-run hyper-parameters for the attention zoo grid search."""

    lr: float
    batch_size: int
    weight_decay: float
    optimizer: str
    epochs: int = 50


def _calculate_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_pred_proba: np.ndarray | None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Compute accuracy, macro precision/recall/F1, macro OVR AUC and a composite."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )

    auc = 0.0
    if y_pred_proba is not None:
        try:
            y_true_bin = label_binarize(y_true_arr, classes=list(range(num_classes)))
            auc = float(roc_auc_score(y_true_bin, y_pred_proba, average="macro", multi_class="ovr"))
        except ValueError:
            auc = 0.0

    composite = (accuracy + float(precision) + float(recall) + float(f1) + auc) / 5.0
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "composite_score": composite,
    }


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Single training pass returning loss + classification metrics."""
    model.train()
    running_loss = 0.0
    preds: list[int] = []
    labels_acc: list[int] = []
    probs: list[np.ndarray] = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
        labels_acc.extend(labels.cpu().tolist())

    metrics = _calculate_metrics(labels_acc, preds, np.concatenate(probs) if probs else None)
    metrics["loss"] = running_loss / max(1, len(train_loader))
    return metrics


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Single evaluation pass returning loss + classification metrics."""
    model.eval()
    running_loss = 0.0
    preds: list[int] = []
    labels_acc: list[int] = []
    probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())

            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            preds.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            labels_acc.extend(labels.cpu().tolist())

    metrics = _calculate_metrics(labels_acc, preds, np.concatenate(probs) if probs else None)
    metrics["loss"] = running_loss / max(1, len(val_loader))
    return metrics


def train_model_with_config(
    model_type: str,
    config: HyperParams,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    *,
    unfreeze_epoch: int = 10,
    max_patience: int = 15,
) -> dict[str, Any]:
    """Train a single model with the given hyper-parameters.

    Implements the two-stage fine-tuning recipe and saves the best checkpoint
    (composite score) to ``save_dir / {model_type}_best.pth``.
    """
    logger.info(
        "Training %s | lr=%g bs=%d wd=%g opt=%s",
        model_type,
        config.lr,
        config.batch_size,
        config.weight_decay,
        config.optimizer,
    )

    model = create_model(model_type, num_classes=NUM_CLASSES, pretrained=True).to(device)
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone(True)

    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        k: []
        for k in (
            "train_loss",
            "train_acc",
            "train_f1",
            "train_composite",
            "val_loss",
            "val_acc",
            "val_f1",
            "val_composite",
        )
    }

    best_composite = 0.0
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"{model_type}_best.pth"
    patience_counter = 0

    start_time = time.time()
    for epoch in range(config.epochs):
        if epoch == unfreeze_epoch and hasattr(model, "freeze_backbone"):
            logger.info("Unfreezing backbone for %s at epoch %d", model_type, epoch)
            model.freeze_backbone(False)

        train_metrics = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = _validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_composite"].append(train_metrics["composite_score"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_composite"].append(val_metrics["composite_score"])

        if val_metrics["composite_score"] > best_composite:
            best_composite = val_metrics["composite_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": config.__dict__,
                },
                best_model_path,
            )
            patience_counter = 0
            logger.info("New best composite=%.4f -> %s", best_composite, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            logger.info("Early stopping %s at epoch %d", model_type, epoch + 1)
            break

    total_time = time.time() - start_time

    return {
        "model_type": model_type,
        "config": config.__dict__,
        "history": history,
        "best_composite": best_composite,
        "best_model_path": str(best_model_path),
        "training_time": total_time,
        "model_info": get_model_info(model),
    }


def grid_search_train(
    model_type: str,
    dataset: Dataset,
    device: torch.device,
    save_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a 3-point hyper-parameter sweep on the given dataset."""
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    if hasattr(train_dataset.dataset, "transform"):
        train_dataset.dataset.transform = build_transforms(train=True)
    if hasattr(val_dataset.dataset, "transform"):
        val_dataset.dataset.transform = build_transforms(train=False)

    configs = [
        HyperParams(lr=5e-5, batch_size=8, weight_decay=1e-4, optimizer="AdamW"),
        HyperParams(lr=1e-4, batch_size=16, weight_decay=1e-4, optimizer="Adam"),
        HyperParams(lr=1e-5, batch_size=8, weight_decay=1e-5, optimizer="AdamW"),
    ]

    best: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []
    for cfg in configs:
        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        result = train_model_with_config(
            model_type, cfg, train_loader, val_loader, device, save_dir
        )
        all_results.append(result)
        if best is None or result["best_composite"] > best["best_composite"]:
            best = result

    assert best is not None
    return best, all_results


def run_zoo(
    data_path: Path = DEFAULT_DATA_PATH,
    models_to_train: list[str] | None = None,
    save_dir: Path | None = None,
    results_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Train every attention-zoo model in sequence and dump per-model logs.

    Args:
        data_path: LabelMe JSON directory.
        models_to_train: Subset of :func:`create_model` keys to train. Defaults
            to the 12 ViT/Swin + attention variants used in the MSc work.
        save_dir: Where checkpoints are written. Defaults to ``MODELS_DIR``.
        results_dir: Where per-model JSON results go. Defaults to
            ``METRICS_DIR / "training_logs"``.

    Returns:
        List of per-model best-result dicts.
    """
    save_dir = save_dir or MODELS_DIR
    results_dir = results_dir or (METRICS_DIR / "training_logs")
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if models_to_train is None:
        models_to_train = [
            "vit_base_16_cbam_head",
            "vit_base_16_se_head",
            "vit_base_16_mha_head",
            "vit_base_16_cbam_stages",
            "vit_base_16_se_stages",
            "vit_base_16_mha_stages",
            "swin_small_cbam_head",
            "swin_small_se_head",
            "swin_small_mha_head",
            "swin_small_cbam_stages",
            "swin_small_se_stages",
            "swin_small_mha_stages",
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    dataset = RealToothDataset(root=data_path, transform=build_transforms(train=False))
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {data_path}")
    logger.info("Loaded %d samples for the grid search", len(dataset))

    all_results: list[dict[str, Any]] = []
    start_time = time.time()
    for model_type in models_to_train:
        try:
            best, _ = grid_search_train(model_type, dataset, device, save_dir)
            all_results.append(best)
            result_file = results_dir / f"{model_type}_training_results.json"
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": best["model_type"],
                        "best_composite": float(best["best_composite"]),
                        "training_time": float(best["training_time"]),
                        "config": best["config"],
                        "best_model_path": best["best_model_path"],
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:  # pragma: no cover — best-effort training
            logger.exception("Training failed for %s: %s", model_type, exc)
            continue

    total_time = time.time() - start_time
    logger.info("All trainings finished in %.1fh", total_time / 3600.0)

    if all_results:
        summary = pd.DataFrame(
            [
                {
                    "Model": r["model_type"],
                    "CompositeScore": r["best_composite"],
                    "TrainingTimeMinutes": r["training_time"] / 60.0,
                }
                for r in all_results
            ]
        ).sort_values("CompositeScore", ascending=False)
        summary.to_csv(results_dir / "training_summary.csv", index=False)
        logger.info("Wrote summary -> %s", results_dir / "training_summary.csv")
    return all_results


__all__ = [
    "CLASS_NAMES",
    "HyperParams",
    "_label_from_string",
    "grid_search_train",
    "run_zoo",
    "train_model_with_config",
]
