"""Evaluation entrypoint.

Loads a trained checkpoint (or, when no checkpoint is found, instantiates an
untrained model just to verify the pipeline) and produces:

* macro / micro F1, macro precision / recall, accuracy, macro AUC
* per-class precision, recall, F1
* a ``sklearn`` classification report saved as
  ``results/classification_report_{model}.txt``
* a confusion matrix PNG at ``results/plots/confusion_matrix.png``
* an inference-time benchmark (ms/image at batch=1)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from tooth_resorption.config import (
    MODELS_DIR,
    NUM_CLASSES,
    PLOTS_DIR,
    RESULTS_DIR,
    DataSource,
    ModelName,
)
from tooth_resorption.data.data_loader import build_dataloaders
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.models.model import build_model

logger = get_logger(__name__)


def _load_model(model_name: ModelName, checkpoint: Path | None) -> torch.nn.Module:
    model = build_model(model_name, pretrained=False)
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        weights = state.get("state_dict", state)
        model.load_state_dict(weights, strict=False)
        logger.info("Loaded checkpoint from %s", checkpoint)
    else:
        logger.warning("No checkpoint found; evaluating an untrained model.")
    model.eval()
    return model


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _benchmark_inference(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
    n_warmup: int = 3,
    n_iters: int = 20,
) -> float:
    """Return average single-image inference time in milliseconds."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return float(1000.0 * elapsed / n_iters)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Macro one-vs-rest AUC; falls back to 0.5 when only one class is present."""
    try:
        return float(
            roc_auc_score(
                labels,
                probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(NUM_CLASSES)),
            )
        )
    except ValueError:
        return 0.5


def evaluate(
    model_name: ModelName,
    source: DataSource,
    data_path: Path | None,
    checkpoint: Path | None,
    n_per_class: int = 24,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run evaluation on the validation split and return a metrics dict.

    Side effects:
        * Writes ``results/evaluation_synthetic.json``.
        * Writes ``results/classification_report_{model}.txt``.
        * Writes ``results/plots/confusion_matrix.png``.
    """
    _, val_loader, class_names = build_dataloaders(
        source=source,
        data_path=data_path,
        n_per_class=n_per_class,
        batch_size=batch_size,
        val_split=0.5,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_name, checkpoint).to(device)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    preds = np.asarray(all_preds)
    labels = np.asarray(all_labels)
    probs = np.asarray(all_probs)

    precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    inference_ms = _benchmark_inference(model, device)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(cm, list(class_names), PLOTS_DIR / "confusion_matrix.png")

    report_str = classification_report(
        labels,
        preds,
        target_names=list(class_names),
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
        digits=4,
    )
    report_path = RESULTS_DIR / f"classification_report_{model_name}.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_str, encoding="utf-8")

    out: dict[str, Any] = {
        "model_name": model_name,
        "source": source,
        "n_samples": len(labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "auc_macro": _safe_auc(labels, probs),
        "inference_time_ms": round(inference_ms, 3),
        "per_class": {
            class_names[i]: {
                "precision": float(precision_pc[i]),
                "recall": float(recall_pc[i]),
                "f1": float(f1_pc[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report_path": str(report_path),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    }
    out_path = RESULTS_DIR / "evaluation_synthetic.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote evaluation results -> %s", out_path)
    logger.info(
        "macro-F1=%.3f | accuracy=%.3f | AUC=%.3f | inference=%.2f ms",
        out["f1_macro"],
        out["accuracy"],
        out["auc_macro"],
        out["inference_time_ms"],
    )
    return out


def parse_args() -> argparse.Namespace:
    """CLI parser for :mod:`tooth_resorption.evaluation.evaluate`."""
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--data", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=("baseline_cnn", "vit_tiny", "vit_base"),
        default="baseline_cnn",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a .pt checkpoint produced by the trainer.",
    )
    parser.add_argument("--n-per-class", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = cast(ModelName, args.model)
    ckpt = args.checkpoint or (MODELS_DIR / f"{model_name}_best.pt")
    if not ckpt.exists():
        ckpt = None
    evaluate(
        model_name=model_name,
        source=cast(DataSource, args.data),
        data_path=args.data_path,
        checkpoint=ckpt,
        n_per_class=args.n_per_class,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


__all__ = ["evaluate", "main", "parse_args"]
