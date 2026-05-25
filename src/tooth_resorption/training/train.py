"""Training entrypoint.

Default invocation::

    python -m tooth_resorption.training.train --data synthetic

Real-data run with mixed precision, gradient clipping and early stopping::

    python -m tooth_resorption.training.train \\
        --data real --data-path /path/to/labelme/json/dir \\
        --model vit_base --pretrained --epochs 50 \\
        --amp --clip 1.0 --early-stopping 15

The synthetic run is intentionally tiny so the smoke test finishes in under a
minute on CPU. The shipped numbers in ``results/metrics.json`` are NOT
produced by this script — they come from the original MSc experiments on the
real clinical dataset. The synthetic run writes to
``results/metrics_synthetic.json`` so the published metrics are never
overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from tooth_resorption.config import (
    CLASS_NAMES,
    MODELS_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
    SEED,
    DataSource,
    ModelName,
    TrainConfig,
)
from tooth_resorption.data.data_loader import build_dataloaders
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.models.model import build_model, count_parameters


def _try_import_mlflow() -> Any:
    try:
        import mlflow as _mlflow
    except ImportError:
        return None
    return _mlflow


mlflow: Any = _try_import_mlflow()

logger = get_logger(__name__)


def _mlflow_enabled() -> bool:
    """Return True iff MLflow tracking is available and not explicitly disabled."""
    return mlflow is not None and os.environ.get("TRD_DISABLE_MLFLOW") != "1"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed ``random``, NumPy and PyTorch (CPU + CUDA) for reproducibility.

    When ``deterministic`` is True, also enables PyTorch's deterministic
    algorithms and the cuBLAS workspace configuration required for fully
    reproducible CUDA matmul. This costs throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _epoch_step(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    *,
    train: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Run a single train or eval pass over ``loader``.

    Args:
        model: Network in train- or eval-mode.
        loader: DataLoader yielding ``(image, label)`` batches.
        criterion: Loss module (e.g. ``CrossEntropyLoss``).
        optimizer: Required iff ``train`` is True.
        device: Target torch device.
        train: True for gradient step; False for evaluation.
        scaler: ``GradScaler`` for mixed-precision training (CUDA only).
        grad_clip: L2-norm gradient clip value; ``None`` disables.

    Returns:
        Dict of average loss + macro precision/recall/F1.
    """
    model.train(mode=train)
    losses: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    use_amp = scaler is not None and device.type == "cuda"
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    with grad_ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            autocast_ctx: Any = torch.cuda.amp.autocast() if use_amp else nullcontext()
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            losses.append(float(loss.detach().item()))
            all_preds.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(all_labels, all_preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(all_labels, all_preds, average="macro", zero_division=0)
        ),
    }


def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically dump a full training checkpoint to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "model_name": cfg.model_name,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "config": {
            "model_name": cfg.model_name,
            "pretrained": cfg.pretrained,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "amp": cfg.amp,
            "grad_clip": cfg.grad_clip,
        },
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _maybe_resume(
    model: nn.Module,
    optimizer: optim.Optimizer,
    resume_path: Path | None,
    device: torch.device,
) -> int:
    """Reload model + optimizer state from ``resume_path``; returns start epoch."""
    if resume_path is None:
        return 1
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume points to a missing file: {resume_path}")
    state = torch.load(resume_path, map_location=device)
    model.load_state_dict(state["state_dict"], strict=False)
    if "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError) as exc:
            logger.warning("Could not restore optimizer state: %s", exc)
    start = int(state.get("epoch", 0)) + 1
    logger.info("Resumed from %s at epoch %d", resume_path, start)
    return start


def train(
    cfg: TrainConfig,
    source: DataSource,
    data_path: Path | None,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Run the full training loop and return the final validation metrics.

    When MLflow is installed and ``TRD_DISABLE_MLFLOW`` is unset, the run is
    tracked under the ``MLFLOW_EXPERIMENT_NAME`` experiment (default
    ``tooth-resorption``). Hyper-parameters are logged once at the start;
    per-epoch ``train_*`` / ``val_*`` scalars are logged at every epoch; the
    best checkpoint is logged as an artefact at the end.
    """
    set_seed(SEED, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | AMP: %s | deterministic: %s", device, cfg.amp, cfg.deterministic)

    train_loader, val_loader, _ = build_dataloaders(
        source=source,
        data_path=data_path,
        n_per_class=cfg.n_per_class_synthetic,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        num_workers=cfg.num_workers,
        seed=SEED,
    )

    model = build_model(cfg.model_name, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type == "cuda") else None

    start_epoch = _maybe_resume(model, optimizer, resume, device)
    logger.info(
        "model=%s params=%s source=%s start_epoch=%d",
        cfg.model_name,
        f"{count_parameters(model):,}",
        source,
        start_epoch,
    )

    use_mlflow = _mlflow_enabled()
    if use_mlflow:
        assert mlflow is not None
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "tooth-resorption")
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=f"{cfg.model_name}-{source}")
        mlflow.log_params(
            {
                "model_name": cfg.model_name,
                "source": source,
                "pretrained": cfg.pretrained,
                "epochs_target": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "amp": cfg.amp,
                "grad_clip": cfg.grad_clip,
                "deterministic": cfg.deterministic,
                "num_classes": NUM_CLASSES,
                "seed": SEED,
                "parameters": count_parameters(model),
            }
        )

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_path = MODELS_DIR / f"{cfg.model_name}_best.pt"
    last_path = MODELS_DIR / f"{cfg.model_name}_last.pt"

    start = time.time()
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_metrics = _epoch_step(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
                scaler=scaler,
                grad_clip=cfg.grad_clip,
            )
            val_metrics = _epoch_step(
                model,
                val_loader,
                criterion,
                None,
                device,
                train=False,
            )
            scheduler.step(val_metrics["f1_macro"])
            lr = optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics})
            logger.info(
                "E%02d | lr=%.2e | train loss=%.3f f1=%.3f | val loss=%.3f f1=%.3f",
                epoch,
                lr,
                train_metrics["loss"],
                train_metrics["f1_macro"],
                val_metrics["loss"],
                val_metrics["f1_macro"],
            )

            if use_mlflow:
                assert mlflow is not None
                mlflow.log_metrics(
                    {
                        "lr": lr,
                        "train_loss": train_metrics["loss"],
                        "train_f1_macro": train_metrics["f1_macro"],
                        "train_precision_macro": train_metrics["precision_macro"],
                        "train_recall_macro": train_metrics["recall_macro"],
                        "val_loss": val_metrics["loss"],
                        "val_f1_macro": val_metrics["f1_macro"],
                        "val_precision_macro": val_metrics["precision_macro"],
                        "val_recall_macro": val_metrics["recall_macro"],
                    },
                    step=epoch,
                )

            _save_checkpoint(model, optimizer, epoch, cfg, last_path)
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                epochs_no_improve = 0
                _save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    cfg,
                    best_path,
                    extra={"best_val_f1": best_val_f1},
                )
                logger.info("New best val macro-F1=%.3f -> %s", best_val_f1, best_path)
            else:
                epochs_no_improve += 1
                if cfg.early_stopping_patience and epochs_no_improve >= cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        epochs_no_improve,
                    )
                    break

        elapsed = time.time() - start
        final_val = history[-1]["val"] if history else {}

        out: dict[str, Any] = {
            "model_name": cfg.model_name,
            "source": source,
            "epochs_run": len(history),
            "epochs_target": cfg.epochs,
            "training_time_seconds": round(elapsed, 2),
            "best_val_f1_macro": best_val_f1,
            "final_val": final_val,
            "history": history,
            "best_checkpoint": str(best_path) if best_val_f1 >= 0 else None,
            "last_checkpoint": str(last_path),
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        smoke_path = RESULTS_DIR / "metrics_synthetic.json"
        history_path = RESULTS_DIR / f"training_history_{cfg.model_name}.json"
        with smoke_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        with history_path.open("w", encoding="utf-8") as f:
            json.dump({"model": cfg.model_name, "history": history}, f, indent=2)
        logger.info("Wrote smoke metrics -> %s", smoke_path)
        logger.info("Wrote training history -> %s", history_path)

        if use_mlflow:
            assert mlflow is not None
            mlflow.log_metric("best_val_f1_macro", best_val_f1)
            mlflow.log_metric("training_time_seconds", elapsed)
            try:
                mlflow.log_artifact(str(history_path))
                if best_val_f1 >= 0 and best_path.exists():
                    mlflow.log_artifact(str(best_path))
            except Exception as exc:
                logger.warning("MLflow artefact logging failed: %s", exc)

        return out
    finally:
        if use_mlflow:
            assert mlflow is not None
            mlflow.end_run()


def parse_args() -> argparse.Namespace:
    """CLI parser for :mod:`tooth_resorption.training.train`."""
    parser = argparse.ArgumentParser(
        description="Train the wisdom-tooth-resorption classifier.",
    )
    parser.add_argument("--data", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to LabelMe JSON directory (required when --data real).",
    )
    parser.add_argument(
        "--model",
        choices=("baseline_cnn", "vit_tiny", "vit_base"),
        default="baseline_cnn",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet weights (ViT only; requires network access).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=24,
        help="Synthetic samples per class.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision.")
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Gradient L2-norm clip value (set to 0 to disable).",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help="Stop after N epochs without val macro-F1 improvement (0 disables).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable PyTorch deterministic algorithms (slower).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint produced by a previous run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model_name=cast(ModelName, args.model),
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        n_per_class_synthetic=args.n_per_class,
        amp=args.amp,
        grad_clip=args.clip if args.clip > 0 else None,
        early_stopping_patience=args.early_stopping,
        deterministic=args.deterministic,
    )
    train(cfg, source=cast(DataSource, args.data), data_path=args.data_path, resume=args.resume)


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args", "set_seed", "train"]
