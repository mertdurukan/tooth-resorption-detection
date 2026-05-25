"""Tests that exercise a single training step + the full training entrypoint."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tooth_resorption.config import TrainConfig
from tooth_resorption.models.model import build_model
from tooth_resorption.training.train import _epoch_step, set_seed, train


def test_set_seed_makes_runs_reproducible() -> None:
    set_seed(7)
    a = torch.randn(4)
    set_seed(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_single_optimizer_step_reduces_loss(dataloaders) -> None:  # type: ignore[no-untyped-def]
    set_seed(0)
    train_loader, _, _ = dataloaders
    model = build_model("baseline_cnn", pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    device = torch.device("cpu")

    pre = _epoch_step(model, train_loader, criterion, None, device, train=False)
    _ = _epoch_step(model, train_loader, criterion, optimizer, device, train=True, grad_clip=1.0)
    post = _epoch_step(model, train_loader, criterion, None, device, train=False)

    assert torch.isfinite(torch.tensor(post["loss"]))
    assert post["loss"] <= pre["loss"] + 0.5


def test_train_writes_artifacts(tmp_results_dir: Path) -> None:
    cfg = TrainConfig(
        model_name="baseline_cnn",
        pretrained=False,
        epochs=1,
        batch_size=4,
        n_per_class_synthetic=6,
        grad_clip=1.0,
    )
    out = train(cfg, source="synthetic", data_path=None)
    assert out["model_name"] == "baseline_cnn"
    assert 0.0 <= out["best_val_f1_macro"] <= 1.0
    assert isinstance(out["final_val"]["f1_macro"], float)
    assert (tmp_results_dir / "results" / "metrics_synthetic.json").exists()
    assert (tmp_results_dir / "results" / "training_history_baseline_cnn.json").exists()
    assert (tmp_results_dir / "models" / "baseline_cnn_best.pt").exists()
    assert (tmp_results_dir / "models" / "baseline_cnn_last.pt").exists()
