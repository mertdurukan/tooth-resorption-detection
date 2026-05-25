"""End-to-end pipeline smoke test (train -> evaluate -> export).

Marked ``@pytest.mark.slow`` because it exercises the full training loop,
the evaluation report writer, and the TorchScript export path on the
in-repo synthetic generator. Run explicitly with::

    pytest -m slow tests/test_pipeline_e2e.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tooth_resorption.config import IMAGE_SIZE, TrainConfig
from tooth_resorption.evaluation.evaluate import evaluate
from tooth_resorption.models.model import build_model
from tooth_resorption.training.train import train


@pytest.mark.slow
def test_train_evaluate_export_pipeline(tmp_results_dir: Path) -> None:
    """Run train -> evaluate -> TorchScript export end-to-end on synthetic data."""
    cfg = TrainConfig(
        model_name="baseline_cnn",
        pretrained=False,
        epochs=1,
        batch_size=4,
        n_per_class_synthetic=6,
        grad_clip=1.0,
    )

    train_out = train(cfg, source="synthetic", data_path=None)
    assert train_out["model_name"] == "baseline_cnn"
    assert train_out["epochs_run"] == 1
    assert 0.0 <= train_out["best_val_f1_macro"] <= 1.0
    best_ckpt_path = Path(train_out["best_checkpoint"])
    assert best_ckpt_path.exists(), "best checkpoint not written by train()"

    eval_out = evaluate(
        model_name="baseline_cnn",
        source="synthetic",
        data_path=None,
        checkpoint=None,
        n_per_class=6,
        batch_size=4,
    )
    assert 0.0 <= eval_out["accuracy"] <= 1.0
    assert 0.0 <= eval_out["f1_macro"] <= 1.0
    assert eval_out["n_samples"] > 0

    model = build_model("baseline_cnn", pretrained=False)
    state = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    export_path = tmp_results_dir / "exported.pt"
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(export_path))
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = torch.jit.load(str(export_path), map_location="cpu")
    reloaded.eval()
    with torch.no_grad():
        logits = reloaded(dummy)
    assert logits.shape == (1, 3)
    assert torch.isfinite(logits).all()
