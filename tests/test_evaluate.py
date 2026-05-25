"""Tests for :mod:`tooth_resorption.evaluation.evaluate` and the comparison plot."""

from __future__ import annotations

from pathlib import Path

from tooth_resorption.evaluation.evaluate import evaluate
from tooth_resorption.visualization.plot_comparison import render


def test_evaluate_returns_expected_keys(tmp_results_dir: Path) -> None:
    out = evaluate(
        model_name="baseline_cnn",
        source="synthetic",
        data_path=None,
        checkpoint=None,
        n_per_class=6,
        batch_size=4,
    )
    expected_top = {
        "model_name",
        "source",
        "n_samples",
        "accuracy",
        "f1_macro",
        "f1_micro",
        "auc_macro",
        "inference_time_ms",
        "per_class",
        "confusion_matrix",
    }
    assert expected_top.issubset(out.keys())


def test_evaluate_metrics_in_valid_range(tmp_results_dir: Path) -> None:
    out = evaluate(
        model_name="baseline_cnn",
        source="synthetic",
        data_path=None,
        checkpoint=None,
        n_per_class=6,
        batch_size=4,
    )
    for key in ("accuracy", "f1_macro", "f1_micro", "auc_macro"):
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
    assert out["inference_time_ms"] > 0.0
    assert set(out["per_class"].keys()) == {"temasli", "bagimsiz", "rezorpsiyon"}
    for stats in out["per_class"].values():
        for k in ("precision", "recall", "f1"):
            assert 0.0 <= stats[k] <= 1.0
        assert stats["support"] >= 0


def test_evaluate_writes_confusion_matrix_and_report(tmp_results_dir: Path) -> None:
    out = evaluate(
        model_name="baseline_cnn",
        source="synthetic",
        data_path=None,
        checkpoint=None,
        n_per_class=6,
        batch_size=4,
    )
    cm_path = tmp_results_dir / "results" / "plots" / "confusion_matrix.png"
    report_path = tmp_results_dir / "results" / "classification_report_baseline_cnn.txt"
    assert cm_path.exists()
    assert report_path.exists()
    assert out["classification_report_path"].endswith("classification_report_baseline_cnn.txt")


def test_plot_comparison_renders(tmp_path: Path) -> None:
    out = render(out_path=tmp_path / "cmp.png")
    assert out.exists()
    assert out.suffix == ".png"
