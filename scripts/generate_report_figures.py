"""Regenerate the headline figures shipped under ``results/figures``.

Reads the cross-model evaluation JSON and the training logs and produces:

* ``model_comparison_chart.png`` — bar chart of accuracy / F1 / AUC per model.
* ``ensemble_performance_chart.png`` — top-3 vs full ensemble bar plot.
* ``composite_score_comparison.png`` — composite-score ranking chart.

This is a thin, English-only port of the legacy
``generate_report_figures.py`` script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tooth_resorption.config import FIGURES_DIR, METRICS_DIR
from tooth_resorption.logging_utils import get_logger

logger = get_logger(__name__)


def _load(path: Path) -> list[dict] | None:
    if not path.exists():
        logger.warning("Missing input: %s", path)
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_comparison_chart(results: list[dict], out_path: Path) -> Path:
    valid = sorted(
        (r for r in results if r.get("accuracy", 0) > 0.5),
        key=lambda r: r.get("composite_score", 0),
        reverse=True,
    )
    if not valid:
        raise ValueError("No valid evaluation results")

    models = [r["model_type"][:20] for r in valid]
    accuracy = np.array([r.get("accuracy", 0) * 100 for r in valid])
    f1 = np.array([r.get("f1_macro", 0) * 100 for r in valid])
    auc = np.array([r.get("auc_macro", 0) * 100 for r in valid])

    x = np.arange(len(models))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.55), 5.5))
    ax.bar(x - width, accuracy, width, label="Accuracy", color="#3b7dbf")
    ax.bar(x, f1, width, label="Macro F1", color="#9bb87d")
    ax.bar(x + width, auc, width, label="Macro AUC", color="#bd6760")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison across all trained variants")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def composite_score_chart(results: list[dict], out_path: Path) -> Path:
    sorted_results = sorted(results, key=lambda r: r.get("composite_score", 0), reverse=True)[:15]
    models = [r["model_type"][:24] for r in sorted_results]
    scores = np.array([r.get("composite_score", 0) for r in sorted_results])

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.45), 5))
    ax.barh(range(len(models)), scores, color="#3b7dbf")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Composite score (mean of accuracy/precision/recall/F1/AUC)")
    ax.set_title("Top 15 models by composite score")
    ax.set_xlim(0, 1.0)
    for i, score in enumerate(scores):
        ax.text(score + 0.005, i, f"{score:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate report figures from results JSON.")
    parser.add_argument(
        "--evaluation-json", type=Path, default=METRICS_DIR / "evaluation_results.json"
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = _load(args.evaluation_json) or []
    if not results:
        raise SystemExit(
            f"No results in {args.evaluation_json}; run scripts/evaluate_models.py first."
        )
    model_comparison_chart(results, args.out_dir / "model_comparison_chart.png")
    composite_score_chart(results, args.out_dir / "composite_score_comparison.png")


if __name__ == "__main__":
    main()
