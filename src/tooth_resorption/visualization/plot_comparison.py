"""Render the baseline-vs-improved comparison bar chart.

Reads ``results/metrics.json`` and writes
``results/plots/comparison_smoke.png``. The figure is regenerated whenever
``metrics.json`` changes.

NOTE: The slide-style benchmark chart shipped at
``results/plots/comparison.png`` is produced by
``scripts/generate_thesis_figures.py`` (function ``figure_slide``) from the
real benchmark table. This module is the legacy smoke-pipeline renderer and
intentionally writes to a different filename to avoid clobbering it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tooth_resorption.config import PLOTS_DIR, RESULTS_DIR
from tooth_resorption.logging_utils import get_logger

logger = get_logger(__name__)


def render(metrics_path: Path | None = None, out_path: Path | None = None) -> Path:
    """Render the comparison bar chart.

    Args:
        metrics_path: Defaults to ``results/metrics.json``.
        out_path: Defaults to ``results/plots/comparison_smoke.png``.

    Returns:
        Path to the written PNG.
    """
    metrics_path = metrics_path or (RESULTS_DIR / "metrics.json")
    out_path = out_path or (PLOTS_DIR / "comparison_smoke.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = ("f1_macro", "precision_macro", "recall_macro", "accuracy", "auc_macro")
    labels = ("Macro F1", "Precision", "Recall", "Accuracy", "AUC")
    baseline = [data["baseline"][m] for m in metrics]
    improved = [data["improved"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars_b = ax.bar(
        x - width / 2,
        baseline,
        width,
        label=data["baseline"]["model"].split("(")[0].strip(),
        color="#bd6760",
    )
    bars_i = ax.bar(
        x + width / 2,
        improved,
        width,
        label=data["improved"]["model"].split("(")[0].strip(),
        color="#3b7dbf",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs improved — wisdom tooth resorption detection")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")

    for bars in (bars_b, bars_i):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return out_path


def main() -> None:
    render()


if __name__ == "__main__":
    main()


__all__ = ["main", "render"]
