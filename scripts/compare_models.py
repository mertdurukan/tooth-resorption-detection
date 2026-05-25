"""Pretty-print and re-render the cross-model comparison table.

Reads ``results/metrics/evaluation_results.json`` (produced by
``scripts/evaluate_models.py``) and ``results/metrics.json`` (the headline
baseline-vs-improved numbers), then writes the comparison bar chart to
``results/plots/comparison.png`` using
:func:`tooth_resorption.visualization.plot_comparison.render`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tooth_resorption.config import METRICS_DIR, RESULTS_DIR
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.visualization.plot_comparison import render

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render comparison plot and print table.")
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=METRICS_DIR / "evaluation_results.json",
    )
    parser.add_argument("--metrics-json", type=Path, default=RESULTS_DIR / "metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.metrics_json.exists():
        out_png = render(metrics_path=args.metrics_json)
        logger.info("Rendered comparison plot -> %s", out_png)
    else:
        logger.warning("Missing metrics file: %s", args.metrics_json)

    if not args.evaluation_json.exists():
        logger.warning("No cross-model evaluation results found at %s", args.evaluation_json)
        return

    with args.evaluation_json.open("r", encoding="utf-8") as f:
        results = json.load(f)

    rows = [
        {
            "Model": r["model_type"],
            "Category": r["model_category"],
            "Accuracy": round(r["accuracy"], 4),
            "F1Macro": round(r["f1_macro"], 4),
            "AUCMacro": round(r["auc_macro"], 4),
            "CompositeScore": round(r["composite_score"], 4),
            "InferenceMs": round(r["inference_time_ms"], 3),
            "SizeMb": round(r.get("model_size_mb", 0.0), 2),
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("CompositeScore", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
