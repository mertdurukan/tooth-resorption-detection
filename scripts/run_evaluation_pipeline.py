"""Master pipeline: evaluate all models, regenerate figures, export the best.

Replaces the legacy ``run_evaluation_pipeline.py`` orchestrator with a
non-interactive, English-only version that delegates to the Python entry
points (no ``subprocess`` shelling out).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tooth_resorption.config import METRICS_DIR, MODELS_DIR
from tooth_resorption.evaluation.model_comparison import (
    DEFAULT_DATA_PATH,
    UnifiedModelEvaluator,
)
from tooth_resorption.logging_utils import get_logger
from tooth_resorption.visualization.plot_comparison import render

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("[Step 1/3] Evaluating all available models")
    evaluator = UnifiedModelEvaluator(data_path=args.data_path)
    results = evaluator.evaluate_all(models_dir=args.models_dir)
    if not results:
        raise SystemExit("No models evaluated; train at least one model first.")
    df = evaluator.save_results(output_dir=args.output_dir)
    logger.info(
        "Best model: %s (composite=%.4f)", df.iloc[0]["Model"], df.iloc[0]["CompositeScore"]
    )

    logger.info("[Step 2/3] Re-rendering headline comparison figure")
    try:
        render()
    except FileNotFoundError as exc:
        logger.warning("Cannot render comparison.png: %s", exc)

    logger.info("[Step 3/3] Done")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
