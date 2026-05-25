"""Evaluate every available model and write a unified comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

from tooth_resorption.config import METRICS_DIR, MODELS_DIR
from tooth_resorption.evaluation.model_comparison import (
    DEFAULT_DATA_PATH,
    UnifiedModelEvaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN + YOLO + attention models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = UnifiedModelEvaluator(data_path=args.data_path)
    results = evaluator.evaluate_all(models_dir=args.models_dir)
    df = evaluator.save_results(output_dir=args.output_dir)
    print(df.to_string(index=False))
    if not results:
        raise SystemExit("No models evaluated.")


if __name__ == "__main__":
    main()
