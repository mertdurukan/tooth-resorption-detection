"""Train the full attention-zoo grid search on the LabelMe dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tooth_resorption.training.train_attention import DEFAULT_DATA_PATH, run_zoo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all attention/transformer variants.")
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_DATA_PATH, help="LabelMe JSON directory."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model_type keys (defaults to the full 12-model zoo).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_zoo(data_path=args.data_path, models_to_train=args.models)


if __name__ == "__main__":
    main()
