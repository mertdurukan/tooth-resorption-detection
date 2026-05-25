"""Train the YOLOv11 tooth-resorption detector from the LabelMe JSON dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tooth_resorption.detection.yolo_detector import (
    DEFAULT_DATA_PATH,
    DEFAULT_DATASET_DIR,
    YoloToothDetector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11 on the tooth-resorption dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--name", type=str, default="tooth_resorption")
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation (use existing dataset_dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YoloToothDetector(data_path=args.data_path, dataset_dir=args.dataset_dir)
    if not args.skip_prepare:
        detector.prepare_dataset()
    detector.train(
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        weights=args.weights,
    )
    detector.evaluate()


if __name__ == "__main__":
    main()
