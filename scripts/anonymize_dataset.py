"""Anonymise patient-identifying file names in the YOLO dataset.

The clinical wisdom-tooth dataset from Mersin University ships with raw image
file names that are derived from patient names (e.g. ``ahmet_yilmaz_3.jpg``).
This violates KVKK / GDPR and must never leak through git history, CI logs,
or model artefacts.

This script renames every ``*.jpg`` / ``*.jpeg`` / ``*.png`` image plus the
matching ``*.txt`` YOLO label into the deterministic pattern::

    patient_<4-digit-zero-padded-index>.<ext>

A JSON mapping ``original_name -> new_name`` is written to
``anonymization_map.json`` next to the dataset so the renaming is reversible
internally. The mapping is gitignored by default.

The script is **idempotent**: filenames that already match the anonymised
pattern are skipped, and re-running on an already-anonymised dataset is a
no-op.

Usage::

    python scripts/anonymize_dataset.py
    python scripts/anonymize_dataset.py --dataset path/to/yolo_dataset --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path

LOGGER = logging.getLogger("anonymize_dataset")

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
ANONYMISED_PATTERN = re.compile(r"^patient_\d{4,}$")

DEFAULT_DATASET = Path("data/processed/yolo_dataset")
DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")


def _iter_images(dataset_root: Path, splits: Iterable[str]) -> list[Path]:
    """Return every image file under ``<dataset_root>/<split>/images``."""
    images: list[Path] = []
    for split in splits:
        images_dir = dataset_root / split / "images"
        if not images_dir.is_dir():
            LOGGER.debug("Skipping missing split directory: %s", images_dir)
            continue
        for ext in IMAGE_EXTENSIONS:
            images.extend(sorted(images_dir.glob(f"*{ext}")))
    return images


def _is_anonymised(stem: str) -> bool:
    """Return True iff ``stem`` already matches the anonymised pattern."""
    return bool(ANONYMISED_PATTERN.match(stem))


def anonymise(
    dataset_root: Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    map_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Rename every image + matching label to a deterministic patient ID.

    Args:
        dataset_root: Path to the YOLO dataset root (the directory that
            contains ``train/``, ``val/``, optionally ``test/``).
        splits: Iterable of split names to walk.
        map_path: Destination for the JSON mapping. Defaults to
            ``<dataset_root>/anonymization_map.json``.
        dry_run: If True, log the planned operations but do not touch disk.

    Returns:
        Mapping ``original_filename -> new_filename`` for files that were
        renamed in this invocation (skipped files are not included).
    """
    splits = tuple(splits)
    map_path = map_path or (dataset_root / "anonymization_map.json")

    existing_map: dict[str, str] = {}
    if map_path.exists():
        try:
            existing_map = json.loads(map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Could not parse existing mapping at %s; starting fresh.", map_path)
            existing_map = {}

    images = _iter_images(dataset_root, splits)
    LOGGER.info("Scanned %d image(s) across splits=%s", len(images), splits)

    used_indices: set[int] = set()
    for new_name in existing_map.values():
        match = re.match(r"^patient_(\d+)\.[A-Za-z0-9]+$", new_name)
        if match:
            used_indices.add(int(match.group(1)))

    next_index = (max(used_indices) + 1) if used_indices else 1
    renamed: dict[str, str] = {}
    skipped_already_anon = 0

    for image_path in images:
        if _is_anonymised(image_path.stem):
            skipped_already_anon += 1
            continue

        new_stem = f"patient_{next_index:04d}"
        next_index += 1

        new_image_name = new_stem + image_path.suffix.lower()
        new_image_path = image_path.with_name(new_image_name)

        label_dir = image_path.parent.parent / "labels"
        label_path = label_dir / (image_path.stem + ".txt")
        new_label_path = label_dir / (new_stem + ".txt")

        LOGGER.info("%s -> %s", image_path.name, new_image_name)
        if not dry_run:
            image_path.rename(new_image_path)
            if label_path.exists():
                label_path.rename(new_label_path)
            else:
                LOGGER.warning("Missing label for %s (expected %s)", image_path.name, label_path)

        renamed[image_path.name] = new_image_name
        if label_path.exists() or new_label_path.exists():
            renamed[image_path.stem + ".txt"] = new_stem + ".txt"

    LOGGER.info(
        "Renamed %d file(s); skipped %d already-anonymised image(s).",
        len(renamed),
        skipped_already_anon,
    )

    if renamed and not dry_run:
        merged = {**existing_map, **renamed}
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        LOGGER.info("Mapping written to %s", map_path)

    return renamed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"YOLO dataset root (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=f"Split directories to walk (default: {' '.join(DEFAULT_SPLITS)}).",
    )
    parser.add_argument(
        "--map-path",
        type=Path,
        default=None,
        help="Destination for the JSON mapping (default: <dataset>/anonymization_map.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the planned renames without touching disk.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.dataset.is_dir():
        LOGGER.error("Dataset directory does not exist: %s", args.dataset)
        return 2
    anonymise(
        args.dataset,
        splits=args.splits,
        map_path=args.map_path,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
