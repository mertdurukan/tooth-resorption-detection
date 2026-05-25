"""Unit tests for :mod:`tooth_resorption.data.synthetic_data`."""

from __future__ import annotations

import pytest
from PIL import Image
from tooth_resorption.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from tooth_resorption.data.synthetic_data import generate


def test_generate_returns_correct_count() -> None:
    images, labels = generate(n_per_class=4, seed=7)
    assert len(images) == NUM_CLASSES * 4
    assert len(labels) == NUM_CLASSES * 4


def test_generate_covers_all_classes() -> None:
    _, labels = generate(n_per_class=3, seed=7)
    assert set(labels) == set(range(NUM_CLASSES))
    assert len(CLASS_NAMES) == NUM_CLASSES


def test_generate_image_shape_and_mode() -> None:
    images, _ = generate(n_per_class=1, seed=7)
    sample = images[0]
    assert isinstance(sample, Image.Image)
    assert sample.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert sample.mode == "RGB"


def test_generate_is_deterministic_with_seed() -> None:
    images_a, labels_a = generate(n_per_class=2, seed=42)
    images_b, labels_b = generate(n_per_class=2, seed=42)
    assert labels_a == labels_b
    assert images_a[0].tobytes() == images_b[0].tobytes()


def test_generate_different_seed_changes_output() -> None:
    images_a, _ = generate(n_per_class=2, seed=1)
    images_b, _ = generate(n_per_class=2, seed=2)
    assert images_a[0].tobytes() != images_b[0].tobytes()


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_rejects_invalid_n_per_class(bad: int) -> None:
    with pytest.raises(ValueError):
        generate(n_per_class=bad, seed=0)
