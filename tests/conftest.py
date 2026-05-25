"""Shared pytest fixtures.

Centralises the fixtures used across the test suite:

* ``synthetic_images_labels`` — a tiny deterministic synthetic set.
* ``dataloaders`` — train/val DataLoaders backed by the synthetic generator.
* ``baseline_model`` / ``vit_tiny_model`` — instantiated models on CPU.
* ``tmp_results_dir`` — redirects ``RESULTS_DIR`` / ``MODELS_DIR`` / ``PLOTS_DIR``
  at module-level so tests never touch the published artefacts.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TRD_DISABLE_MLFLOW", "1")

from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import pytest
import torch
from PIL import Image
from tooth_resorption import config as cfg_module
from tooth_resorption.data.data_loader import build_dataloaders
from tooth_resorption.data.synthetic_data import generate
from tooth_resorption.models.model import build_model

torch.manual_seed(0)


@pytest.fixture(scope="session")
def synthetic_images_labels() -> tuple[list[Image.Image], list[int]]:
    """Return a fixed, small synthetic dataset (deterministic, fast)."""
    return generate(n_per_class=3, seed=123)


@pytest.fixture()
def dataloaders():  # type: ignore[no-untyped-def]
    """Return ``(train_loader, val_loader, class_names)`` for a tiny smoke run."""
    return build_dataloaders(
        source="synthetic", n_per_class=6, batch_size=4, val_split=0.25, seed=42
    )


@pytest.fixture(scope="session")
def baseline_model() -> torch.nn.Module:
    return build_model("baseline_cnn", pretrained=False)


@pytest.fixture(scope="session")
def vit_tiny_model() -> torch.nn.Module:
    return build_model("vit_tiny", pretrained=False)


@pytest.fixture()
def tmp_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect every output path on :mod:`tooth_resorption.config` to ``tmp_path``.

    Both submodules import the path constants into their own globals, so
    patching ``tooth_resorption.config`` alone is not enough — we must also
    patch the submodule globals where ``RESULTS_DIR`` / ``MODELS_DIR`` were
    bound at import time.

    ``from tooth_resorption.training import train`` resolves to the function
    re-exported by ``__init__.py`` (not the module), so the modules are
    pulled directly out of ``sys.modules`` here to make the ``monkeypatch``
    calls actually land on module globals.
    """
    import importlib
    import sys

    results = tmp_path / "results"
    plots = results / "plots"
    models = tmp_path / "models"
    for d in (results, plots, models):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg_module, "RESULTS_DIR", results)
    monkeypatch.setattr(cfg_module, "PLOTS_DIR", plots)
    monkeypatch.setattr(cfg_module, "MODELS_DIR", models)

    importlib.import_module("tooth_resorption.training.train")
    importlib.import_module("tooth_resorption.evaluation.evaluate")
    train_mod = sys.modules["tooth_resorption.training.train"]
    eval_mod = sys.modules["tooth_resorption.evaluation.evaluate"]

    monkeypatch.setattr(train_mod, "RESULTS_DIR", results, raising=False)
    monkeypatch.setattr(train_mod, "MODELS_DIR", models, raising=False)
    monkeypatch.setattr(eval_mod, "RESULTS_DIR", results, raising=False)
    monkeypatch.setattr(eval_mod, "PLOTS_DIR", plots, raising=False)
    monkeypatch.setattr(eval_mod, "MODELS_DIR", models, raising=False)
    yield tmp_path
