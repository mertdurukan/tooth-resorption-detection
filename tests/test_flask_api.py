"""Flask REST API smoke tests.

The real ``ToothResorptionInference`` constructor calls ``torch.jit.load`` on
a TorchScript artefact that is not checked into the repo, so importing
``deployment.api.flask_api`` would fail at module-import time in CI. We
monkey-patch a lightweight stub before the first import.
"""

from __future__ import annotations

import importlib
import io
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


class _StubInference:
    """Minimal stand-in for :class:`ToothResorptionInference` used in tests."""

    DEFAULT_CLASS_NAMES = ("temasli", "bagimsiz", "rezorpsiyon")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.model_path = kwargs.get("model_path")
        self.device = kwargs.get("device", "cpu")

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        return {
            "predicted_class": "temasli",
            "predicted_class_index": 0,
            "confidence": 0.95,
            "probabilities": {"temasli": 0.95, "bagimsiz": 0.03, "rezorpsiyon": 0.02},
            "model_name": "stub",
        }

    def predict_batch(self, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        return [self.predict(p) for p in image_paths]


@pytest.fixture()
def flask_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Yield a freshly-imported Flask app whose inference backend is stubbed."""
    monkeypatch.setenv("TRD_MODEL_PATH", str(tmp_path / "stub.pt"))
    monkeypatch.setenv("TRD_UPLOAD_FOLDER", str(tmp_path / "uploads"))

    stub_module = types.ModuleType("deployment.api.inference")
    stub_module.ToothResorptionInference = _StubInference  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deployment.api.inference", stub_module)

    sys.modules.pop("deployment.api.flask_api", None)
    flask_api = importlib.import_module("deployment.api.flask_api")
    flask_api.app.config["TESTING"] = True
    yield flask_api.app
    sys.modules.pop("deployment.api.flask_api", None)


@pytest.fixture()
def client(flask_app: Any) -> Any:
    return flask_app.test_client()


def test_health_endpoint(client: Any) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "healthy"


def test_predict_endpoint_rejects_missing_file(client: Any) -> None:
    resp = client.post("/predict")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_endpoint_returns_stub_payload(client: Any, tmp_path: Path) -> None:
    image_bytes = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\t\x08"
        b"\xff\xd9"
    )
    data = {"file": (io.BytesIO(image_bytes), "test.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["predicted_class"] in {"temasli", "bagimsiz", "rezorpsiyon"}
    assert 0.0 <= payload["confidence"] <= 1.0


def test_predict_batch_rejects_empty(client: Any) -> None:
    resp = client.post("/predict_batch")
    assert resp.status_code == 400
    assert "error" in resp.get_json()
