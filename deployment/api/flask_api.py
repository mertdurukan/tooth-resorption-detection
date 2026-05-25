"""Flask REST API for tooth-resorption inference.

The model path defaults to ``deployment/torchscript/vit_base_16.pt`` and
can be overridden with the ``TRD_MODEL_PATH`` environment variable. The
device defaults to ``cpu`` and can be overridden with ``TRD_DEVICE``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from .inference import ToothResorptionInference

MODEL_PATH = Path(os.environ.get("TRD_MODEL_PATH", "deployment/torchscript/vit_base_16.pt"))
DEVICE = os.environ.get("TRD_DEVICE", "cpu")
UPLOAD_FOLDER = Path(os.environ.get("TRD_UPLOAD_FOLDER", "uploads"))
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

detector = ToothResorptionInference(model_path=MODEL_PATH, device=DEVICE)


@app.route("/health", methods=["GET"])
def health_check() -> Any:
    return jsonify({"status": "healthy", "model": "vit_base_16"})


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)
    try:
        result = detector.predict(filepath)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001 - surfaced over HTTP
        return jsonify({"error": str(exc)}), 500
    finally:
        filepath.unlink(missing_ok=True)


@app.route("/predict_batch", methods=["POST"])
def predict_batch() -> Any:
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    filepaths: list[Path] = []
    try:
        for file in files:
            if not file.filename:
                continue
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            file.save(filepath)
            filepaths.append(filepath)

        results = detector.predict_batch([str(p) for p in filepaths])
        return jsonify({"results": results})
    except Exception as exc:  # noqa: BLE001 - surfaced over HTTP
        return jsonify({"error": str(exc)}), 500
    finally:
        for filepath in filepaths:
            filepath.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("TRD_PORT", "5000")), debug=False)
