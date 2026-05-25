# Deployment

## Overview

The deployment artefacts live under `deployment/`. The directory is
populated by `scripts/export_model.py`, which takes the best PyTorch
checkpoint produced by the attention-zoo trainer and exports it to three
production formats plus a metadata sidecar.

```
deployment/
├── api/                         # Flask inference micro-service
│   ├── flask_api.py
│   ├── inference.py
│   ├── README.md
│   └── requirements.txt
├── onnx/                        # ONNX 14 graphs (gitignored)
├── torchscript/                 # `torch.jit.trace`d binaries (gitignored)
├── quantized/                   # int8 dynamic-quant state dicts (gitignored)
└── model_metadata.json          # Class names, normalisation, perf metrics
```

## Export pipeline

```bash
python scripts/export_model.py \
    --model-type vit_base_16 \
    --checkpoint models/vit_base_16_best.pth
```

Flags:

| Flag                 | Default                              | Description                                    |
| :------------------- | :----------------------------------- | :--------------------------------------------- |
| `--model-type`       | required                             | Architecture key from `create_model`.          |
| `--checkpoint`       | required                             | Path to the trained `*_best.pth` artifact.     |
| `--metrics-json`     | `results/metrics/evaluation_results.json` | Looked up by `model_type` if it exists.  |
| `--deployment-dir`   | `deployment/`                        | Destination root.                              |
| `--skip-quantization`| `False`                              | Skip the dynamic int8 export.                  |

`model_metadata.json` always carries: model name, version, export date,
class names, input shape, normalisation mean/std and (when available) the
reported performance metrics — everything the consumer needs to wire up
preprocessing correctly.

## Flask API

`deployment/api/flask_api.py` provides a minimal `/predict` endpoint that
accepts a JPEG/PNG upload and returns the top-1 class plus full per-class
softmax. Install the API requirements separately:

```bash
pip install -e ".[deployment]"
python -m deployment.api.flask_api
```

> Run the module from the repository root — the API uses relative imports
> inside `deployment/api/`, so launching it as a script will break.

The API expects the ONNX export at the path declared in
`model_metadata.json`. Override paths via environment variables (see the
top of `flask_api.py`).

## ONNX Runtime serving (recommended)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

session = ort.InferenceSession("deployment/onnx/vit_base_16.onnx")
tx = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = Image.open("sample.jpg").convert("RGB")
tensor = tx(img).unsqueeze(0).numpy().astype(np.float32)
logits = session.run(None, {"input": tensor})[0]
print(np.argmax(logits, axis=1))
```

## TorchScript

```python
import torch

model = torch.jit.load("deployment/torchscript/vit_base_16.pt")
model.eval()

with torch.no_grad():
    logits = model(torch.randn(1, 3, 224, 224))
```

## Quantization

The dynamic int8 quantization step targets `nn.Linear` layers — adequate
for transformer backbones where most of the runtime is in matmuls. For
convolutional models (BaselineCNN, ResNet-based variants) prefer static
post-training quantization with a calibration loader.

## Production checklist

- [ ] Run `scripts/evaluate_models.py` to refresh
  `results/metrics/evaluation_results.json` before exporting.
- [ ] Run `scripts/export_model.py` with the best `--model-type`.
- [ ] Verify metadata under `deployment/model_metadata.json` matches the
  exported architecture, class names, and reported metrics.
- [ ] Smoke-test the ONNX file with `onnxruntime` on a held-out image.
- [ ] Pin model versions and metric thresholds in your deployment manifest
  (e.g. Helm chart, GitHub Release notes).

## Clinical use

This pipeline is research code, not a medical device. Any clinical use
requires regulatory review, prospective validation, and a defined
intended-use statement.
