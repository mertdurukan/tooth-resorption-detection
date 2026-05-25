# Tooth Resorption Detection - Deployment

## Model Information
- **Model**: vit_base_16
- **Accuracy**: 0.8987
- **F1-Score**: 0.8981
- **AUC**: 0.9862
- **Inference Time**: 7.44 ms

## Classes
1. `temasli` (contact)
2. `bagimsiz` (independent)
3. `rezorpsiyon` (resorption)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
from deployment.api.inference import ToothResorptionInference

detector = ToothResorptionInference(
    model_path="deployment/torchscript/vit_base_16.pt",
    device="cpu",
)

result = detector.predict("xray.jpg")
print(result)
```

### Flask REST API

Always launch the API as a module from the repository root so the relative
imports inside `deployment/api/` resolve correctly:

```bash
python -m deployment.api.flask_api
```

Then use curl or any HTTP client:

```bash
curl -X POST -F "file=@xray.jpg" http://localhost:5000/predict
```

## Model Files

- `vit_base_16.onnx` - ONNX format (cross-platform)
- `vit_base_16.pt` - TorchScript format (PyTorch)
- `vit_base_16_quantized.pt` - Quantized INT8 (faster, smaller)

## Performance

- **Inference Time**: ~7 ms per image
- **Model Size**: 327.30 MB
- **Hardware**: CPU or GPU compatible

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single image prediction
- `POST /predict_batch` - Batch prediction

## Notes

- Input images should be dental X-rays
- Recommended image size: 224x224 pixels
- Supported formats: JPG, PNG
