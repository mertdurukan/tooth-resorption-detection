# Training

## Headline trainer (`tooth_resorption.training.train`)

The main trainer covers the `BaselineCNN` vs `ViT-Base/16` comparison
reported in the README. It accepts a `TrainConfig` dataclass (from
`tooth_resorption.config`) and runs an `AdamW` + `ReduceLROnPlateau` loop
with optional AMP, gradient clipping and early stopping.

```bash
# Synthetic smoke run (~60s on CPU)
python -m tooth_resorption.training.train --data synthetic

# Real-data full run (requires data/raw/labelme_dataset/)
python -m tooth_resorption.training.train \
    --data real --data-path data/raw/labelme_dataset \
    --model vit_base --pretrained \
    --epochs 50 --batch-size 8 \
    --amp --clip 1.0 --early-stopping 15 --deterministic
```

| Flag                | Default      | Description                                                  |
| :------------------ | :----------- | :----------------------------------------------------------- |
| `--data`            | `synthetic`  | `synthetic` or `real`.                                       |
| `--data-path`       | `None`       | Required when `--data real`.                                 |
| `--model`           | `baseline_cnn` | `baseline_cnn`, `vit_tiny`, or `vit_base`.                 |
| `--pretrained`      | `False`      | ViT-only; loads ImageNet-21k weights via `timm`.             |
| `--epochs`          | `2`          | Number of epochs.                                            |
| `--batch-size`      | `8`          | Mini-batch size.                                             |
| `--lr`              | `5e-4`       | AdamW learning rate.                                         |
| `--amp`             | `False`      | Enable CUDA mixed precision (`torch.cuda.amp`).              |
| `--clip`            | `1.0`        | Gradient L2 clip. `0` disables.                              |
| `--early-stopping`  | `0`          | Stop after N epochs without val macro-F1 improvement.        |
| `--deterministic`   | `False`      | `torch.use_deterministic_algorithms(True, warn_only=True)`.  |
| `--resume`          | `None`       | Resume from a previously saved checkpoint.                   |

### Outputs

- `models/{model}_best.pt`, `models/{model}_last.pt` — atomic checkpoints
  written via `*.tmp` + `rename` so a crash never corrupts existing data.
- `results/metrics_synthetic.json` — final-epoch metrics for the smoke run.
- `results/training_history_{model}.json` — per-epoch loss / F1 history.

## Attention zoo trainer (`scripts/train_attention.py`)

Grid-search trainer for the full attention/transformer zoo:

```bash
python scripts/train_attention.py --data-path data/raw/labelme_dataset
```

Each architecture is trained with three hyper-parameter configurations
(varying learning rate / batch size / weight decay / optimizer choice) and
the configuration with the highest validation composite score is saved as
`models/<arch>_best.pth`. The composite score is the unweighted mean of
accuracy, macro precision, macro recall, macro F1 and macro one-vs-rest
AUC — matching the original MSc evaluation protocol.

Per-model JSON logs are written to
`results/metrics/training_logs/{arch}_training_results.json` and a sorted
summary table to `results/metrics/training_logs/training_summary.csv`.

### Two-stage fine-tuning

For each model the backbone is frozen for the first 10 epochs (head-only
warm-up), then unfrozen for the rest of the run. The trainer also enables
early stopping with patience 15.

## YOLOv11 trainer (`scripts/train_yolo.py`)

Builds a YOLO-format dataset (ASCII-only filenames) under
`data/processed/yolo_dataset/` and trains the `yolo11n` backbone via
`ultralytics`:

```bash
python scripts/train_yolo.py --epochs 50 --batch 4
```

Outputs land under `runs/tooth_resorption/`. Move the best weights into
`models/` and re-run `scripts/evaluate_models.py` to fold the YOLO
detector into the cross-model comparison table.

## CNN baseline (TensorFlow)

The legacy Keras CNN is preserved in
`tooth_resorption.detection.tooth_resorption_detector.ToothResorptionDetector`.
Train it programmatically:

```python
from tooth_resorption.detection import ToothResorptionDetector

detector = ToothResorptionDetector()
detector.load_data()
detector.grid_search_train(epochs=50)
metrics = detector.evaluate_model()
```

The grid search writes `models/cnn_baseline.h5` and
`results/metrics/training_logs/cnn_training_results.json`.

## Reproducibility

- A single `SEED = 42` lives in `tooth_resorption.config` and is forwarded
  to `random`, NumPy, PyTorch (CPU + CUDA), and every DataLoader worker.
- Pass `--deterministic` to the main trainer to enable
  `torch.use_deterministic_algorithms(True, warn_only=True)` plus the
  cuBLAS workspace configuration required for fully reproducible CUDA
  matmul (at a measurable throughput cost).
- The synthetic data generator is deterministic per seed.
