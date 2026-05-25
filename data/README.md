# Data

This directory holds local copies of the input datasets. Nothing in
`data/raw/`, `data/processed/`, `data/interim/`, `data/synthetic/`, or
`data/clinical/` is committed to git — see [`/.gitignore`](../.gitignore).

## Layout

```
data/
├── raw/                  # Original, immutable input data (never edit in place)
│   └── labelme_dataset/  # 666 LabelMe JPEG+JSON pairs from Mersin University
├── processed/            # Reproducible derived datasets
│   └── yolo_dataset/     # YOLO-format train/val splits exported by yolo_detector
└── README.md             # This file
```

## Raw dataset

`data/raw/labelme_dataset/` contains panoramic dental radiographs from
Mersin University (≈ 222 patients × multiple acquisitions = 666 image/JSON
pairs). Each `<patient>.json` is a [LabelMe](https://github.com/wkentaro/labelme)
annotation that:

- Embeds the original JPEG inline as base64 under `imageData`.
- Stores one or more bounding-box annotations under `shapes[]`, each with
  a `label` string in `{"temasli", "bagimsiz", "rezorpsiyon"}` (Turkish
  domain terms preserved as class identifiers).

This dataset cannot be redistributed due to patient privacy (KVKK / GDPR /
institutional ownership) and is excluded from version control. Obtain it
through the original principal investigator at Mersin University. The
historical folder name `20 lik diş rezorpsiyon` has been replaced by the
ASCII path `data/raw/labelme_dataset/`; downstream code reads the new path.

## Processed dataset

`data/processed/yolo_dataset/` is a deterministic derivative of the raw
LabelMe data exported by
[`tooth_resorption.detection.yolo_detector.YoloToothDetector.prepare_dataset`](../src/tooth_resorption/detection/yolo_detector.py).
Re-running that method recreates the directory from scratch; do not commit
it.

## Synthetic dataset

For reviewers without access to the real data,
[`tooth_resorption.data.synthetic_data.generate`](../src/tooth_resorption/data/synthetic_data.py)
produces 224×224 procedurally synthesised RGB images that exercise every
code path end-to-end. The synthetic generator is fully deterministic given
a seed and writes to `data/synthetic/` if you call `save_synthetic_dataset`.
