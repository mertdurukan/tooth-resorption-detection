# Model Card — Wisdom-Tooth Root Resorption Classifier (ViT-Base/16)

> Hugging Face style model card for the headline ViT-Base/16 checkpoint
> reported in the MSc thesis "Detection of External Root Resorption in
> Wisdom Teeth on Panoramic Dental Radiographs" (Mersin University,
> Graduate School of Natural and Applied Sciences, 2026).

---

## Model details

| Field             | Value                                                                  |
| :---------------- | :--------------------------------------------------------------------- |
| Model name        | `vit_base_16` (ViT-Base/16, 224x224 input)                             |
| Architecture      | Vision Transformer (12 layers, 12 heads, 86M parameters)               |
| Backbone source   | `timm.create_model("vit_base_patch16_224", pretrained=True)`           |
| Pretraining       | ImageNet-21k -> ImageNet-1k (supervised)                               |
| Fine-tuning       | Mersin University clinical panoramic radiographs (3 classes)           |
| Framework         | PyTorch >= 2.0, timm >= 0.9                                            |
| License           | MIT (code). Weights derived from clinical data are NOT redistributed.  |
| Version           | 0.2.0                                                                  |
| Developed by      | Mert Durukan, MSc Artificial Intelligence, Mersin University           |
| Contact           | github.com/mertdurukan                                                 |

---

## Intended use

### Primary intended uses

- **Academic research** on transfer learning for small clinical
  imaging datasets.
- **Reference baseline** for benchmarking attention-based models against
  CNN baselines on panoramic dental radiographs.
- **Pedagogical example** of a reproducible MSc-thesis pipeline
  (lint, types, tests, CI, deployment).

### Primary intended users

- Dental-imaging researchers and graduate students.
- Open-source contributors extending the attention-zoo with new
  architectures or new datasets.

### Out-of-scope use cases

The model is **not** a medical device and is not approved for clinical
decision support. Concretely it must **not** be used for:

- Diagnosing, triaging, or screening real patients without
  prospective regulatory-grade validation.
- Operating on radiographs from a different acquisition protocol,
  scanner, or population than the Mersin University cohort. External
  validity is untested.
- Standalone reporting without a licensed clinician in the loop.
- Forensic identification or any task involving recovery of patient
  identity. The fine-tuning data is anonymised at source.

---

## How to use

```python
from deployment.api.inference import ToothResorptionInference

detector = ToothResorptionInference(
    model_path="deployment/torchscript/vit_base_16.pt",
    device="cpu",
)
result = detector.predict("xray.jpg")
print(result["predicted_class"], result["confidence"])
```

Serve as a Flask micro-service (see `docs/deployment.md`):

```bash
python -m deployment.api.flask_api
```

---

## Training data

| Field                       | Value                                                                  |
| :-------------------------- | :--------------------------------------------------------------------- |
| Source                      | Mersin University, Faculty of Dentistry archives                       |
| Modality                    | Panoramic dental radiographs (2D, grayscale)                           |
| Patients                    | ~55 unique cases (single centre)                                       |
| Samples (post-augmentation) | 1530 train / 306 test                                                  |
| Classes                     | `temasli`, `bagimsiz`, `rezorpsiyon` (3-class single-label)            |
| Class support (test)        | temasli=130, bagimsiz=84, rezorpsiyon=92                               |
| Labelling                   | Manual, LabelMe JSON, dentist-reviewed                                 |
| Splitting                   | Stratified train/val, fixed seed (`SEED=42`)                           |
| Pre-processing              | Resize 224x224, ImageNet mean/std normalisation                        |
| Augmentations               | h-flip, +-15 deg rotation, +-5% affine, brightness/contrast jitter     |
| Redistribution              | **Not allowed.** KVKK / institutional ownership restricts release.     |

See [`docs/DATASET_CARD.md`](DATASET_CARD.md) for the full dataset card.

---

## Evaluation data

Identical preprocessing pipeline as training; held-out stratified split
of 306 augmented samples, never seen during training or model selection.

---

## Metrics

Reported on the held-out test split.

| Metric                  | BaselineCNN | ViT-Base/16 (this model) |
| :---------------------- | ----------: | -----------------------: |
| accuracy                |      0.4248 |               **0.8987** |
| macro F1                |      0.1988 |               **0.8981** |
| macro precision         |      0.1416 |                   0.8949 |
| macro recall            |      0.3333 |                   0.9026 |
| macro AUC (OvR)         |      0.5295 |                   0.9862 |
| inference (ms/img, CPU) |       84.96 |                 **7.44** |

Per-class performance (ViT-Base/16):

| Class         | Precision | Recall |     F1 | Support |
| :------------ | --------: | -----: | -----: | ------: |
| `temasli`     |    0.9262 | 0.8692 | 0.8968 |     130 |
| `bagimsiz`    |    0.8427 | 0.8929 | 0.8671 |      84 |
| `rezorpsiyon` |    0.9158 | 0.9457 | **0.9305** |  92 |

Source: [`results/metrics.json`](../results/metrics.json) and the
classification report under `results/classification_report_*.txt`.

---

## Decision threshold and operating point

The model returns a softmax distribution; the published metrics use the
argmax decision rule (top-1). No threshold tuning per class was
performed. Operating points for clinically motivated trade-offs
(sensitivity vs specificity) are **not** characterised.

---

## Limitations

- **Single-centre, small dataset.** ~55 patients, one university
  hospital, one scanner protocol. Macro-F1 on out-of-distribution
  panoramic radiographs (different scanner, different population) is
  unknown.
- **Class imbalance is not actively corrected.** Unweighted
  `CrossEntropyLoss`; weighted loss / weighted sampling could shift the
  precision-recall trade-off, especially for the under-represented
  `bagimsiz` class.
- **No external test set.** A cross-dataset transfer experiment on the
  RSNA 2023 dental imaging benchmark was not run.
- **No uncertainty quantification.** The model outputs a point softmax;
  no calibration, conformal prediction, or epistemic uncertainty
  estimate ships with the checkpoint.
- **TorchScript trace is shape-fixed at 224x224.** Inference on a
  different input shape requires re-tracing or re-exporting to ONNX
  with dynamic axes.

---

## Bias

- **Acquisition bias.** All training images come from a single dental
  faculty in Turkey; demographic distribution (age, sex, ethnicity)
  reflects that catchment area and is not representative globally.
- **Label bias.** Labels are produced by clinicians from the same
  institution. Inter-rater variability with external dentists has not
  been measured.
- **Subgroup performance unmeasured.** The thesis does not stratify
  metrics by patient age, sex, or impaction angulation. Disparate
  per-subgroup performance cannot be ruled out.

---

## Ethical considerations and KVKK compliance

- The fine-tuning data was collected for routine clinical care and
  re-used for research under institutional protocols. **Identifying
  filenames (patient names, IDs) are removed at ingest** via
  [`scripts/anonymize_dataset.py`](../scripts/anonymize_dataset.py); the
  resulting `anonymization_map.json` is git-ignored and never leaves the
  workstation.
- Raw radiographs (`data/raw/`, `data/processed/`) and trained
  checkpoints (`models/`) are **never committed** to version control.
- Outputs of the model are **not** patient-identifying.
- Any clinical use must comply with the Turkish Personal Data Protection
  Law (KVKK, Law No. 6698), EU GDPR if applicable, and the relevant
  medical-device regulation (CE-MDR or equivalent).
- Recommended use is research-only with anonymised data and a defined
  intended-use statement.

---

## Environmental impact

- Training the headline ViT-Base/16 takes ~50 epochs on a single
  consumer GPU (~30 minutes on an RTX 3060). Estimated emissions per
  full training run: < 0.05 kg CO2-eq (using ML CO2 Impact assumptions
  for a 170 W GPU on the European grid).
- Inference is cheap: ~7 ms/image on CPU; no special hardware required
  to serve.

---

## Caveats and recommendations

- Always pair the model output with a clinician's review.
- Re-validate on local data before any deployment.
- Track the model and the data jointly through MLflow (see
  `tooth_resorption.training.train`) to keep the data-model lineage
  auditable.

---

## Citation

```bibtex
@software{durukan2026toothresorption,
  author  = {Durukan, Mert},
  title   = {Wisdom Tooth Root Resorption Detection on Panoramic Dental Radiographs},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/mertdurukan/tooth-resorption-detection}
}
```
