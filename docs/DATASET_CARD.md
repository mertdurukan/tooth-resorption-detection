# Dataset Card — Mersin University Wisdom-Tooth Panoramic Radiographs

> Hugging Face style dataset card for the private clinical dataset used
> in "Detection of External Root Resorption in Wisdom Teeth on Panoramic
> Dental Radiographs" (MSc thesis, Mersin University, 2026).

The dataset is **not redistributable**. This card documents its
structure, provenance, and intended use so that downstream consumers can
reproduce the methodology against compatible institutional data.

---

## Dataset summary

| Field             | Value                                                                  |
| :---------------- | :--------------------------------------------------------------------- |
| Name              | Mersin University Wisdom-Tooth Resorption Panoramic Radiograph Dataset |
| Curator           | Mert Durukan / Faculty of Dentistry, Mersin University                 |
| Modality          | Panoramic dental radiographs (2D, grayscale, single-projection)        |
| Task              | 3-class single-label image classification                              |
| Number of patients | ~55 (single centre)                                                   |
| Number of samples (after augmentation) | 1530 train / 306 test                              |
| Number of classes | 3                                                                      |
| Class labels      | `temasli`, `bagimsiz`, `rezorpsiyon`                                   |
| Image format      | JPEG/PNG, original resolution preserved on disk                        |
| Labelling tool    | LabelMe (JSON with base64 `imageData`, class string in `shapes[0].label`) |
| Licence           | Institutional — **not redistributed**                                  |
| Compliance        | KVKK (Turkish Law No. 6698) and institutional research protocols       |

---

## Supported tasks

- **Image classification** (primary): three-class single-label
  classification of impacted wisdom teeth.
- **Object detection** (secondary): the YOLO export under
  `data/processed/yolo_dataset/` ships bounding boxes for the same
  three classes (see [`dataset.yaml`](../data/processed/yolo_dataset/dataset.yaml)).

---

## Languages

Class labels are Turkish loan terms (`temasli`, `bagimsiz`,
`rezorpsiyon`) preserved through the pipeline for consistency with the
MSc thesis text. Documentation and code identifiers are English.

---

## Class definitions

| Label         | Meaning (clinical)                                              |
| :------------ | :-------------------------------------------------------------- |
| `temasli`     | Wisdom tooth is in contact with the adjacent second molar.      |
| `bagimsiz`    | Wisdom tooth is independent of the adjacent second molar.       |
| `rezorpsiyon` | External root resorption is visible on the adjacent tooth.      |

Class support on the held-out test split (post-augmentation): `temasli=130`,
`bagimsiz=84`, `rezorpsiyon=92`.

---

## Data collection

| Step              | Description                                                                 |
| :---------------- | :-------------------------------------------------------------------------- |
| Source            | Routine clinical panoramic radiographs from the Mersin University Faculty of Dentistry archives. |
| Acquisition       | Standard panoramic acquisition protocol; single scanner per acquisition session. |
| Inclusion         | Adults with impacted lower wisdom teeth visible on the panoramic field of view. |
| Labelling         | Manual segmentation + class label via LabelMe by the thesis author, reviewed by a clinician. |
| Pre-processing    | Crop region of interest around the wisdom tooth, resize to 224x224 for the classifier; YOLO branch keeps the original aspect ratio with letterbox. |
| Train/test split  | Stratified per class with fixed seed (`SEED=42`).                            |

---

## Personal and sensitive information

- The original filenames encoded patient names. **These have been
  stripped** by [`scripts/anonymize_dataset.py`](../scripts/anonymize_dataset.py),
  which renames each image and its matching YOLO label to
  `patient_<4-digit-id>.<ext>` and writes the reverse mapping to
  `anonymization_map.json`.
- `anonymization_map.json` is `.gitignore`d and must never leave the
  workstation. It is the only file that ties pseudonymised IDs back to
  patient identity.
- No DICOM metadata (patient name, birth date, accession number) is
  shipped through the processed split.
- Radiographs themselves are anatomical images and contain no overt
  identifying overlay text once cropped.

---

## Considerations for use

### Social impact

The downstream model could, in principle, support dentists in flagging
suspected external root resorption earlier. It is **not** a substitute
for clinical judgement. Releasing such tooling without local validation
can shift the standard of care and harm patients whose presentation
differs from the training distribution.

### Discussion of bias

- **Single-centre, single-population dataset.** Demographics and
  scanner characteristics reflect one Turkish dental faculty's
  catchment. Subgroup performance (age, sex, ethnicity) is not
  characterised.
- **Class imbalance.** Support is skewed toward `temasli`. The training
  pipeline does not apply class re-weighting by default.
- **Annotation bias.** Single labeller workflow; inter-rater agreement
  with independent dentists is not measured.

### Other known limitations

- Small absolute patient count (~55) — augmentation inflates the
  sample count but does not introduce new clinical variability.
- No follow-up labels (e.g. eventual extraction outcome) attached.
- No paired CBCT ground truth: resorption labels are derived from the
  2D projection only.

---

## Dataset structure

```
data/processed/yolo_dataset/
├── dataset.yaml                # Ultralytics descriptor (relative paths)
├── anonymization_map.json      # gitignored reverse lookup
├── train/
│   ├── images/patient_*.jpg    # 288 anonymised samples
│   └── labels/patient_*.txt    # YOLO format, one line per box
└── val/
    ├── images/patient_*.jpg    # 73 anonymised samples
    └── labels/patient_*.txt
```

`dataset.yaml` shipped at v0.2.0:

```yaml
path: .
train: train/images
val: val/images
nc: 3
names:
  - temasli
  - bagimsiz
  - rezorpsiyon
```

---

## Reproducing the methodology on compatible data

If you have an analogous institutional panoramic dataset:

1. Convert your annotations to either LabelMe JSON (for the classifier
   branch) or YOLO `txt` (for the detection branch).
2. Place images + labels under the layout above.
3. Run `python scripts/anonymize_dataset.py` to strip patient-derived
   filenames.
4. Train: `python -m tooth_resorption.training.train --data real --data-path data/raw/labelme_dataset --model vit_base --pretrained --epochs 50 --early-stopping 15`.
5. Evaluate: `python -m tooth_resorption.evaluation.evaluate --data real --data-path ...`.

The full synthetic-data smoke run is reproducible in this repo without
any clinical data: `python -m tooth_resorption.training.train --data synthetic`.

---

## Licensing

The dataset is **not licensed for redistribution**. Code and
documentation in this repository are MIT-licensed (see [`LICENSE`](../LICENSE)).
Use of the methodology against any clinical dataset is the user's
responsibility under the relevant regulatory regime (KVKK, GDPR,
CE-MDR, HIPAA, etc.).

---

## Citation

```bibtex
@misc{mersin2026toothresorption_dataset,
  author       = {Durukan, Mert and {Mersin University Faculty of Dentistry}},
  title        = {Mersin University Wisdom-Tooth Resorption Panoramic Radiograph Dataset (private)},
  year         = {2026},
  howpublished = {Internal MSc thesis dataset, Mersin University},
  note         = {Not redistributed due to KVKK / institutional restrictions.}
}
```
