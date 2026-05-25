# Migration notes

This document records the consolidation of the legacy `yontem_deneme/`
folder into `tooth-resorption-detection/`.

## High-level summary

- The flat `src/` module was promoted to a proper importable package at
  `src/tooth_resorption/` with topical subpackages.
- Every working artifact from `yontem_deneme/` was migrated, renamed and
  re-integrated; obvious junk (caches, logs, default downloadable weights,
  duplicate run folders) was deleted.
- Every Turkish identifier in production code was translated to English
  `snake_case`. The three domain class labels `temasli` / `bagimsiz` /
  `rezorpsiyon` are preserved as-is because they appear inside the
  dataset annotations and are part of the public contract.
- The non-ASCII folder `20 lik diş rezorpsiyon/` was renamed to
  `data/raw/labelme_dataset/` and is git-ignored together with the
  derived YOLO dataset.

## Package restructure

### Renames

| Old path                                                  | New path                                                      |
| :-------------------------------------------------------- | :------------------------------------------------------------ |
| `src/__init__.py`                                         | `src/tooth_resorption/__init__.py`                            |
| `src/_logging.py`                                         | `src/tooth_resorption/logging_utils.py`                       |
| `src/config.py`                                           | `src/tooth_resorption/config.py`                              |
| `src/preprocessing.py`                                    | `src/tooth_resorption/data/preprocessing.py`                  |
| `src/synthetic_data.py`                                   | `src/tooth_resorption/data/synthetic_data.py`                 |
| `src/data_loader.py`                                      | `src/tooth_resorption/data/data_loader.py`                    |
| `src/model.py`                                            | `src/tooth_resorption/models/model.py`                        |
| `src/train.py`                                            | `src/tooth_resorption/training/train.py`                      |
| `src/evaluate.py`                                         | `src/tooth_resorption/evaluation/evaluate.py`                 |
| `src/plot_comparison.py`                                  | `src/tooth_resorption/visualization/plot_comparison.py`       |

### Imports

All `from src.X import …` references were rewritten to
`from tooth_resorption.X import …` (or the corresponding sub-package
path). Tests, the conftest and the plotting helpers were updated in
lockstep with the package move.

## yontem_deneme migration

### Python modules

| Old path                                                       | New path                                                                | Notes                                                       |
| :------------------------------------------------------------- | :---------------------------------------------------------------------- | :---------------------------------------------------------- |
| `yontem_deneme/model_architectures.py`                         | `src/tooth_resorption/models/architectures.py`                          | English docstrings; type hints; module/class names unchanged. |
| `yontem_deneme/tooth_resorption_detector.py`                   | `src/tooth_resorption/detection/tooth_resorption_detector.py`           | Now reads from `data/raw/labelme_dataset/`; uses `SEED`.    |
| `yontem_deneme/yolo_tooth_detector.py`                         | `src/tooth_resorption/detection/yolo_detector.py`                       | Renamed class `YOLOToothDetector` → `YoloToothDetector`.     |
| `yontem_deneme/train_attention_transformers.py`                | `src/tooth_resorption/training/train_attention.py` + `scripts/train_attention.py` | Split between library code and CLI wrapper.        |
| `yontem_deneme/evaluate_all_models.py`                         | `src/tooth_resorption/evaluation/model_comparison.py` + `scripts/evaluate_models.py` | Same split.                                          |
| `yontem_deneme/compare_models.py`                              | `scripts/compare_models.py`                                             | Simplified; reads from `results/metrics/`.                  |
| `yontem_deneme/export_best_model.py`                           | `scripts/export_model.py`                                               | Trimmed to ONNX + TorchScript + int8 dynamic quant.          |
| `yontem_deneme/generate_report_figures.py`                     | `scripts/generate_report_figures.py`                                    | English-only port; reads `results/metrics/`.                |
| `yontem_deneme/generate_saliency_map.py`                       | `scripts/generate_saliency_map.py` (CLI) + `src/tooth_resorption/visualization/saliency.py` (logic) | Same split.                                                 |
| `yontem_deneme/run_evaluation_pipeline.py`                     | `scripts/run_evaluation_pipeline.py`                                    | Non-interactive Python entry-point version.                  |

### Python scripts deleted (redundant / abandoned)

These were either dead code, duplicates of better modules, or
exploratory analyses with no remaining value beyond the figures they
produced (those figures live under `results/figures/`):

- `advanced_analysis.py`
- `all_metrics_comparison.py`
- `check_training_status.py` (orchestration helper, replaced by Python API)
- `compare_attention_variants.py`
- `complete_advanced_analysis.py`
- `comprehensive_model_comparison.py`
- `evaluate_results.py`
- `evaluate_single_model.py`
- `extract_sample_image.py`
- `final_results_analysis.py`
- `high_confidence_ensemble_test.py`
- `map50_detailed_analysis.py`
- `metrics_table_comparison.py`
- `model_comparison_map50.py`
- `rsna_attention_benchmark.py`
- `select_best_model.py`
- `simple_test.py`
- `yolo_manual_test.py`
- `yolo_rsna_comparison.py`

The historical reports under `results/reports/` document what these
scripts produced; if any specific analysis needs to be re-run, port the
relevant logic onto `tooth_resorption.evaluation.model_comparison` rather
than reviving the script verbatim.

### Models and artifacts

| Old path                                                        | New path                                                       |
| :-------------------------------------------------------------- | :------------------------------------------------------------- |
| `yontem_deneme/tooth_resorption_model.h5`                       | `models/cnn_baseline.h5` (gitignored)                          |
| `yontem_deneme/yolo_final/tooth_detection/weights/best.pt`      | `models/yolo11n_best.pt` (gitignored)                          |
| `yontem_deneme/yolo_final/tooth_detection/weights/last.pt`      | `models/yolo11n_last.pt` (gitignored)                          |
| `yontem_deneme/yolo_runs/tooth_resorption3/weights/best.pt`     | `models/yolo_run3_best.pt` (gitignored)                        |
| `yontem_deneme/yolo_runs/tooth_resorption3/weights/last.pt`     | `models/yolo_run3_last.pt` (gitignored)                        |
| `yontem_deneme/deployment/`                                     | `tooth-resorption-detection/deployment/`                       |
| `yontem_deneme/20 lik diş rezorpsiyon/`                         | `data/raw/labelme_dataset/` (gitignored)                       |
| `yontem_deneme/yolo_dataset/`                                   | `data/processed/yolo_dataset/` (gitignored)                    |

### Results / figures / reports

| Old path                                                              | New path                                                                  |
| :-------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| `yontem_deneme/1DOCS/COMPLETE_PROJECT_DOCUMENTATION.md`               | `results/reports/project_overview.md`                                     |
| `yontem_deneme/1DOCS/TECHNICAL_REPORT.md`                             | `results/reports/technical_report.md`                                     |
| `yontem_deneme/1DOCS/*.png`                                           | `results/figures/*.png`                                                   |
| `yontem_deneme/1DOCS/high_confidence_*.{csv,json}`                    | `results/metrics/high_confidence_*.{csv,json}`                            |
| `yontem_deneme/1DOCS/model_summary_table.csv`                         | `results/metrics/model_summary_table.csv`                                 |
| `yontem_deneme/result_images/*.png`                                   | `results/figures/*.png`                                                   |
| `yontem_deneme/model_performance_results.png`                         | `results/figures/model_performance_results.png`                           |
| `yontem_deneme/rsna_yolo_comparison.png`                              | `results/figures/rsna_yolo_comparison.png`                                |
| `yontem_deneme/results/*.{csv,json}` (top-level)                      | `results/metrics/*.{csv,json}`                                            |
| `yontem_deneme/results/training_logs/*`                               | `results/metrics/training_logs/*`                                         |
| `yontem_deneme/results/visualizations/*.png`                          | `results/figures/*.png`                                                   |
| `yontem_deneme/results/advanced_analysis/*` (png/json/csv)            | `results/figures/*.png` + `results/metrics/*.{json,csv}`                  |
| `yontem_deneme/results/complete_analysis/*` (png/json/csv)            | `results/figures/*.png` + `results/metrics/*.{json,csv}`                  |
| `yontem_deneme/yolo_final/tooth_detection/*.{png,jpg,csv,yaml}`       | `results/figures/yolo_final_*.{png,jpg}` + `results/metrics/yolo_final_*.{csv,yaml}` |
| `yontem_deneme/yolo_runs/tooth_resorption3/results.csv`               | `results/metrics/yolo_run3_results.csv`                                   |

### Documentation

The legacy multi-file documentation in `yontem_deneme/` was deduplicated
and folded into the new layout:

- `yontem_deneme/PROJECT_STRUCTURE.md`, `QUICK_START.md`,
  `README_ATTENTION_TRANSFORMERS.md`, `TRAINING_STATUS_README.md`,
  `yapilanlar.txt` — all deleted. The structural and quickstart
  information now lives in the top-level `README.md` and `docs/`.
- `yontem_deneme/COMPLETE_PROJECT_DOCUMENTATION.md` → renamed and moved
  to `results/reports/project_overview.md`.
- `yontem_deneme/TECHNICAL_REPORT.md` → renamed and moved to
  `results/reports/technical_report.md`.

### Junk deleted outright

- `yontem_deneme/__pycache__/`
- `yontem_deneme/models_backup_20251119_213928/`
- `yontem_deneme/runs/` (mlflow scratch)
- `yontem_deneme/yolo_runs/tooth_resorption/`,
  `yontem_deneme/yolo_runs/tooth_resorption2/` (older duplicate runs)
- `yontem_deneme/yolo_final/tooth_detection2/` (duplicate of `tooth_detection/`)
- `yontem_deneme/yolo_simple/` (early-iteration scratch dataset)
- `yontem_deneme/yolo11n.pt` (default downloadable weight; re-fetched on
  demand by `ultralytics`)
- `yontem_deneme/transformer_error.txt`, `transformer_output.txt`,
  `training_error.log`, `training_output.log` (training logs / empty
  scratch)

### Requirements

`yontem_deneme/requirements.txt` was merged into the top-level
`requirements.txt` and `pyproject.toml`. Optional dependency groups:

- `[detection]` — `ultralytics`, `opencv-python` (YOLO track).
- `[tensorflow]` — `tensorflow>=2.13,<2.16` (CNN baseline).
- `[deployment]` — `onnx`, `onnxruntime`, `flask`.
- `[tracking]` — `mlflow`.
- `[dev]` — `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`, `types-*`.

## Verification

1. `yontem_deneme/` no longer exists.
2. `python -c "import tooth_resorption; print(tooth_resorption.__version__)"`
   imports the package without errors.
3. `python -m pytest` exercises the synthetic-data smoke tests against the
   new package paths.
4. `ruff check src scripts tests` and `mypy src` are clean modulo expected
   third-party stub gaps.

## Follow-ups for the user

- The original clinical dataset must be re-placed under
  `data/raw/labelme_dataset/` on every fresh clone (it is git-ignored).
- Re-run `scripts/train_attention.py` and `scripts/train_yolo.py` to
  regenerate model artefacts under `models/` after a fresh clone.
- The default `yolo11n.pt` will be re-fetched automatically by
  `ultralytics` on first use.
- The published numbers under `results/metrics.json` are pinned to the
  MSc thesis run; re-running the full training pipeline may shift them
  slightly because of stochastic init / shuffle order even with the
  fixed seed.
