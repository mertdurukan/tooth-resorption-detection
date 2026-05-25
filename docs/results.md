# Results

## Headline numbers

Reported on the held-out test split of 306 augmented samples (3 classes,
single-label), from the original MSc experiments on the private clinical
dataset (`results/metrics.json`).

| Metric                  | BaselineCNN | ViT-Base/16 | Δ        |
| :---------------------- | ----------: | ----------: | -------: |
| macro-F1                |      0.1988 |  **0.8981** | +0.6993  |
| macro-Precision         |      0.1416 |      0.8949 | +0.7533  |
| macro-Recall            |      0.3333 |      0.9026 | +0.5693  |
| accuracy                |      0.4248 |      0.8987 | +0.4739  |
| macro-AUC (one-vs-rest) |      0.5295 |      0.9862 | +0.4567  |
| trainable parameters    |        653K |       85.8M | —        |
| inference (ms/image)    |       84.96 |   **7.44**  | −11.4×   |

Per-class ViT-Base/16:

| Class         | Precision | Recall |     F1 | Support |
| :------------ | --------: | -----: | -----: | ------: |
| `temasli`     |    0.9262 | 0.8692 | 0.8968 |     130 |
| `bagimsiz`    |    0.8427 | 0.8929 | 0.8671 |      84 |
| `rezorpsiyon` |    0.9158 | 0.9457 | 0.9305 |      92 |

The full attention-zoo cross-model comparison from the MSc work is
preserved under `results/metrics/comparison_table.csv` and
`results/metrics/evaluation_results.json` (10+ architectures, including
Swin-Tiny / Small / Base, ResNet50-CBAM / SE, EfficientNet-Attention,
DenseNet-Attention, and the ViT/Swin + CBAM/SE/MHA variants).

## Where to look

| Path                                                | What lives there                                              |
| :-------------------------------------------------- | :------------------------------------------------------------ |
| `results/metrics.json`                              | Versioned headline numbers (baseline vs improved).           |
| `results/plots/comparison.png`                      | Versioned headline bar chart, rendered by `plot_comparison`. |
| `results/figures/*.png`                             | All comparison/chart PNGs from the MSc work.                  |
| `results/metrics/comparison_table.csv`              | Sortable cross-model summary.                                 |
| `results/metrics/evaluation_results.json`           | Full per-model metrics dump (used by `compare_models.py`).    |
| `results/metrics/training_logs/`                    | Per-model best hyper-params + composite scores.               |
| `results/reports/project_overview.md`               | Project overview report (formerly `COMPLETE_PROJECT_DOCUMENTATION.md`). |
| `results/reports/technical_report.md`               | Technical report (formerly `TECHNICAL_REPORT.md`).            |

## Regenerating the headline plot

```bash
python -m tooth_resorption.visualization.plot_comparison
# or:
python scripts/compare_models.py
```

Both commands write `results/plots/comparison.png` from
`results/metrics.json`. The synthetic smoke run never overwrites the
published metrics — it writes to `results/metrics_synthetic.json` and
`results/training_history_*.json` instead.

## Reproducing the full evaluation table

```bash
python scripts/evaluate_models.py
# Writes results/metrics/evaluation_results.json + comparison_table.csv

python scripts/generate_report_figures.py
# Re-renders results/figures/model_comparison_chart.png + composite_score_comparison.png
```

You need at least one trained checkpoint under `models/` for this to
produce a non-empty table. See [`docs/training.md`](training.md) for the
training recipes.
