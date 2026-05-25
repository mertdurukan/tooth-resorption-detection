"""Evaluation entrypoints (single model + cross-model comparison)."""

from __future__ import annotations

from tooth_resorption.evaluation.evaluate import evaluate
from tooth_resorption.evaluation.model_comparison import (
    UnifiedModelEvaluator,
    compute_metrics,
)

__all__ = ["UnifiedModelEvaluator", "compute_metrics", "evaluate"]
