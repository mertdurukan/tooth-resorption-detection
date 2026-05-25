"""Plotting and saliency utilities."""

from __future__ import annotations

from tooth_resorption.visualization.plot_comparison import render
from tooth_resorption.visualization.saliency import (
    compute_saliency_map,
    create_overlay,
    load_sample_image,
)

__all__ = ["compute_saliency_map", "create_overlay", "load_sample_image", "render"]
