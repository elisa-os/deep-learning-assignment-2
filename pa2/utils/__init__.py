"""Utilitários do PA2 — seeds, dispositivo, visualização e exportação de métricas."""

from .seed import set_seed
from .device import get_device
from .visualize import (
    generate_tracking_colors,
    show_tracking_frame,
    plot_idf1_vs_frames,
    plot_comparison_baseline_vs_temporal,
    plot_gradient_vanishing_curve,
    plot_occlusion_survival,
    save_figure,
)
from .export import PerSequenceMetricsWriter

__all__ = [
    "set_seed",
    "get_device",
    "generate_tracking_colors",
    "show_tracking_frame",
    "plot_idf1_vs_frames",
    "plot_comparison_baseline_vs_temporal",
    "plot_gradient_vanishing_curve",
    "plot_occlusion_survival",
    "save_figure",
    "PerSequenceMetricsWriter",
]
