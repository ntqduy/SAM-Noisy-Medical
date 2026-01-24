"""Visualization modules for benchmark results.

Extended for AIO25 NoisySAM project with:
  - Noise gallery visualizations
  - Global sensitivity heatmaps
  - PSNR/uncertainty correlation plots
"""
from viz.overlays import overlay
from viz.grids import save_preview_pdf, save_side_by_side_grid, save_noise_gallery
from viz.plots import (
    plot_metric_vs_level,
    plot_ofat_sensitivity,
    plot_grid_heatmap,
    plot_model_comparison,
    plot_noise_comparison,
    plot_stability_summary,
    plot_global_sensitivity,
    plot_summary_heatmap,
    plot_uncertainty_vs_performance,
    plot_psnr_vs_performance,
)
from viz.failure_cases import (
    identify_top_failures,
    export_failure_cases,
    create_failure_summary,
    analyze_failure_patterns,
)

__all__ = [
    "overlay",
    "save_preview_pdf",
    "save_side_by_side_grid",
    "save_noise_gallery",
    "plot_metric_vs_level",
    "plot_ofat_sensitivity",
    "plot_grid_heatmap",
    "plot_model_comparison",
    "plot_noise_comparison",
    "plot_stability_summary",
    "plot_global_sensitivity",
    "plot_summary_heatmap",
    "plot_uncertainty_vs_performance",
    "plot_psnr_vs_performance",
    "identify_top_failures",
    "export_failure_cases",
    "create_failure_summary",
    "analyze_failure_patterns",
]
