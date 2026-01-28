"""Visualization modules for benchmark results.

Extended for AIO25 NoisySAM project with:
  - Noise gallery visualizations
  - Global sensitivity heatmaps
  - PSNR/uncertainty correlation plots
  - Robust path resolution for predictions
  - Multi-level failure case visualization
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
from viz.path_resolver import (
    resolve_pred_path,
    get_pred_root,
    validate_paths_in_df,
    format_path_validation_report,
    PathResolutionResult,
)

# Import enhanced failure cases module
try:
    from viz.failure_cases_v2 import (
        export_failure_cases_multilevel,
        export_random_cases_multilevel,
    )
except ImportError:
    export_failure_cases_multilevel = None
    export_random_cases_multilevel = None

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
    "export_failure_cases_multilevel",
    "export_random_cases_multilevel",
    "create_failure_summary",
    "analyze_failure_patterns",
    "resolve_pred_path",
    "get_pred_root",
    "validate_paths_in_df",
    "format_path_validation_report",
    "PathResolutionResult",
]
