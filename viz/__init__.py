"""Step-2 visualization – OOP classes and backwards-compat free functions."""

# ── default level descriptors ────────────────────────────────────────────
DEFAULT_LEVEL_NAMES = {
    "L0": "clean",
    "L1": "very mild",
    "L2": "mild",
    "L3": "moderate",
    "L4": "strong",
    "L5": "severe",
    "L6": "extreme",
    "L7": "destructive",
    "L8": "near failure",
    "L9": "catastrophic",
}


def format_level_label(level: str, level_names: dict | None = None) -> str:
    """Return a two-line tick label, e.g. ``'L3\\n(moderate)'``."""
    names = level_names or DEFAULT_LEVEL_NAMES
    desc = names.get(str(level), "")
    return f"{level}\n({desc})" if desc else str(level)


# OOP classes
from viz.plot_metrics import MetricPlotter
from viz.model_comparison import ModelComparisonPlotter
from viz.noise_gallery import NoiseGalleryGenerator
from viz.prediction_overlay import PredictionVisualizer
from viz.prompt_visualization import PromptComparisonPlotter
from viz.statistical_tables import StatisticalTableGenerator

# backwards-compat free functions
from viz.model_comparison import generate_model_comparison
from viz.noise_gallery import generate_noise_gallery
from viz.plot_metrics import generate_metric_curves, generate_robustness_plot
from viz.prediction_overlay import generate_prediction_overlays
from viz.prompt_visualization import generate_prompt_comparison, generate_prompt_visualization
from viz.statistical_tables import generate_statistical_tables

__all__ = [
    # constants & helpers
    "DEFAULT_LEVEL_NAMES",
    "format_level_label",
    # classes
    "MetricPlotter",
    "ModelComparisonPlotter",
    "NoiseGalleryGenerator",
    "PredictionVisualizer",
    "PromptComparisonPlotter",
    "StatisticalTableGenerator",
    # free functions
    "generate_metric_curves",
    "generate_robustness_plot",
    "generate_noise_gallery",
    "generate_prompt_visualization",
    "generate_prompt_comparison",
    "generate_prediction_overlays",
    "generate_model_comparison",
    "generate_statistical_tables",
]
