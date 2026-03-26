"""Analysis package – metric aggregation, statistics and visualization."""

from analysis.aggregator import MetricAggregator
from analysis.stats_merger import StatisticsMerger
from analysis.comprehensive_statistics import (
    ComprehensiveStatistics,
    generate_comprehensive_statistics,
    METRIC_HIGHER_IS_BETTER,
)
from analysis.comprehensive_visualization import (
    ComprehensiveVisualization,
    generate_comprehensive_visualizations,
)

__all__ = [
    "MetricAggregator",
    "StatisticsMerger",
    "ComprehensiveStatistics",
    "generate_comprehensive_statistics",
    "ComprehensiveVisualization",
    "generate_comprehensive_visualizations",
    "METRIC_HIGHER_IS_BETTER",
]
