"""
StatisticsMerger – merges per-file ``*_stats.csv`` into a single
``statistics_merged.csv`` at the experiment root.

This corresponds to STEP 1b of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from analysis.aggregator import MetricAggregator


class StatisticsMerger:
    """
    Walk an experiment directory, aggregate every ``*_raw.csv``,
    and merge all stats into ``statistics_merged.csv``.
    """

    def __init__(self) -> None:
        self.aggregator = MetricAggregator()

    def run(self, exp_dir: Path) -> Dict[str, Any]:
        """
        Aggregate all raw CSVs under *exp_dir* and produce the merged file.

        Returns summary metadata.
        """
        exp_dir = Path(exp_dir)
        raw_files = sorted(exp_dir.glob("*/*/*_raw.csv"))
        if not raw_files:
            raise RuntimeError(
                f"No raw CSV files found under {exp_dir}. Run STEP 1 first."
            )

        merged_parts: List[pd.DataFrame] = []
        for raw_path in raw_files:
            stats_df = self.aggregator.aggregate_file(raw_path)
            stats_path = raw_path.with_name(
                raw_path.name.replace("_raw.csv", "_stats.csv")
            )
            stats_df.to_csv(stats_path, index=False)
            if not stats_df.empty:
                stats_df = stats_df.copy()
                stats_df["source_stats_file"] = str(stats_path)
                merged_parts.append(stats_df)

        merged_df = (
            pd.concat(merged_parts, ignore_index=True)
            if merged_parts
            else pd.DataFrame()
        )
        merged_path = exp_dir / "statistics_merged.csv"
        merged_df.to_csv(merged_path, index=False)

        summary = {
            "experiment": exp_dir.name,
            "n_raw_files": len(raw_files),
            "n_stats_files": len(raw_files),
            "merged_statistics_csv": str(merged_path),
        }
        pd.DataFrame([summary]).to_csv(
            exp_dir / "stage1b_summary.csv", index=False
        )
        return summary
