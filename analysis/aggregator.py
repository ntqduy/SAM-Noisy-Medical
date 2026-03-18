"""
MetricAggregator – converts per-image raw CSV files into statistics
(mean, std, CV%) grouped by (dataset, model, prompt_mode, noise_type, noise_level).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


METRICS = ["Dice", "IoU", "Recall", "Precision", "F1", "HD"]
GROUP_KEYS = ["dataset", "model", "prompt_mode", "noise_type", "noise_level"]


def _parse_level_idx(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


class MetricAggregator:
    """
    Reads a per-image ``*_raw.csv`` and produces a ``*_stats.csv`` with
    mean, std, CV% for each metric over each group.
    """

    def aggregate_file(self, raw_csv: Path) -> pd.DataFrame:
        """
        Aggregate one raw CSV.

        Returns a DataFrame with columns:
        ``GROUP_KEYS`` + per-metric (mean, ``{M}_std``, ``{M}_cv_pct``)
        + ``n_images``, ``n_rows``.
        """
        df = pd.read_csv(raw_csv)
        if df.empty:
            return pd.DataFrame(columns=GROUP_KEYS)

        for metric in METRICS:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")
                df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)

        grouped = df.groupby(GROUP_KEYS, dropna=False)
        rows: List[Dict[str, Any]] = []
        for keys, g in grouped:
            row = dict(zip(GROUP_KEYS, keys))
            if "is_gt_empty" in g.columns:
                gt_empty = pd.to_numeric(g["is_gt_empty"], errors="coerce").fillna(0)
                row["gt_empty_rate"] = float(gt_empty.mean()) if len(gt_empty) else np.nan
                row["n_gt_non_empty"] = int((gt_empty == 0).sum())
            if "is_pred_empty" in g.columns:
                pred_empty = pd.to_numeric(g["is_pred_empty"], errors="coerce").fillna(0)
                row["pred_empty_rate"] = float(pred_empty.mean()) if len(pred_empty) else np.nan
            for metric in METRICS:
                if metric not in g.columns:
                    continue
                vals = g[metric].dropna()
                mean = float(vals.mean()) if len(vals) else np.nan
                std = float(vals.std(ddof=0)) if len(vals) else np.nan
                cv_pct = (
                    float(std / mean * 100.0)
                    if np.isfinite(mean) and abs(mean) > 1e-12 and np.isfinite(std)
                    else np.nan
                )
                row[metric] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_cv_pct"] = cv_pct
                row[f"{metric}_n_valid"] = int(len(vals))
            row["n_images"] = (
                int(g["image_id"].nunique()) if "image_id" in g.columns else int(len(g))
            )
            row["n_rows"] = int(len(g))
            rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            out["__level_idx"] = out["noise_level"].map(_parse_level_idx)
            out = out.sort_values(
                ["dataset", "model", "prompt_mode", "noise_type", "__level_idx", "noise_level"]
            )
            out = out.drop(columns=["__level_idx"])
        return out

    def aggregate_and_save(self, raw_csv: Path) -> Path:
        """Aggregate a raw CSV and write the result alongside it as ``*_stats.csv``."""
        stats_df = self.aggregate_file(raw_csv)
        stats_path = raw_csv.with_name(raw_csv.name.replace("_raw.csv", "_stats.csv"))
        stats_df.to_csv(stats_path, index=False)
        return stats_path
