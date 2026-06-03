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


METRICS = [
    "Dice",
    "IoU",
    "Recall",
    "Precision",
    "F1",
    "HD",
    "HD_px",
    "HD95_px",
    "HD_mm",
    "HD95_mm",
    "inference_time_ms",
    "FPS",
]
BASE_GROUP_KEYS = ["dataset", "model", "prompt_mode", "noise_type", "noise_level"]
GROUP_KEYS = BASE_GROUP_KEYS
MODEL_COMPLEXITY_COLUMNS = [
    "params",
    "trainable_params",
    "FLOPs",
    "GFLOPs",
    "GLOPs",
]


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
            return pd.DataFrame(columns=BASE_GROUP_KEYS)

        if "experiment_type" not in df.columns:
            df["experiment_type"] = "main_prompt_mode_benchmark"
        if "prompt_variant" not in df.columns:
            df["prompt_variant"] = "default"

        for metric in METRICS:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")
                df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)
        for col in MODEL_COMPLEXITY_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        group_keys = [
            "experiment_type",
            "dataset",
            "model",
            "prompt_mode",
            "prompt_variant",
            "noise_type",
            "noise_level",
        ]
        grouped = df.groupby(group_keys, dropna=False)
        rows: List[Dict[str, Any]] = []
        for keys, g in grouped:
            row = dict(zip(group_keys, keys))
            if "is_gt_empty" in g.columns:
                gt_empty = pd.to_numeric(g["is_gt_empty"], errors="coerce").fillna(0)
                row["gt_empty_rate"] = float(gt_empty.mean()) if len(gt_empty) else np.nan
                row["n_gt_non_empty"] = int((gt_empty == 0).sum())
            if "is_pred_empty" in g.columns:
                pred_empty = pd.to_numeric(g["is_pred_empty"], errors="coerce").fillna(0)
                row["pred_empty_rate"] = float(pred_empty.mean()) if len(pred_empty) else np.nan
            if "Dice" in g.columns:
                dice_vals = pd.to_numeric(g["Dice"], errors="coerce")
                row["failure_rate_dice_lt_0_5"] = float((dice_vals < 0.5).mean()) if len(dice_vals) else np.nan
                row["failure_rate_dice_lt_0_7"] = float((dice_vals < 0.7).mean()) if len(dice_vals) else np.nan
            if "is_bbox_center_inside_mask" in g.columns:
                inside_vals = pd.to_numeric(g["is_bbox_center_inside_mask"], errors="coerce").dropna()
                row["bbox_center_inside_mask_percentage"] = (
                    float(inside_vals.mean() * 100.0) if len(inside_vals) else np.nan
                )
            for col in MODEL_COMPLEXITY_COLUMNS:
                if col not in g.columns:
                    continue
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                row[col] = float(vals.iloc[0]) if len(vals) else np.nan
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
                [
                    "experiment_type",
                    "dataset",
                    "model",
                    "prompt_mode",
                    "prompt_variant",
                    "noise_type",
                    "__level_idx",
                    "noise_level",
                ]
            )
            out = out.drop(columns=["__level_idx"])
        return out

    def aggregate_and_save(self, raw_csv: Path) -> Path:
        """Aggregate a raw CSV and write the result alongside it as ``*_stats.csv``."""
        stats_df = self.aggregate_file(raw_csv)
        stats_path = raw_csv.with_name(raw_csv.name.replace("_raw.csv", "_stats.csv"))
        stats_df.to_csv(stats_path, index=False)
        return stats_path
