"""
Comprehensive Statistics Module for SAM Robustness Benchmark.

Generates all required statistical summaries from raw/merged CSV data:
1. Overall summary
2. Model-wise summary
3. Mode-wise summary
4. Noise-wise summary
5. Level-wise summary (L0-L9 complete)
6. Model × Noise summary
7. Model × Level summary
8. Noise × Level summary
9. Metric-wise summary
10. Stability/robustness statistics

Key features:
- Handles metric direction correctly (HD is lower-is-better)
- Preserves all levels L0-L9
- Exports to CSV and optional LaTeX/Markdown
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Numpy 2.x compatibility: trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

METRICS = [
    "IoU",
    "Dice",
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
GROUP_KEYS = ["dataset", "model", "prompt_mode", "noise_type", "noise_level"]

# Metric direction: True = higher-is-better, False = lower-is-better
METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "IoU": True,
    "Dice": True,
    "Recall": True,
    "Precision": True,
    "F1": True,
    "HD": False,  # Hausdorff Distance: lower is better
    "HD_px": False,
    "HD95_px": False,
    "HD_mm": False,
    "HD95_mm": False,
    "inference_time_ms": False,
    "FPS": True,
}


def _level_key(level: str) -> int:
    """Extract numeric index from level string (L0→0, L9→9)."""
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _sorted_levels(levels: List[str]) -> List[str]:
    """Sort levels L0, L1, ..., L9 in correct order."""
    return sorted(set(str(lv) for lv in levels), key=_level_key)


def _is_clean_level(level: str) -> bool:
    """Check if level is L0 (clean)."""
    return str(level).strip().upper() == "L0"


def _metric_direction_multiplier(metric: str) -> float:
    """
    Return +1 for higher-is-better, -1 for lower-is-better.
    Used for ranking and drop calculations.
    """
    return 1.0 if METRIC_HIGHER_IS_BETTER.get(metric, True) else -1.0


def _compute_drop(clean_val: float, noisy_val: float, metric: str) -> float:
    """
    Compute degradation drop from clean to noisy.

    For higher-is-better metrics: drop = clean - noisy (positive = degraded)
    For lower-is-better metrics: drop = noisy - clean (positive = degraded)
    """
    if not np.isfinite(clean_val) or not np.isfinite(noisy_val):
        return np.nan

    if METRIC_HIGHER_IS_BETTER.get(metric, True):
        return clean_val - noisy_val
    else:
        return noisy_val - clean_val


def _compute_relative_drop(clean_val: float, noisy_val: float, metric: str) -> float:
    """Compute relative (percentage) drop."""
    drop = _compute_drop(clean_val, noisy_val, metric)
    if not np.isfinite(drop):
        return np.nan

    # For relative drop, we use absolute clean value as reference
    ref = abs(clean_val) if np.isfinite(clean_val) and abs(clean_val) > 1e-10 else 1.0
    return (drop / ref) * 100.0


# ═══════════════════════════════════════════════════════════════════════════
#  COMPREHENSIVE STATISTICS GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class ComprehensiveStatistics:
    """
    Generate comprehensive statistical summaries from benchmark CSV data.

    Parameters
    ----------
    csv_path : Path
        Path to merged statistics CSV (statistics_merged.csv) or raw CSV.
    output_dir : Path
        Directory to save statistical outputs.
    """

    def __init__(self, csv_path: Path, output_dir: Path) -> None:
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.csv_path)
        self._preprocess()

    def _preprocess(self) -> None:
        """Standardize column names and add derived columns."""
        if "experiment_type" not in self.df.columns:
            self.df["experiment_type"] = "main_prompt_mode_benchmark"
        if "prompt_variant" not in self.df.columns:
            self.df["prompt_variant"] = "default"

        # Ensure string columns
        for col in [
            "experiment_type",
            "dataset",
            "model",
            "prompt_mode",
            "prompt_variant",
            "noise_type",
            "noise_level",
        ]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

        # Add level index for sorting
        if "noise_level" in self.df.columns:
            self.df["level_idx"] = self.df["noise_level"].map(_level_key)

        # Add is_clean flag
        if "noise_level" in self.df.columns:
            self.df["is_clean"] = self.df["noise_level"].map(_is_clean_level)

        # Coerce metrics to numeric
        for metric in METRICS:
            if metric in self.df.columns:
                self.df[metric] = pd.to_numeric(self.df[metric], errors="coerce")

        for col in [
            "failure_rate_dice_lt_0_5",
            "failure_rate_dice_lt_0_7",
            "bbox_center_inside_mask_percentage",
        ]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _available_metrics(self) -> List[str]:
        """Return metrics actually present in data."""
        return [m for m in METRICS if m in self.df.columns]

    def _available_levels(self) -> List[str]:
        """Return all levels present in data, sorted."""
        if "noise_level" not in self.df.columns:
            return []
        return _sorted_levels(self.df["noise_level"].dropna().unique().tolist())

    # ─────────────────────────────────────────────────────────────────────────
    #  1. OVERALL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def overall_summary(self) -> pd.DataFrame:
        """
        Generate overall statistics for each metric.

        Returns DataFrame with:
        - metric
        - overall_mean, overall_std, overall_min, overall_max
        - clean_mean (L0 only)
        - noisy_mean (L1-L9)
        - drop (clean → noisy, direction-aware)
        - relative_drop_pct
        """
        metrics = self._available_metrics()
        rows = []

        for metric in metrics:
            vals = self.df[metric].dropna()

            # Overall stats
            overall_mean = vals.mean() if len(vals) else np.nan
            overall_std = vals.std() if len(vals) else np.nan
            overall_min = vals.min() if len(vals) else np.nan
            overall_max = vals.max() if len(vals) else np.nan

            # Clean (L0) stats
            clean_vals = self.df.loc[self.df["is_clean"] == True, metric].dropna()
            clean_mean = clean_vals.mean() if len(clean_vals) else np.nan

            # Noisy (L1-L9) stats
            noisy_vals = self.df.loc[self.df["is_clean"] == False, metric].dropna()
            noisy_mean = noisy_vals.mean() if len(noisy_vals) else np.nan

            # Drop
            drop = _compute_drop(clean_mean, noisy_mean, metric)
            rel_drop = _compute_relative_drop(clean_mean, noisy_mean, metric)

            rows.append({
                "metric": metric,
                "direction": "higher_better" if METRIC_HIGHER_IS_BETTER.get(metric, True) else "lower_better",
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "overall_min": overall_min,
                "overall_max": overall_max,
                "clean_mean": clean_mean,
                "noisy_mean": noisy_mean,
                "drop": drop,
                "relative_drop_pct": rel_drop,
                "n_samples": len(vals),
            })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    #  2. MODEL-WISE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def model_summary(self) -> pd.DataFrame:
        """
        Generate per-model statistics.

        For each (model, metric):
        - clean_perf, noisy_mean, overall_mean
        - std_across_levels, std_across_noises
        - drop, relative_drop
        - robustness_rank (direction-aware)
        """
        metrics = self._available_metrics()
        rows = []

        for model in self.df["model"].dropna().unique():
            model_df = self.df[self.df["model"] == model]

            for metric in metrics:
                vals = model_df[metric].dropna()
                clean_vals = model_df.loc[model_df["is_clean"] == True, metric].dropna()
                noisy_vals = model_df.loc[model_df["is_clean"] == False, metric].dropna()

                clean_mean = clean_vals.mean() if len(clean_vals) else np.nan
                noisy_mean = noisy_vals.mean() if len(noisy_vals) else np.nan
                overall_mean = vals.mean() if len(vals) else np.nan

                # Std across levels
                level_means = model_df.groupby("noise_level")[metric].mean()
                std_levels = level_means.std() if len(level_means) > 1 else np.nan

                # Std across noise types
                noise_means = model_df.groupby("noise_type")[metric].mean()
                std_noises = noise_means.std() if len(noise_means) > 1 else np.nan

                drop = _compute_drop(clean_mean, noisy_mean, metric)
                rel_drop = _compute_relative_drop(clean_mean, noisy_mean, metric)

                rows.append({
                    "model": model,
                    "metric": metric,
                    "clean_perf": clean_mean,
                    "noisy_mean": noisy_mean,
                    "overall_mean": overall_mean,
                    "std_across_levels": std_levels,
                    "std_across_noises": std_noises,
                    "drop": drop,
                    "relative_drop_pct": rel_drop,
                })

        result = pd.DataFrame(rows)

        # Add ranking per metric (direction-aware)
        for metric in metrics:
            mask = result["metric"] == metric
            mult = _metric_direction_multiplier(metric)
            # Higher score = better ranking, so we negate for lower-is-better
            result.loc[mask, "rank"] = (
                result.loc[mask, "overall_mean"] * mult
            ).rank(ascending=False, method="min")

        return result.sort_values(["metric", "rank"])

    # ─────────────────────────────────────────────────────────────────────────
    #  3. MODE-WISE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def mode_summary(self) -> pd.DataFrame:
        """Compare prompt modes: bbox, point, point+bbox."""
        metrics = self._available_metrics()
        rows = []

        for mode in self.df["prompt_mode"].dropna().unique():
            mode_df = self.df[self.df["prompt_mode"] == mode]

            for metric in metrics:
                vals = mode_df[metric].dropna()
                clean_vals = mode_df.loc[mode_df["is_clean"] == True, metric].dropna()
                noisy_vals = mode_df.loc[mode_df["is_clean"] == False, metric].dropna()

                rows.append({
                    "prompt_mode": mode,
                    "metric": metric,
                    "mean": vals.mean() if len(vals) else np.nan,
                    "std": vals.std() if len(vals) else np.nan,
                    "clean_mean": clean_vals.mean() if len(clean_vals) else np.nan,
                    "noisy_mean": noisy_vals.mean() if len(noisy_vals) else np.nan,
                    "n_samples": len(vals),
                })

        result = pd.DataFrame(rows)

        # Add ranking
        for metric in metrics:
            mask = result["metric"] == metric
            mult = _metric_direction_multiplier(metric)
            result.loc[mask, "rank"] = (
                result.loc[mask, "mean"] * mult
            ).rank(ascending=False, method="min")

        return result.sort_values(["metric", "rank"])

    # ─────────────────────────────────────────────────────────────────────────
    #  4. NOISE-WISE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def noise_summary(self) -> pd.DataFrame:
        """
        Analyze each noise type's destructiveness.

        Returns per (noise_type, metric):
        - mean across models
        - degradation from L0
        - difficulty ranking (most destructive = rank 1)
        """
        metrics = self._available_metrics()
        rows = []

        for noise in self.df["noise_type"].dropna().unique():
            noise_df = self.df[self.df["noise_type"] == noise]

            for metric in metrics:
                vals = noise_df[metric].dropna()
                clean_vals = noise_df.loc[noise_df["is_clean"] == True, metric].dropna()
                noisy_vals = noise_df.loc[noise_df["is_clean"] == False, metric].dropna()

                clean_mean = clean_vals.mean() if len(clean_vals) else np.nan
                noisy_mean = noisy_vals.mean() if len(noisy_vals) else np.nan
                drop = _compute_drop(clean_mean, noisy_mean, metric)

                rows.append({
                    "noise_type": noise,
                    "metric": metric,
                    "mean_across_models": vals.mean() if len(vals) else np.nan,
                    "clean_baseline": clean_mean,
                    "noisy_mean": noisy_mean,
                    "degradation": drop,
                })

        result = pd.DataFrame(rows)

        # Difficulty ranking: higher degradation = more destructive = rank 1
        for metric in metrics:
            mask = result["metric"] == metric
            result.loc[mask, "difficulty_rank"] = (
                result.loc[mask, "degradation"]
            ).rank(ascending=False, method="min", na_option="bottom")

        return result.sort_values(["metric", "difficulty_rank"])

    # ─────────────────────────────────────────────────────────────────────────
    #  5. LEVEL-WISE SUMMARY (ALL L0-L9)
    # ─────────────────────────────────────────────────────────────────────────

    def level_summary(self) -> pd.DataFrame:
        """
        Statistics for each level L0-L9 (COMPLETE, no levels dropped).

        Shows degradation progression across severity levels.
        """
        metrics = self._available_metrics()
        levels = self._available_levels()
        rows = []

        for level in levels:
            level_df = self.df[self.df["noise_level"] == level]

            for metric in metrics:
                vals = level_df[metric].dropna()

                rows.append({
                    "noise_level": level,
                    "level_idx": _level_key(level),
                    "metric": metric,
                    "mean": vals.mean() if len(vals) else np.nan,
                    "std": vals.std() if len(vals) else np.nan,
                    "min": vals.min() if len(vals) else np.nan,
                    "max": vals.max() if len(vals) else np.nan,
                    "n_samples": len(vals),
                })

        return pd.DataFrame(rows).sort_values(["metric", "level_idx"])

    # ─────────────────────────────────────────────────────────────────────────
    #  6. MODEL × NOISE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def model_noise_matrix(self, metric: str = "Dice") -> pd.DataFrame:
        """
        Pivot table: rows=models, columns=noise_types, values=mean metric.
        Suitable for heatmap visualization.
        """
        if metric not in self.df.columns:
            return pd.DataFrame()

        pivot = (
            self.df.groupby(["model", "noise_type"])[metric]
            .mean()
            .reset_index()
            .pivot(index="model", columns="noise_type", values=metric)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        return pivot

    # ─────────────────────────────────────────────────────────────────────────
    #  7. MODEL × LEVEL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def model_level_matrix(self, metric: str = "Dice") -> pd.DataFrame:
        """
        Pivot table: rows=models, columns=levels (L0-L9 COMPLETE).
        Shows model stability across severity progression.
        """
        if metric not in self.df.columns:
            return pd.DataFrame()

        pivot = (
            self.df.groupby(["model", "noise_level", "level_idx"])[metric]
            .mean()
            .reset_index()
            .sort_values("level_idx")
            .pivot(index="model", columns="noise_level", values=metric)
            .sort_index(axis=0)
        )

        # Reorder columns by level index
        levels = _sorted_levels(pivot.columns.tolist())
        pivot = pivot[[lv for lv in levels if lv in pivot.columns]]
        return pivot

    # ─────────────────────────────────────────────────────────────────────────
    #  8. NOISE × LEVEL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────

    def noise_level_matrix(self, metric: str = "Dice") -> pd.DataFrame:
        """
        Pivot table: rows=noise_types, columns=levels (L0-L9 COMPLETE).
        Shows severity progression for each noise type.
        """
        if metric not in self.df.columns:
            return pd.DataFrame()

        pivot = (
            self.df.groupby(["noise_type", "noise_level", "level_idx"])[metric]
            .mean()
            .reset_index()
            .sort_values("level_idx")
            .pivot(index="noise_type", columns="noise_level", values=metric)
            .sort_index(axis=0)
        )

        levels = _sorted_levels(pivot.columns.tolist())
        pivot = pivot[[lv for lv in levels if lv in pivot.columns]]
        return pivot

    # ─────────────────────────────────────────────────────────────────────────
    #  9. METRIC-WISE EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_metric_summaries(self) -> Dict[str, Path]:
        """Export separate summary for each metric."""
        metrics = self._available_metrics()
        paths = {}

        metric_dir = self.output_dir / "by_metric"
        metric_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            # Filter data for this metric
            metric_slug = metric.lower()
            is_lower_better = not METRIC_HIGHER_IS_BETTER.get(metric, True)

            summary_rows = []
            for model in self.df["model"].dropna().unique():
                model_df = self.df[self.df["model"] == model]
                vals = model_df[metric].dropna()
                clean_vals = model_df.loc[model_df["is_clean"] == True, metric].dropna()
                noisy_vals = model_df.loc[model_df["is_clean"] == False, metric].dropna()

                clean_mean = clean_vals.mean() if len(clean_vals) else np.nan
                noisy_mean = noisy_vals.mean() if len(noisy_vals) else np.nan

                summary_rows.append({
                    "model": model,
                    "clean": clean_mean,
                    "noisy_mean": noisy_mean,
                    "overall": vals.mean() if len(vals) else np.nan,
                    "drop": _compute_drop(clean_mean, noisy_mean, metric),
                })

            summary_df = pd.DataFrame(summary_rows)

            # Rank: for HD (lower is better), lower overall = better = rank 1
            if is_lower_better:
                summary_df["rank"] = summary_df["overall"].rank(ascending=True, method="min")
            else:
                summary_df["rank"] = summary_df["overall"].rank(ascending=False, method="min")

            summary_df = summary_df.sort_values("rank")

            out_path = metric_dir / f"summary_{metric_slug}.csv"
            summary_df.to_csv(out_path, index=False)
            paths[metric] = out_path

        return paths

    # ─────────────────────────────────────────────────────────────────────────
    #  10. ROBUSTNESS STATISTICS
    # ─────────────────────────────────────────────────────────────────────────

    def robustness_analysis(self) -> pd.DataFrame:
        """
        Comprehensive robustness analysis per model.

        Computes:
        - clean_score
        - noisy_mean
        - relative_drop
        - degradation_slope (linear fit across L0-L9)
        - AUC_robustness (area under performance-vs-level curve)
        - stability_rank
        """
        metrics = self._available_metrics()
        levels = self._available_levels()
        rows = []

        for model in self.df["model"].dropna().unique():
            model_df = self.df[self.df["model"] == model]

            for metric in metrics:
                # Get level-wise means
                level_means = (
                    model_df.groupby("noise_level")[metric]
                    .mean()
                    .reindex(levels)
                )

                clean_score = level_means.get("L0", np.nan)
                noisy_scores = level_means.drop("L0", errors="ignore").dropna()
                noisy_mean = noisy_scores.mean() if len(noisy_scores) else np.nan

                rel_drop = _compute_relative_drop(clean_score, noisy_mean, metric)

                # Degradation slope (linear regression on level index vs metric)
                valid_levels = level_means.dropna()
                if len(valid_levels) >= 2:
                    x = np.array([_level_key(lv) for lv in valid_levels.index])
                    y = valid_levels.values
                    try:
                        slope = np.polyfit(x, y, 1)[0]
                    except Exception:
                        slope = np.nan
                else:
                    slope = np.nan

                # AUC robustness (trapezoid integration)
                if len(valid_levels) >= 2:
                    x_vals = np.array([_level_key(lv) for lv in valid_levels.index])
                    y_vals = valid_levels.values
                    # Normalize x to [0, 1]
                    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-10)
                    auc = _trapz(y_vals, x_norm)
                    # For HD (lower is better), we want higher AUC to mean worse
                    # So we don't flip it here, but note in interpretation
                else:
                    auc = np.nan

                rows.append({
                    "model": model,
                    "metric": metric,
                    "clean_score": clean_score,
                    "noisy_mean": noisy_mean,
                    "relative_drop_pct": rel_drop,
                    "degradation_slope": slope,
                    "auc_robustness": auc,
                })

        result = pd.DataFrame(rows)

        # Stability rank based on relative drop (lower drop = more stable = rank 1)
        for metric in metrics:
            mask = result["metric"] == metric
            result.loc[mask, "stability_rank"] = (
                result.loc[mask, "relative_drop_pct"].abs()
            ).rank(ascending=True, method="min", na_option="bottom")

        return result.sort_values(["metric", "stability_rank"])

    # ─────────────────────────────────────────────────────────────────────────
    #  GENERATE ALL & EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def _benchmark_summary(self, df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=group_keys)

        rows = []
        metrics = [m for m in self._available_metrics() if m in df.columns]
        extra_numeric = [
            c for c in [
                "failure_rate_dice_lt_0_5",
                "failure_rate_dice_lt_0_7",
                "bbox_center_inside_mask_percentage",
            ]
            if c in df.columns
        ]

        grouped = df.groupby(group_keys, dropna=False)
        for keys, g in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_keys, keys))
            for metric in metrics:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{metric}_std"] = vals.std(ddof=0) if len(vals) else np.nan
            for col in extra_numeric:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                row[col] = vals.mean() if len(vals) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def main_prompt_mode_summary(self) -> pd.DataFrame:
        df = self.df[self.df["experiment_type"] == "main_prompt_mode_benchmark"].copy()
        return self._benchmark_summary(
            df,
            ["dataset", "model", "prompt_mode", "noise_type", "noise_level"],
        )

    def prompt_variant_summary(self) -> pd.DataFrame:
        df = self.df[self.df["experiment_type"] == "prompt_variant_benchmark"].copy()
        return self._benchmark_summary(
            df,
            ["dataset", "model", "prompt_mode", "prompt_variant", "noise_type", "noise_level"],
        )

    def prompt_variant_comparison(self) -> pd.DataFrame:
        summary = self.prompt_variant_summary()
        if summary.empty:
            return summary
        summary = summary.copy()
        summary["prompt_variant_comparison"] = (
            summary["prompt_mode"].astype(str) + ":" + summary["prompt_variant"].astype(str)
        )
        return summary

    def generate_all(self) -> Dict[str, Path]:
        """Generate all statistics and export to CSV files."""
        paths = {}

        # 1. Overall summary
        overall = self.overall_summary()
        p = self.output_dir / "01_overall_summary.csv"
        overall.to_csv(p, index=False)
        paths["overall_summary"] = p

        # 2. Model summary
        model = self.model_summary()
        p = self.output_dir / "02_model_summary.csv"
        model.to_csv(p, index=False)
        paths["model_summary"] = p

        # 3. Mode summary
        mode = self.mode_summary()
        p = self.output_dir / "03_mode_summary.csv"
        mode.to_csv(p, index=False)
        paths["mode_summary"] = p

        # 4. Noise summary
        noise = self.noise_summary()
        p = self.output_dir / "04_noise_summary.csv"
        noise.to_csv(p, index=False)
        paths["noise_summary"] = p

        # 5. Level summary
        level = self.level_summary()
        p = self.output_dir / "05_level_summary.csv"
        level.to_csv(p, index=False)
        paths["level_summary"] = p

        # 6-8. Matrix summaries per metric
        metrics = self._available_metrics()
        for metric in metrics:
            metric_slug = metric.lower()

            # Model × Noise
            mn = self.model_noise_matrix(metric)
            if not mn.empty:
                p = self.output_dir / f"06_model_noise_matrix_{metric_slug}.csv"
                mn.to_csv(p)
                paths[f"model_noise_{metric_slug}"] = p

            # Model × Level
            ml = self.model_level_matrix(metric)
            if not ml.empty:
                p = self.output_dir / f"07_model_level_matrix_{metric_slug}.csv"
                ml.to_csv(p)
                paths[f"model_level_{metric_slug}"] = p

            # Noise × Level
            nl = self.noise_level_matrix(metric)
            if not nl.empty:
                p = self.output_dir / f"08_noise_level_matrix_{metric_slug}.csv"
                nl.to_csv(p)
                paths[f"noise_level_{metric_slug}"] = p

        # 9. Metric-wise exports
        metric_paths = self.export_metric_summaries()
        paths.update({f"metric_{k}": v for k, v in metric_paths.items()})

        # 10. Robustness analysis
        robust = self.robustness_analysis()
        p = self.output_dir / "10_robustness_analysis.csv"
        robust.to_csv(p, index=False)
        paths["robustness_analysis"] = p

        main_prompt = self.main_prompt_mode_summary()
        p = self.output_dir / "11_main_prompt_mode_summary.csv"
        main_prompt.to_csv(p, index=False)
        paths["main_prompt_mode_summary"] = p

        prompt_variant = self.prompt_variant_summary()
        p = self.output_dir / "12_prompt_variant_summary.csv"
        prompt_variant.to_csv(p, index=False)
        paths["prompt_variant_summary"] = p

        prompt_comparison = self.prompt_variant_comparison()
        p = self.output_dir / "13_prompt_variant_comparison.csv"
        prompt_comparison.to_csv(p, index=False)
        paths["prompt_variant_comparison"] = p

        # Summary manifest
        manifest = pd.DataFrame([
            {"file": str(v.name), "description": k}
            for k, v in paths.items()
        ])
        manifest.to_csv(self.output_dir / "statistics_manifest.csv", index=False)

        return paths


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def generate_comprehensive_statistics(
    csv_path: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate all comprehensive statistics from benchmark CSV.

    Parameters
    ----------
    csv_path : Path
        Path to merged statistics CSV or raw CSV.
    output_dir : Path
        Directory to save outputs.

    Returns
    -------
    Dict[str, Path]
        Mapping from summary name to output file path.
    """
    stats = ComprehensiveStatistics(csv_path, output_dir)
    return stats.generate_all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python comprehensive_statistics.py <csv_path> <output_dir>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    paths = generate_comprehensive_statistics(csv_path, output_dir)
    print(f"Generated {len(paths)} statistical files in {output_dir}")
    for name, path in paths.items():
        print(f"  - {name}: {path.name}")
