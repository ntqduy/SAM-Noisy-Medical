"""
Aggregation and stability computation for AIO25 NoisySAM benchmark.

Extended features:
  - Level stability: drop_Lmax, slope, AUC, CV
  - Seed stability: seed_std, seed_cv
  - Group-level summaries
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


GROUP_KEYS = ["phase", "dataset", "model", "weight", "mode", "protocol", "noise", "level"]


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-sample results to group-level statistics.
    
    Args:
        df: Per-sample results DataFrame
        
    Returns:
        Aggregated DataFrame with mean, std, count per group
    """
    metrics = ["dice", "iou", "hd95"]
    
    # Add optional metrics if present
    optional = ["pred_iou_score", "psnr", "ssim", "mean_confidence", "mean_entropy",
                "boundary_entropy", "mean_confidence_proxy", "mask_consistency_iou",
                "intensity_scalar", "severity_scalar"]
    metrics += [c for c in optional if c in df.columns]

    agg = {}
    for m in metrics:
        if m in df.columns:
            agg[m] = ["mean", "std", "count"]

    if not agg:
        return pd.DataFrame()

    out = df.groupby(GROUP_KEYS, dropna=False).agg(agg).reset_index()
    # Flatten columns
    out.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in out.columns]
    return out


def compute_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic stability computation: PerfDrop = Dice(L0) - Dice(L4).
    
    For backward compatibility.
    """
    base = df[df["protocol"] == "P0"].copy()
    base = base.rename(columns={"dice": "dice_L0"})[
        ["dataset", "model", "weight", "mode", "id", "dice_L0"]
    ]

    worst = df[(df["protocol"] == "P1") & (df["level"] == "L4")].copy()
    if len(worst) == 0:
        return pd.DataFrame()
    
    worst = worst.rename(columns={"dice": "dice_L4"})[
        ["dataset", "model", "weight", "mode", "noise", "id", "dice_L4"]
    ]

    merged = worst.merge(base, on=["dataset", "model", "weight", "mode", "id"], how="left")
    merged["perf_drop"] = merged["dice_L0"] - merged["dice_L4"]

    stab = merged.groupby(["dataset", "model", "weight", "mode", "noise"]).agg(
        perf_drop_mean=("perf_drop", "mean"),
        perf_drop_std=("perf_drop", "std"),
        n=("perf_drop", "count"),
    ).reset_index()
    return stab


def compute_stability_extended(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Compute comprehensive stability metrics:
    
    Level stability (across L0-L4):
      - drop_Lmax: metric(L0) - metric(L4)
      - slope: linear regression slope of metric vs intensity_scalar
      - auc: area under curve (normalized)
      - cv: coefficient of variation (std/mean)
    
    Seed stability (across noise seeds):
      - seed_std: standard deviation across seeds
      - seed_cv: coefficient of variation across seeds
    
    Args:
        df: Per-sample results DataFrame
        cfg: Configuration dictionary
        
    Returns:
        Stability summary DataFrame
    """
    cfg = cfg or {}
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Group keys for stability computation
    stability_group = ["dataset", "model", "weight", "mode", "noise", "protocol"]
    
    rows = []
    
    # Get unique groups
    if not all(k in df.columns for k in stability_group):
        return compute_stability(df)  # Fallback
    
    for keys, group in df.groupby(stability_group):
        dataset, model, weight, mode, noise, protocol = keys
        
        row = {
            "dataset": dataset,
            "model": model,
            "weight": weight,
            "mode": mode,
            "noise": noise,
            "protocol": protocol,
        }
        
        # Get P0 baseline for this group
        baseline = df[
            (df["dataset"] == dataset) &
            (df["model"] == model) &
            (df["weight"] == weight) &
            (df["mode"] == mode) &
            (df["protocol"] == "P0")
        ]
        
        if len(baseline) == 0:
            baseline_dice = None
        else:
            baseline_dice = baseline["dice"].mean()
        
        # Level stability metrics
        levels_present = group["level"].unique()
        level_metrics = group.groupby("level")["dice"].mean()
        
        # drop_Lmax
        if baseline_dice is not None and "L4" in level_metrics.index:
            row["drop_Lmax"] = float(baseline_dice - level_metrics["L4"])
        elif "L1" in level_metrics.index and len(level_metrics) > 1:
            max_level = level_metrics.index[-1]  # Assume ordered
            row["drop_Lmax"] = float(level_metrics.iloc[0] - level_metrics.iloc[-1])
        else:
            row["drop_Lmax"] = None
        
        # Slope (linear regression on intensity_scalar)
        if "intensity_scalar" in group.columns:
            level_data = group.groupby("level").agg({
                "dice": "mean",
                "intensity_scalar": "mean"
            }).reset_index()
            
            if len(level_data) >= 2:
                x = level_data["intensity_scalar"].values
                y = level_data["dice"].values
                
                # Simple linear regression
                if np.std(x) > 1e-6:
                    slope = np.polyfit(x, y, 1)[0]
                    row["slope"] = float(slope)
                else:
                    row["slope"] = 0.0
            else:
                row["slope"] = None
        else:
            row["slope"] = None
        
        # AUC (normalized area under curve)
        if len(level_metrics) >= 2:
            # Assume levels are ordered L1, L2, L3, L4
            y_values = level_metrics.values
            x_values = np.linspace(0, 1, len(y_values))
            auc = float(np.trapz(y_values, x_values))
            row["auc"] = auc
        else:
            row["auc"] = None
        
        # CV (coefficient of variation across levels)
        all_dice = group["dice"].values
        if len(all_dice) > 1 and np.mean(all_dice) > 1e-6:
            row["cv"] = float(np.std(all_dice) / np.mean(all_dice))
        else:
            row["cv"] = None
        
        # Seed stability (if multiple seeds)
        if "noise_seed" in group.columns:
            seeds = group["noise_seed"].unique()
            if len(seeds) > 1:
                seed_means = group.groupby("noise_seed")["dice"].mean()
                row["seed_std"] = float(seed_means.std())
                if seed_means.mean() > 1e-6:
                    row["seed_cv"] = float(seed_means.std() / seed_means.mean())
                else:
                    row["seed_cv"] = None
            else:
                row["seed_std"] = None
                row["seed_cv"] = None
        
        # Mean dice for this group
        row["dice_mean"] = float(group["dice"].mean())
        row["dice_std"] = float(group["dice"].std())
        row["n_samples"] = int(len(group))
        
        # Mean PSNR if available
        if "psnr" in group.columns:
            row["psnr_mean"] = float(group["psnr"].mean()) if group["psnr"].notna().any() else None
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    
    # Sort by drop_Lmax descending
    if "drop_Lmax" in result.columns:
        result = result.sort_values("drop_Lmax", ascending=False, na_position="last")
    
    return result


def compute_seed_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stability metrics specifically across noise seeds.
    
    For each (dataset, model, weight, mode, noise, level), computes
    statistics across different noise seeds.
    
    Returns:
        DataFrame with seed stability metrics
    """
    if "noise_seed" not in df.columns:
        return pd.DataFrame()
    
    group_keys = ["dataset", "model", "weight", "mode", "protocol", "noise", "level"]
    
    rows = []
    for keys, group in df.groupby(group_keys):
        seeds = group["noise_seed"].unique()
        
        if len(seeds) < 2:
            continue
        
        row = dict(zip(group_keys, keys))
        
        # Statistics across seeds
        seed_stats = group.groupby("noise_seed")["dice"].agg(["mean", "std"])
        
        row["n_seeds"] = len(seeds)
        row["dice_seed_mean"] = float(seed_stats["mean"].mean())
        row["dice_seed_std"] = float(seed_stats["mean"].std())
        
        if row["dice_seed_mean"] > 1e-6:
            row["dice_seed_cv"] = float(row["dice_seed_std"] / row["dice_seed_mean"])
        else:
            row["dice_seed_cv"] = None
        
        rows.append(row)
    
    return pd.DataFrame(rows)
