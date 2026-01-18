"""
Stability metrics for evaluating model robustness under noise.
Includes:
  - PerfDrop: Dice(L0) - Dice(L4)
  - AUC over levels
  - MaskConsistency: IoU(mask_L0, mask_Lk)
  - Optional confidence metrics
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from metrics.seg import iou


def perf_drop(dice_l0: float, dice_l4: float) -> float:
    """
    Calculate performance drop from clean to most severe noise.
    
    PerfDrop = Dice(L0) - Dice(L4)
    Positive value means performance degradation.
    """
    return float(dice_l0 - dice_l4)


def perf_drop_relative(dice_l0: float, dice_l4: float) -> float:
    """
    Relative performance drop as percentage.
    
    RelDrop = (Dice(L0) - Dice(L4)) / Dice(L0) * 100
    """
    if dice_l0 <= 0:
        return 0.0
    return float((dice_l0 - dice_l4) / dice_l0 * 100.0)


def mask_consistency(mask_l0: np.ndarray, mask_lk: np.ndarray) -> float:
    """
    Calculate mask consistency between clean and noisy predictions.
    
    MaskConsistency = IoU(mask_L0, mask_Lk)
    Higher is better (more consistent predictions).
    """
    return iou(mask_lk, mask_l0)


def auc_over_levels(metrics_per_level: Dict[str, float], levels: List[str] = None) -> float:
    """
    Calculate Area Under Curve for metric across noise levels.
    Uses trapezoidal integration.
    
    Args:
        metrics_per_level: Dict mapping level -> metric value
        levels: Ordered list of levels (default: L0-L4)
        
    Returns:
        AUC value (normalized by number of intervals)
    """
    if levels is None:
        levels = ["L0", "L1", "L2", "L3", "L4"]
    
    values = []
    for lv in levels:
        if lv in metrics_per_level:
            values.append(float(metrics_per_level[lv]))
        else:
            values.append(np.nan)
    
    values = np.array(values)
    valid = ~np.isnan(values)
    
    if valid.sum() < 2:
        return 0.0
    
    # Fill NaN with linear interpolation
    x = np.arange(len(values))
    valid_x = x[valid]
    valid_v = values[valid]
    values = np.interp(x, valid_x, valid_v)
    
    # Trapezoidal AUC
    auc = np.trapz(values, x)
    # Normalize by range
    auc = auc / (len(values) - 1)
    
    return float(auc)


def compute_stability_metrics(
    dice_per_level: Dict[str, float],
    iou_per_level: Dict[str, float],
    masks_per_level: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive stability metrics.
    
    Args:
        dice_per_level: Dice scores per level
        iou_per_level: IoU scores per level
        masks_per_level: Optional prediction masks per level
        
    Returns:
        Dict with stability metrics
    """
    result = {}
    
    # Performance drop
    if "L0" in dice_per_level and "L4" in dice_per_level:
        result["perf_drop_dice"] = perf_drop(dice_per_level["L0"], dice_per_level["L4"])
        result["perf_drop_dice_rel"] = perf_drop_relative(dice_per_level["L0"], dice_per_level["L4"])
    
    if "L0" in iou_per_level and "L4" in iou_per_level:
        result["perf_drop_iou"] = perf_drop(iou_per_level["L0"], iou_per_level["L4"])
    
    # AUC over levels
    result["auc_dice"] = auc_over_levels(dice_per_level)
    result["auc_iou"] = auc_over_levels(iou_per_level)
    
    # Mask consistency
    if masks_per_level is not None and "L0" in masks_per_level:
        mask_l0 = masks_per_level["L0"]
        for lv in ["L1", "L2", "L3", "L4"]:
            if lv in masks_per_level:
                mc = mask_consistency(mask_l0, masks_per_level[lv])
                result[f"mask_consistency_{lv}"] = mc
        
        # Average mask consistency
        mc_values = [result.get(f"mask_consistency_{lv}") for lv in ["L1", "L2", "L3", "L4"]]
        mc_values = [v for v in mc_values if v is not None]
        if mc_values:
            result["mask_consistency_avg"] = float(np.mean(mc_values))
    
    return result


def compute_sample_stability(df: pd.DataFrame, sample_id: str, model: str, weight: str, mode: str, noise: str) -> Dict[str, float]:
    """
    Compute stability metrics for a single sample across levels.
    
    Args:
        df: Results DataFrame
        sample_id: Sample ID
        model: Model name
        weight: Weight ID
        mode: Inference mode
        noise: Noise type
        
    Returns:
        Dict with stability metrics
    """
    sub = df[
        (df["id"] == sample_id) &
        (df["model"] == model) &
        (df["weight"] == weight) &
        (df["mode"] == mode) &
        (df["noise"].isin([noise, "clean"]))
    ].copy()
    
    if len(sub) == 0:
        return {}
    
    dice_per_level = {}
    iou_per_level = {}
    
    # Get clean baseline from P0
    clean = sub[(sub["protocol"] == "P0") | (sub["noise"] == "clean")]
    if len(clean) > 0:
        dice_per_level["L0"] = float(clean.iloc[0]["dice"])
        iou_per_level["L0"] = float(clean.iloc[0]["iou"])
    
    # Get noisy results from P1
    noisy = sub[(sub["protocol"] == "P1") & (sub["noise"] == noise)]
    for _, row in noisy.iterrows():
        lv = row["level"]
        dice_per_level[lv] = float(row["dice"])
        iou_per_level[lv] = float(row["iou"])
    
    return compute_stability_metrics(dice_per_level, iou_per_level)


def compute_aggregate_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate stability metrics across all samples.
    
    Groups by (dataset, model, weight, mode, noise) and computes:
      - perf_drop_mean, perf_drop_std
      - auc_dice_mean, auc_iou_mean
      - mask_consistency_avg
    """
    # Get clean baseline metrics
    base = df[df["protocol"] == "P0"].copy()
    base = base.rename(columns={"dice": "dice_L0", "iou": "iou_L0"})
    base = base[["dataset", "model", "weight", "mode", "id", "dice_L0", "iou_L0"]]
    
    # Get worst case (L4) metrics
    worst = df[(df["protocol"] == "P1") & (df["level"] == "L4")].copy()
    worst = worst.rename(columns={"dice": "dice_L4", "iou": "iou_L4"})
    worst = worst[["dataset", "model", "weight", "mode", "noise", "id", "dice_L4", "iou_L4"]]
    
    # Merge
    merged = worst.merge(base, on=["dataset", "model", "weight", "mode", "id"], how="left")
    
    # Compute perf_drop
    merged["perf_drop"] = merged["dice_L0"] - merged["dice_L4"]
    merged["perf_drop_rel"] = (merged["dice_L0"] - merged["dice_L4"]) / merged["dice_L0"].clip(lower=1e-6) * 100
    
    # Add mask consistency if available
    if "mask_consistency_iou" in df.columns:
        mc = df[(df["protocol"] == "P1") & (df["level"] == "L4")][
            ["dataset", "model", "weight", "mode", "noise", "id", "mask_consistency_iou"]
        ].copy()
        merged = merged.merge(mc, on=["dataset", "model", "weight", "mode", "noise", "id"], how="left")
    
    # Aggregate
    group_keys = ["dataset", "model", "weight", "mode", "noise"]
    agg_dict = {
        "perf_drop": ["mean", "std", "min", "max"],
        "perf_drop_rel": ["mean", "std"],
    }
    
    if "mask_consistency_iou" in merged.columns:
        agg_dict["mask_consistency_iou"] = ["mean", "std"]
    
    stab = merged.groupby(group_keys).agg(agg_dict).reset_index()
    
    # Flatten columns
    stab.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in stab.columns
    ]
    
    # Add count
    counts = merged.groupby(group_keys).size().reset_index(name="n_samples")
    stab = stab.merge(counts, on=group_keys)
    
    return stab


def identify_failure_cases(df: pd.DataFrame, top_k: int = 10, metric: str = "dice") -> pd.DataFrame:
    """
    Identify samples with largest performance drop.
    
    Args:
        df: Results DataFrame
        top_k: Number of top failure cases to return
        metric: Metric to use for comparison
        
    Returns:
        DataFrame with top failure cases
    """
    # Get clean baseline
    base = df[df["protocol"] == "P0"].copy()
    base = base.rename(columns={metric: f"{metric}_L0"})
    base = base[["dataset", "model", "weight", "mode", "id", f"{metric}_L0"]]
    
    # Get L4 results
    l4 = df[(df["protocol"] == "P1") & (df["level"] == "L4")].copy()
    l4 = l4.rename(columns={metric: f"{metric}_L4"})
    
    # Merge
    merged = l4.merge(base, on=["dataset", "model", "weight", "mode", "id"], how="left")
    merged["drop"] = merged[f"{metric}_L0"] - merged[f"{metric}_L4"]
    
    # Sort by drop (descending)
    failures = merged.nlargest(top_k, "drop")
    
    return failures
