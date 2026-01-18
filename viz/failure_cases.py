"""
Failure case analysis and export.
Identifies and visualizes samples with largest performance drops.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from viz.overlays import overlay


def _safe_read_gray(path: str) -> np.ndarray:
    """Read image as grayscale."""
    return np.asarray(Image.open(path).convert("L")).astype(np.uint8)


def _safe_read_mask(path: str) -> np.ndarray:
    """Read mask as binary."""
    m = np.asarray(Image.open(path).convert("L")).astype(np.uint8)
    return (m > 0).astype(np.uint8)


def identify_top_failures(
    df: pd.DataFrame,
    top_k: int = 10,
    metric: str = "dice"
) -> pd.DataFrame:
    """
    Identify samples with largest performance drop from L0 to L4.
    
    Args:
        df: Results DataFrame
        top_k: Number of top failures to return
        metric: Metric to measure drop
        
    Returns:
        DataFrame with failure cases
    """
    # Get clean baseline (P0)
    base = df[df["protocol"] == "P0"].copy()
    if len(base) == 0:
        return pd.DataFrame()
    
    base = base.rename(columns={metric: f"{metric}_L0"})
    base = base[["dataset", "model", "weight", "mode", "id", f"{metric}_L0", "img_path", "mask_path"]]
    
    # Get worst case (L4) from P1
    worst = df[(df["protocol"] == "P1") & (df["level"] == "L4")].copy()
    if len(worst) == 0:
        return pd.DataFrame()
    
    worst = worst.rename(columns={metric: f"{metric}_L4"})
    keep_cols = ["dataset", "model", "weight", "mode", "noise", "id", f"{metric}_L4"]
    worst = worst[keep_cols]
    
    # Merge
    merged = worst.merge(base, on=["dataset", "model", "weight", "mode", "id"], how="left")
    
    # Compute drop
    merged["drop"] = merged[f"{metric}_L0"] - merged[f"{metric}_L4"]
    merged["drop_rel"] = merged["drop"] / merged[f"{metric}_L0"].clip(lower=1e-6) * 100
    
    # Sort by absolute drop
    failures = merged.nlargest(top_k, "drop")
    
    return failures


def export_failure_cases(
    df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    top_k: int = 8,
    metric: str = "dice"
) -> List[str]:
    """
    Export visualization of top failure cases.
    
    Creates side-by-side images showing:
      - Original image
      - GT mask overlay
      - Clean prediction overlay
      - Noisy L4 prediction overlay
      - Performance metrics
    
    Args:
        df: Results DataFrame
        cfg: Config dictionary
        out_dir: Output directory
        top_k: Number of failure cases
        metric: Metric for ranking
        
    Returns:
        List of exported image paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    failures = identify_top_failures(df, top_k=top_k, metric=metric)
    if len(failures) == 0:
        return []
    
    exp_name = cfg["exp"]["name"]
    pred_root = Path(cfg["exp"].get("out_root", "outputs")) / exp_name / "pred_masks"
    
    exported = []
    
    for idx, row in failures.iterrows():
        sid = row["id"]
        dataset = row["dataset"]
        model = row["model"]
        weight = row["weight"]
        mode = row["mode"]
        noise = row["noise"]
        
        img_path = row.get("img_path", "")
        mask_path = row.get("mask_path", "")
        
        if not img_path or not Path(img_path).exists():
            continue
        if not mask_path or not Path(mask_path).exists():
            continue
        
        # Load image and GT
        img = _safe_read_gray(img_path)
        gt = _safe_read_mask(mask_path)
        
        # Load predictions
        pred_l0_path = pred_root / dataset / model / weight / mode / "P0" / "clean" / "L0" / f"{sid}.png"
        pred_l4_path = pred_root / dataset / model / weight / mode / "P1" / noise / "L4" / f"{sid}.png"
        
        pred_l0 = None
        pred_l4 = None
        
        if pred_l0_path.exists():
            pred_l0 = _safe_read_mask(str(pred_l0_path))
        if pred_l4_path.exists():
            pred_l4 = _safe_read_mask(str(pred_l4_path))
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # GT overlay
        axes[1].imshow(overlay(img, gt, alpha=0.4, color=(0, 255, 0)))
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Clean prediction
        if pred_l0 is not None:
            axes[2].imshow(overlay(img, pred_l0, alpha=0.4, color=(0, 0, 255)))
            axes[2].set_title(f"Clean (L0): {metric}={row[f'{metric}_L0']:.3f}")
        else:
            axes[2].imshow(img, cmap="gray")
            axes[2].set_title("Clean pred not found")
        axes[2].axis("off")
        
        # Noisy L4 prediction
        if pred_l4 is not None:
            axes[3].imshow(overlay(img, pred_l4, alpha=0.4, color=(255, 0, 0)))
            axes[3].set_title(f"Noisy L4 ({noise}): {metric}={row[f'{metric}_L4']:.3f}")
        else:
            axes[3].imshow(img, cmap="gray")
            axes[3].set_title("L4 pred not found")
        axes[3].axis("off")
        
        # Title with drop info
        drop_val = row["drop"]
        drop_rel = row["drop_rel"]
        plt.suptitle(
            f"Failure Case: {sid}\n"
            f"Model: {model}/{weight} | Mode: {mode} | Noise: {noise}\n"
            f"Drop: {drop_val:.3f} ({drop_rel:.1f}%)",
            fontsize=10
        )
        
        plt.tight_layout()
        
        # Save
        out_path = out_dir / f"failure_{idx:02d}_{sid}_{noise}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        exported.append(str(out_path))
    
    return exported


def create_failure_summary(
    df: pd.DataFrame,
    out_path: Path,
    top_k: int = 20
) -> None:
    """
    Create CSV summary of top failure cases.
    
    Args:
        df: Results DataFrame
        out_path: Output CSV path
        top_k: Number of cases to include
    """
    failures = identify_top_failures(df, top_k=top_k, metric="dice")
    if len(failures) > 0:
        failures.to_csv(out_path, index=False)


def analyze_failure_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in failure cases.
    
    Returns:
        Dict with analysis results:
          - worst_noise: Noise type causing most failures
          - worst_model: Model with most failures
          - avg_drop_by_noise: Average drop per noise type
          - avg_drop_by_model: Average drop per model
    """
    failures = identify_top_failures(df, top_k=100, metric="dice")
    
    if len(failures) == 0:
        return {}
    
    result = {}
    
    # Count failures by noise type
    noise_counts = failures["noise"].value_counts()
    result["failure_counts_by_noise"] = noise_counts.to_dict()
    result["worst_noise"] = noise_counts.idxmax() if len(noise_counts) > 0 else None
    
    # Count failures by model
    model_counts = failures["model"].value_counts()
    result["failure_counts_by_model"] = model_counts.to_dict()
    result["worst_model"] = model_counts.idxmax() if len(model_counts) > 0 else None
    
    # Average drop by noise
    avg_by_noise = failures.groupby("noise")["drop"].mean()
    result["avg_drop_by_noise"] = avg_by_noise.to_dict()
    
    # Average drop by model
    avg_by_model = failures.groupby("model")["drop"].mean()
    result["avg_drop_by_model"] = avg_by_model.to_dict()
    
    return result
