"""
Failure case analysis and export.
Identifies and visualizes samples with largest performance drops.
Now uses robust path resolution with fallback strategies.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import logging

from viz.overlays import overlay
from viz.path_resolver import (
    resolve_pred_path, get_pred_root,
    resolve_noisy_image_path, get_noisy_root
)

# Configure logging
logger = logging.getLogger(__name__)


def _safe_read_gray(path: str) -> Optional[np.ndarray]:
    """Read image as grayscale with error handling."""
    try:
        return np.asarray(Image.open(path).convert("L")).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read image: {path} - {e}")
        return None


def _safe_read_mask(path: str) -> Optional[np.ndarray]:
    """Read mask as binary with error handling."""
    try:
        m = np.asarray(Image.open(path).convert("L")).astype(np.uint8)
        return (m > 0).astype(np.uint8)
    except Exception as e:
        logger.warning(f"Failed to read mask: {path} - {e}")
        return None


def _resolve_image_for_level(
    cfg: dict,
    dataset: str,
    noise: str,
    level: str,
    sid: str,
    noise_seed: int,
    fallback_img_path: str,
    log_debug: bool = False
) -> Optional[np.ndarray]:
    """
    Resolve the correct image (clean or noisy) for a given level.
    
    Args:
        cfg: Configuration dictionary
        dataset: Dataset name
        noise: Noise type (clean, gaussian, etc.)
        level: Level (L0, L1, L2, L3, L4)
        sid: Sample ID
        noise_seed: Noise seed
        fallback_img_path: Original image path to use if noisy image not found
        log_debug: Whether to log debug info
        
    Returns:
        Image array (uint8 grayscale HxW) or None
    """
    noisy_root = get_noisy_root(cfg)
    
    # For P0/clean/L0, noise is "clean"
    noise_for_lookup = "clean" if level == "L0" or noise == "clean" else noise
    
    noisy_result = resolve_noisy_image_path(
        noisy_root=noisy_root,
        dataset=dataset,
        noise=noise_for_lookup,
        level=str(level),
        sid=str(sid),
        noise_seed=noise_seed,
        log_debug=log_debug
    )
    
    if noisy_result.found:
        img = _safe_read_gray(str(noisy_result.path))
        if img is not None:
            return img
    
    # Fallback to original image
    if fallback_img_path and Path(fallback_img_path).exists():
        img = _safe_read_gray(fallback_img_path)
        if img is not None:
            if log_debug and level != "L0":
                logger.warning(
                    f"[{sid}/{level}] Noisy image not found, using original. "
                    f"Tried: {noisy_result.search_attempts[0] if noisy_result.search_attempts else 'N/A'}"
                )
            return img
    
    return None


def identify_top_failures(
    df: pd.DataFrame,
    top_k: int = 10,
    metric: str = "dice",
    mode: str = None
) -> pd.DataFrame:
    """
    Identify samples with largest performance drop from L0 to L4.
    
    Args:
        df: Results DataFrame
        top_k: Number of top failures to return
        metric: Metric to measure drop
        mode: Optional filter by mode (e.g., 'automatic', 'prompt_bbox')
        
    Returns:
        DataFrame with failure cases
    """
    # Filter by mode if specified
    df_filtered = df.copy()
    if mode is not None:
        df_filtered = df_filtered[df_filtered["mode"] == mode]
    
    # Get clean baseline (P0)
    base = df_filtered[df_filtered["protocol"] == "P0"].copy()
    if len(base) == 0:
        return pd.DataFrame()
    
    base = base.rename(columns={metric: f"{metric}_L0"})
    base = base[["dataset", "model", "weight", "mode", "id", f"{metric}_L0", "img_path", "mask_path"]]
    
    # Get worst case (L4) from P1
    worst = df_filtered[(df_filtered["protocol"] == "P1") & (df_filtered["level"] == "L4")].copy()
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
    Export visualization of top failure cases for each mode separately.
    
    Creates side-by-side images showing:
      - Original image
      - GT mask overlay
      - Clean prediction overlay
      - Noisy L4 prediction overlay
      - Performance metrics
    
    Exports separate folders and visualizations for:
      - automatic mode
      - prompt_bbox mode
      - combined (all modes)
    
    Args:
        df: Results DataFrame
        cfg: Config dictionary
        out_dir: Output directory
        top_k: Number of failure cases per mode
        metric: Metric for ranking
        
    Returns:
        List of exported image paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_root = get_pred_root(cfg)
    noise_seed = cfg.get("noise_config", {}).get("base_seed", 42)
    
    # Get available modes
    available_modes = df["mode"].unique().tolist()
    logger.info(f"Available modes: {available_modes}")
    
    all_exported = []
    
    # Export for each mode separately
    for mode in available_modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        failures = identify_top_failures(df, top_k=top_k, metric=metric, mode=mode)
        if len(failures) == 0:
            logger.warning(f"No failures found for mode: {mode}")
            continue
        
        logger.info(f"Exporting {len(failures)} failure cases for mode: {mode}")
        
        exported = _export_failure_list(
            failures=failures,
            cfg=cfg,
            out_dir=mode_dir,
            pred_root=pred_root,
            noise_seed=noise_seed,
            metric=metric,
            prefix=f"{mode}_"
        )
        all_exported.extend(exported)
    
    # Also export combined (all modes)
    combined_dir = out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    failures_all = identify_top_failures(df, top_k=top_k, metric=metric, mode=None)
    if len(failures_all) > 0:
        logger.info(f"Exporting {len(failures_all)} combined failure cases")
        exported_combined = _export_failure_list(
            failures=failures_all,
            cfg=cfg,
            out_dir=combined_dir,
            pred_root=pred_root,
            noise_seed=noise_seed,
            metric=metric,
            prefix="combined_"
        )
        all_exported.extend(exported_combined)
    
    # Create summary markdown
    _create_failure_summary_markdown(df, out_dir, top_k, metric)
    
    logger.info(f"Total exported: {len(all_exported)} failure case visualizations")
    return all_exported


def _export_failure_list(
    failures: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    pred_root: Path,
    noise_seed: int,
    metric: str,
    prefix: str = ""
) -> List[str]:
    """
    Export a list of failure cases to visualizations.
    
    Args:
        failures: DataFrame of failure cases
        cfg: Config dictionary
        out_dir: Output directory
        pred_root: Prediction root directory
        noise_seed: Noise seed
        metric: Metric name
        prefix: Filename prefix
        
    Returns:
        List of exported paths
    """
    exported = []
    not_found_count = 0
    
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
            logger.warning(f"[{sid}] Image not found: {img_path}")
            continue
        if not mask_path or not Path(mask_path).exists():
            logger.warning(f"[{sid}] Mask not found: {mask_path}")
            continue
        
        # Load original image and GT
        img = _safe_read_gray(img_path)
        gt = _safe_read_mask(mask_path)
        if img is None or gt is None:
            continue
        
        # Load noisy image for L4 level (uses saved noisy images)
        img_l4 = _resolve_image_for_level(cfg, dataset, noise, "L4", sid, noise_seed, img_path)
        if img_l4 is None:
            img_l4 = img  # Fallback to original
        
        # Load predictions using robust path resolution
        pred_l0_result = resolve_pred_path(
            pred_root=pred_root,
            dataset=dataset,
            model=model,
            weight=weight,
            mode=mode,
            protocol="P0",
            noise="clean",
            level="L0",
            sid=str(sid),
            noise_seed=noise_seed,
            log_debug=False
        )
        
        pred_l4_result = resolve_pred_path(
            pred_root=pred_root,
            dataset=dataset,
            model=model,
            weight=weight,
            mode=mode,
            protocol="P1",
            noise=noise,
            level="L4",
            sid=str(sid),
            noise_seed=noise_seed,
            log_debug=False
        )
        
        pred_l0 = None
        pred_l4 = None
        
        if pred_l0_result.found:
            pred_l0 = _safe_read_mask(str(pred_l0_result.path))
        else:
            not_found_count += 1
            logger.warning(f"[{sid}] L0 pred not found. Searched: {pred_l0_result.search_attempts[0] if pred_l0_result.search_attempts else 'N/A'}")
            
        if pred_l4_result.found:
            pred_l4 = _safe_read_mask(str(pred_l4_result.path))
        else:
            not_found_count += 1
            logger.warning(f"[{sid}] L4 pred not found ({noise}). Searched: {pred_l4_result.search_attempts[0] if pred_l4_result.search_attempts else 'N/A'}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original clean image
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original Image (Clean)")
        axes[0].axis("off")
        
        # GT overlay
        axes[1].imshow(overlay(img, gt, alpha=0.4, color=(0, 255, 0)))
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Clean prediction on clean image
        if pred_l0 is not None:
            axes[2].imshow(overlay(img, pred_l0, alpha=0.4, color=(0, 0, 255)))
            axes[2].set_title(f"Clean (L0): {metric}={row[f'{metric}_L0']:.3f}")
        else:
            axes[2].imshow(img, cmap="gray")
            axes[2].set_title("Clean pred not found")
        axes[2].axis("off")
        
        # Noisy L4 prediction on actual noisy image
        if pred_l4 is not None:
            axes[3].imshow(overlay(img_l4, pred_l4, alpha=0.4, color=(255, 0, 0)))
            axes[3].set_title(f"Noisy L4 ({noise}): {metric}={row[f'{metric}_L4']:.3f}")
        else:
            axes[3].imshow(img_l4, cmap="gray")
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
        
        # Save as PNG and PDF
        out_path_png = out_dir / f"{prefix}failure_{idx:03d}_{sid}_{noise}.png"
        out_path_pdf = out_dir / f"{prefix}failure_{idx:03d}_{sid}_{noise}.pdf"
        
        plt.savefig(out_path_png, dpi=150, bbox_inches="tight", facecolor='white')
        plt.savefig(out_path_pdf, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close()
        
        exported.append(str(out_path_png))
    
    if not_found_count > 0:
        logger.warning(f"Total predictions not found: {not_found_count}")
    
    logger.info(f"Exported {len(exported)} failure case visualizations")
    return exported


def _create_failure_summary_markdown(
    df: pd.DataFrame,
    out_dir: Path,
    top_k: int,
    metric: str
) -> None:
    """
    Create a markdown summary of failure cases for each mode.
    
    Args:
        df: Results DataFrame
        out_dir: Output directory
        top_k: Number of top failures
        metric: Metric name
    """
    summary_path = out_dir / "failure_summary.md"
    
    available_modes = df["mode"].unique().tolist()
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Failure Case Analysis Summary\n\n")
        f.write(f"**Metric**: {metric}  \n")
        f.write(f"**Top K per mode**: {top_k}\n\n")
        
        # Summary by mode
        for mode in available_modes:
            f.write(f"\n## Mode: {mode}\n\n")
            
            failures = identify_top_failures(df, top_k=top_k, metric=metric, mode=mode)
            if len(failures) == 0:
                f.write("No failure cases found.\n")
                continue
            
            # Statistics
            avg_drop = failures["drop"].mean()
            max_drop = failures["drop"].max()
            worst_noise = failures.groupby("noise")["drop"].mean().idxmax()
            
            f.write(f"- **Average Drop**: {avg_drop:.4f}  \n")
            f.write(f"- **Max Drop**: {max_drop:.4f}  \n")
            f.write(f"- **Worst Noise Type**: {worst_noise}  \n\n")
            
            # Table header
            f.write("| Rank | Sample ID | Noise | Dice L0 | Dice L4 | Drop | Drop % |\n")
            f.write("|------|-----------|-------|---------|---------|------|--------|\n")
            
            for rank, (_, row) in enumerate(failures.iterrows(), 1):
                f.write(
                    f"| {rank} | {row['id']} | {row['noise']} | "
                    f"{row[f'{metric}_L0']:.3f} | {row[f'{metric}_L4']:.3f} | "
                    f"{row['drop']:.3f} | {row['drop_rel']:.1f}% |\n"
                )
            
            f.write("\n")
        
        # Noise type analysis
        f.write("\n## Noise Type Analysis (All Modes)\n\n")
        
        all_failures = identify_top_failures(df, top_k=100, metric=metric, mode=None)
        if len(all_failures) > 0:
            noise_stats = all_failures.groupby("noise").agg({
                "drop": ["mean", "std", "count"]
            }).round(4)
            noise_stats.columns = ["Mean Drop", "Std Drop", "Count"]
            noise_stats = noise_stats.sort_values("Mean Drop", ascending=False)
            
            f.write("| Noise Type | Mean Drop | Std Drop | Count |\n")
            f.write("|------------|-----------|----------|-------|\n")
            
            for noise_type, row in noise_stats.iterrows():
                f.write(
                    f"| {noise_type} | {row['Mean Drop']:.4f} | "
                    f"{row['Std Drop']:.4f} | {int(row['Count'])} |\n"
                )
        
        f.write("\n---\n")
        f.write("*Generated by NoisySAM Benchmark*\n")
    
    logger.info(f"Created failure summary: {summary_path}")


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
    Analyze patterns in failure cases, including breakdown by mode.
    
    Returns:
        Dict with analysis results:
          - worst_noise: Noise type causing most failures
          - worst_model: Model with most failures
          - avg_drop_by_noise: Average drop per noise type
          - avg_drop_by_model: Average drop per model
          - by_mode: Breakdown of failure patterns per mode
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
    
    # Analysis by mode
    result["by_mode"] = {}
    for mode in failures["mode"].unique():
        mode_failures = failures[failures["mode"] == mode]
        mode_result = {
            "count": len(mode_failures),
            "avg_drop": float(mode_failures["drop"].mean()),
            "max_drop": float(mode_failures["drop"].max()),
            "worst_noise": mode_failures.groupby("noise")["drop"].mean().idxmax() if len(mode_failures) > 0 else None,
            "avg_drop_by_noise": mode_failures.groupby("noise")["drop"].mean().to_dict()
        }
        result["by_mode"][mode] = mode_result
    
    return result
