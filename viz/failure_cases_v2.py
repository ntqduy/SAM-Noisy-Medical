"""
Failure case analysis and export - Enhanced for NoisySAM benchmark.
Identifies and visualizes samples with largest performance drops.

Features:
- Robust path resolution with fallback strategies
- Multi-level visualization (L0..L4)
- Per-mode and per-noise breakdown
- Detailed debug logging
- PDF export with proper overlays
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
import warnings

from viz.overlays import overlay
from viz.path_resolver import (
    resolve_pred_path, 
    get_pred_root,
    PathResolutionResult,
    validate_paths_in_df,
    format_path_validation_report
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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


def identify_top_failures(
    df: pd.DataFrame,
    top_k: int = 10,
    metric: str = "dice",
    per_mode: bool = False,
    per_noise: bool = False
) -> pd.DataFrame:
    """
    Identify samples with largest performance drop from L0 to L4.
    
    Args:
        df: Results DataFrame
        top_k: Number of top failures to return
        metric: Metric to measure drop
        per_mode: If True, return top_k per mode
        per_noise: If True, return top_k per noise type
        
    Returns:
        DataFrame with failure cases
    """
    # Get clean baseline (P0)
    base = df[df["protocol"] == "P0"].copy()
    if len(base) == 0:
        logger.warning("No P0 (clean baseline) rows found in DataFrame")
        return pd.DataFrame()
    
    base = base.rename(columns={metric: f"{metric}_L0"})
    base_cols = ["dataset", "model", "weight", "mode", "id", f"{metric}_L0", "img_path", "mask_path"]
    base = base[[c for c in base_cols if c in base.columns]]
    
    # Get worst case (L4) from P1
    worst = df[(df["protocol"] == "P1") & (df["level"] == "L4")].copy()
    if len(worst) == 0:
        logger.warning("No P1/L4 rows found in DataFrame")
        return pd.DataFrame()
    
    worst = worst.rename(columns={metric: f"{metric}_L4"})
    keep_cols = ["dataset", "model", "weight", "mode", "noise", "id", f"{metric}_L4"]
    worst = worst[[c for c in keep_cols if c in worst.columns]]
    
    # Merge
    merge_keys = ["dataset", "model", "weight", "mode", "id"]
    merged = worst.merge(base, on=merge_keys, how="left")
    
    # Compute drop
    merged["drop"] = merged[f"{metric}_L0"] - merged[f"{metric}_L4"]
    merged["drop_rel"] = merged["drop"] / merged[f"{metric}_L0"].clip(lower=1e-6) * 100
    
    # Get top failures
    if per_mode and per_noise:
        # Top K per (mode, noise) combination
        failures = merged.groupby(["mode", "noise"], group_keys=False).apply(
            lambda g: g.nlargest(top_k, "drop")
        ).reset_index(drop=True)
    elif per_mode:
        failures = merged.groupby("mode", group_keys=False).apply(
            lambda g: g.nlargest(top_k, "drop")
        ).reset_index(drop=True)
    elif per_noise:
        failures = merged.groupby("noise", group_keys=False).apply(
            lambda g: g.nlargest(top_k, "drop")
        ).reset_index(drop=True)
    else:
        failures = merged.nlargest(top_k, "drop")
    
    return failures


def _create_missing_panel(ax, title: str, searched_path: str = ""):
    """Create a panel indicating missing data."""
    ax.imshow(np.ones((100, 100, 3), dtype=np.uint8) * 200)  # Gray placeholder
    ax.set_title(f"{title}\n[MISSING]", fontsize=9, color="red")
    if searched_path:
        ax.text(0.5, 0.5, f"Path not found:\n{Path(searched_path).name}", 
                ha='center', va='center', fontsize=6, wrap=True,
                transform=ax.transAxes)
    ax.axis("off")


def export_failure_cases(
    df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    top_k: int = 8,
    metric: str = "dice",
    noise_seed: int = 42,
    log_debug: bool = True
) -> List[str]:
    """
    Export visualization of top failure cases with robust path resolution.
    
    Creates side-by-side images showing:
      - Original image
      - GT mask overlay
      - Clean prediction overlay (L0)
      - Noisy L4 prediction overlay
      - Performance metrics
    
    Args:
        df: Results DataFrame
        cfg: Config dictionary
        out_dir: Output directory
        top_k: Number of failure cases
        metric: Metric for ranking
        noise_seed: Noise seed for path resolution
        log_debug: Enable debug logging
        
    Returns:
        List of exported image paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    failures = identify_top_failures(df, top_k=top_k, metric=metric)
    if len(failures) == 0:
        logger.warning("No failure cases identified")
        return []
    
    pred_root = get_pred_root(cfg)
    logger.info(f"Prediction root: {pred_root}")
    logger.info(f"Exporting {len(failures)} failure cases to {out_dir}")
    
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
        
        # Validate source paths
        if not img_path or not Path(img_path).exists():
            logger.warning(f"[{sid}] Image not found: {img_path}")
            continue
        if not mask_path or not Path(mask_path).exists():
            logger.warning(f"[{sid}] Mask not found: {mask_path}")
            continue
        
        # Load image and GT
        img = _safe_read_gray(img_path)
        gt = _safe_read_mask(mask_path)
        if img is None or gt is None:
            continue
        
        # Resolve prediction paths using robust resolver
        pred_l0_result = resolve_pred_path(
            pred_root=pred_root,
            dataset=dataset,
            model=model,
            weight=weight,
            mode=mode,
            protocol="P0",
            noise="clean",
            level="L0",
            sid=sid,
            noise_seed=noise_seed,
            log_debug=log_debug
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
            sid=sid,
            noise_seed=noise_seed,
            log_debug=log_debug
        )
        
        # Load predictions
        pred_l0 = None
        pred_l4 = None
        
        if pred_l0_result.found:
            pred_l0 = _safe_read_mask(str(pred_l0_result.path))
        else:
            not_found_count += 1
            logger.warning(f"[{sid}] L0 pred not found. Searched: {pred_l0_result.search_attempts[0]}")
            
        if pred_l4_result.found:
            pred_l4 = _safe_read_mask(str(pred_l4_result.path))
        else:
            not_found_count += 1
            logger.warning(f"[{sid}] L4 pred not found ({noise}). Searched: {pred_l4_result.search_attempts[0]}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # GT overlay
        axes[1].imshow(overlay(img, gt, alpha=0.4, color=(0, 255, 0)))
        axes[1].set_title("Ground Truth (green)")
        axes[1].axis("off")
        
        # Clean prediction (L0)
        if pred_l0 is not None:
            axes[2].imshow(overlay(img, pred_l0, alpha=0.4, color=(0, 0, 255)))
            axes[2].set_title(f"Clean (L0): {metric}={row[f'{metric}_L0']:.3f}")
        else:
            _create_missing_panel(axes[2], f"Clean (L0)", 
                                 pred_l0_result.search_attempts[0] if pred_l0_result.search_attempts else "")
        axes[2].axis("off")
        
        # Noisy L4 prediction
        if pred_l4 is not None:
            axes[3].imshow(overlay(img, pred_l4, alpha=0.4, color=(255, 0, 0)))
            axes[3].set_title(f"Noisy L4 ({noise}): {metric}={row[f'{metric}_L4']:.3f}")
        else:
            _create_missing_panel(axes[3], f"Noisy L4 ({noise})",
                                 pred_l4_result.search_attempts[0] if pred_l4_result.search_attempts else "")
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
        out_path_png = out_dir / f"failure_{idx:03d}_{sid}_{noise}.png"
        out_path_pdf = out_dir / f"failure_{idx:03d}_{sid}_{noise}.pdf"
        
        plt.savefig(out_path_png, dpi=150, bbox_inches="tight", facecolor='white')
        plt.savefig(out_path_pdf, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close()
        
        exported.append(str(out_path_png))
    
    if not_found_count > 0:
        logger.warning(f"Total predictions not found: {not_found_count}")
    
    logger.info(f"Exported {len(exported)} failure case visualizations")
    return exported


def export_failure_cases_multilevel(
    df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    top_k_fail: int = 5,
    levels: List[str] = None,
    per_mode: bool = True,
    per_noise: bool = True,
    metric: str = "dice",
    noise_seed: int = 42
) -> List[str]:
    """
    Export multi-level failure case visualizations.
    
    For each failure case, shows prediction at all noise levels (L0..L4).
    
    Layout per case:
        Row: levels (L0, L1, L2, L3, L4)
        Columns: [Noisy Image (if available), GT overlay, Pred overlay]
    
    Args:
        df: Results DataFrame
        cfg: Config dictionary
        out_dir: Output directory
        top_k_fail: Number of top failures per group
        levels: List of levels to show (default: L0..L4)
        per_mode: Separate failures by mode
        per_noise: Separate failures by noise type
        metric: Metric for ranking
        noise_seed: Noise seed
        
    Returns:
        List of exported PDF paths
    """
    if levels is None:
        # Get levels from config or default
        levels_cfg = cfg.get("levels", {}).get("names", ["L0", "L1", "L2", "L3", "L4"])
        levels = levels_cfg
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_root = get_pred_root(cfg)
    exported = []
    
    # Get failure cases
    failures = identify_top_failures(
        df, 
        top_k=top_k_fail, 
        metric=metric,
        per_mode=per_mode,
        per_noise=per_noise
    )
    
    if len(failures) == 0:
        logger.warning("No failure cases identified for multi-level export")
        return []
    
    logger.info(f"Exporting {len(failures)} multi-level failure cases")
    
    # Create combined PDF
    combined_pdf_path = out_dir / "failure_cases_multilevel.pdf"
    
    with PdfPages(combined_pdf_path) as pdf:
        for idx, row in failures.iterrows():
            sid = str(row["id"])
            dataset = row["dataset"]
            model = row["model"]
            weight = row["weight"]
            mode = row["mode"]
            noise = row["noise"]
            
            img_path = row.get("img_path", "")
            mask_path = row.get("mask_path", "")
            
            # Load base image and GT
            if not Path(img_path).exists():
                continue
            if not Path(mask_path).exists():
                continue
                
            img = _safe_read_gray(img_path)
            gt = _safe_read_mask(mask_path)
            if img is None or gt is None:
                continue
            
            # Create figure: rows = levels, cols = [Image, GT+Pred overlay, Pred only]
            n_levels = len(levels)
            fig, axes = plt.subplots(n_levels, 3, figsize=(12, 3 * n_levels))
            if n_levels == 1:
                axes = axes.reshape(1, -1)
            
            for i, lv in enumerate(levels):
                # Determine protocol and noise for this level
                if lv == "L0":
                    protocol = "P0"
                    noise_name = "clean"
                else:
                    protocol = "P1"
                    noise_name = noise
                
                # Get metrics for this level from df
                level_rows = df[
                    (df["id"] == sid) & 
                    (df["protocol"] == protocol) & 
                    (df["level"] == lv) &
                    (df["noise"] == noise_name) &
                    (df["mode"] == mode) &
                    (df["model"] == model) &
                    (df["weight"] == weight)
                ]
                
                dice_val = level_rows.iloc[0][metric] if len(level_rows) > 0 else None
                
                # Resolve prediction path
                pred_result = resolve_pred_path(
                    pred_root=pred_root,
                    dataset=dataset,
                    model=model,
                    weight=weight,
                    mode=mode,
                    protocol=protocol,
                    noise=noise_name,
                    level=lv,
                    sid=sid,
                    noise_seed=noise_seed,
                    log_debug=False
                )
                
                # Column 0: Original image (same for all levels since we don't store noisy)
                axes[i, 0].imshow(img, cmap="gray")
                axes[i, 0].set_title(f"{lv}: Image", fontsize=10)
                axes[i, 0].set_ylabel(lv, fontsize=12, fontweight='bold')
                axes[i, 0].axis("off")
                
                # Column 1: GT overlay (green)
                axes[i, 1].imshow(overlay(img, gt, alpha=0.4, color=(0, 255, 0)))
                axes[i, 1].set_title(f"{lv}: GT Overlay", fontsize=10)
                axes[i, 1].axis("off")
                
                # Column 2: Prediction overlay (red/blue depending on level)
                if pred_result.found:
                    pred = _safe_read_mask(str(pred_result.path))
                    if pred is not None:
                        color = (0, 0, 255) if lv == "L0" else (255, 100, 0)
                        axes[i, 2].imshow(overlay(img, pred, alpha=0.4, color=color))
                        dice_str = f"{dice_val:.3f}" if dice_val else "N/A"
                        axes[i, 2].set_title(f"{lv}: Pred ({metric}={dice_str})", fontsize=10)
                    else:
                        _create_missing_panel(axes[i, 2], f"{lv}: Pred", str(pred_result.path))
                else:
                    _create_missing_panel(axes[i, 2], f"{lv}: Pred", 
                                         pred_result.search_attempts[0] if pred_result.search_attempts else "")
                axes[i, 2].axis("off")
            
            # Overall title
            drop_val = row.get("drop", 0)
            drop_rel = row.get("drop_rel", 0)
            plt.suptitle(
                f"Multi-Level Failure: {sid}\n"
                f"Model: {model}/{weight} | Mode: {mode} | Noise: {noise}\n"
                f"Drop (L0→L4): {drop_val:.3f} ({drop_rel:.1f}%)",
                fontsize=12, fontweight='bold'
            )
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Also save individual files
            individual_path = out_dir / f"multilevel_{idx:03d}_{sid}_{mode}_{noise}.pdf"
            with PdfPages(individual_path) as ind_pdf:
                # Re-create figure for individual file
                fig2, axes2 = plt.subplots(n_levels, 3, figsize=(12, 3 * n_levels))
                if n_levels == 1:
                    axes2 = axes2.reshape(1, -1)
                    
                for i, lv in enumerate(levels):
                    if lv == "L0":
                        protocol = "P0"
                        noise_name = "clean"
                    else:
                        protocol = "P1"
                        noise_name = noise
                    
                    level_rows = df[
                        (df["id"] == sid) & 
                        (df["protocol"] == protocol) & 
                        (df["level"] == lv) &
                        (df["noise"] == noise_name) &
                        (df["mode"] == mode)
                    ]
                    dice_val = level_rows.iloc[0][metric] if len(level_rows) > 0 else None
                    
                    pred_result = resolve_pred_path(
                        pred_root=pred_root, dataset=dataset, model=model,
                        weight=weight, mode=mode, protocol=protocol,
                        noise=noise_name, level=lv, sid=sid,
                        noise_seed=noise_seed, log_debug=False
                    )
                    
                    axes2[i, 0].imshow(img, cmap="gray")
                    axes2[i, 0].set_title(f"{lv}: Image")
                    axes2[i, 0].axis("off")
                    
                    axes2[i, 1].imshow(overlay(img, gt, alpha=0.4, color=(0, 255, 0)))
                    axes2[i, 1].set_title(f"{lv}: GT")
                    axes2[i, 1].axis("off")
                    
                    if pred_result.found:
                        pred = _safe_read_mask(str(pred_result.path))
                        if pred is not None:
                            color = (0, 0, 255) if lv == "L0" else (255, 100, 0)
                            axes2[i, 2].imshow(overlay(img, pred, alpha=0.4, color=color))
                            dice_str = f"{dice_val:.3f}" if dice_val else "N/A"
                            axes2[i, 2].set_title(f"{lv}: Pred ({metric}={dice_str})")
                        else:
                            _create_missing_panel(axes2[i, 2], f"{lv}: Pred")
                    else:
                        _create_missing_panel(axes2[i, 2], f"{lv}: Pred [missing]")
                    axes2[i, 2].axis("off")
                
                plt.suptitle(
                    f"Multi-Level: {sid} | {model}/{weight} | {mode} | {noise}\n"
                    f"Drop: {drop_val:.3f} ({drop_rel:.1f}%)",
                    fontsize=11
                )
                plt.tight_layout()
                ind_pdf.savefig(fig2, bbox_inches='tight', facecolor='white')
                plt.close(fig2)
            
            exported.append(str(individual_path))
    
    exported.insert(0, str(combined_pdf_path))
    logger.info(f"Exported multi-level failure PDF: {combined_pdf_path}")
    return exported


def export_random_cases_multilevel(
    df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    random_k_per_mode: int = 3,
    random_k_per_noise: int = 2,
    levels: List[str] = None,
    noise_seed: int = 42,
    random_state: int = 42
) -> List[str]:
    """
    Export random sample cases for each mode and noise type.
    
    Args:
        df: Results DataFrame
        cfg: Config dictionary
        out_dir: Output directory
        random_k_per_mode: Number of random samples per mode
        random_k_per_noise: Number of random samples per noise type
        levels: Levels to show
        noise_seed: Noise seed
        random_state: Random state for reproducibility
        
    Returns:
        List of exported PDF paths
    """
    if levels is None:
        levels = cfg.get("levels", {}).get("names", ["L0", "L1", "L2", "L3", "L4"])
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_root = get_pred_root(cfg)
    exported = []
    
    # Get unique modes and noises
    modes = df["mode"].unique().tolist()
    noises = [n for n in df["noise"].unique() if n != "clean"]
    
    np.random.seed(random_state)
    
    combined_pdf_path = out_dir / "random_samples_multilevel.pdf"
    
    with PdfPages(combined_pdf_path) as pdf:
        # Random samples per mode
        for mode in modes:
            mode_df = df[(df["mode"] == mode) & (df["protocol"] == "P0")]
            sample_ids = mode_df["id"].unique()
            
            if len(sample_ids) == 0:
                continue
                
            selected = np.random.choice(
                sample_ids, 
                size=min(random_k_per_mode, len(sample_ids)),
                replace=False
            )
            
            for sid in selected:
                # Get base row
                base_row = mode_df[mode_df["id"] == sid].iloc[0]
                
                img_path = base_row.get("img_path", "")
                mask_path = base_row.get("mask_path", "")
                
                if not Path(img_path).exists() or not Path(mask_path).exists():
                    continue
                
                img = _safe_read_gray(img_path)
                gt = _safe_read_mask(mask_path)
                if img is None or gt is None:
                    continue
                
                dataset = base_row["dataset"]
                model = base_row["model"]
                weight = base_row["weight"]
                
                # Pick a random noise for visualization
                sample_noise = np.random.choice(noises) if noises else "gaussian"
                
                # Create figure
                n_levels = len(levels)
                fig, axes = plt.subplots(n_levels, 3, figsize=(12, 3 * n_levels))
                if n_levels == 1:
                    axes = axes.reshape(1, -1)
                
                for i, lv in enumerate(levels):
                    if lv == "L0":
                        protocol, noise_name = "P0", "clean"
                    else:
                        protocol, noise_name = "P1", sample_noise
                    
                    pred_result = resolve_pred_path(
                        pred_root=pred_root, dataset=dataset, model=model,
                        weight=weight, mode=mode, protocol=protocol,
                        noise=noise_name, level=lv, sid=str(sid),
                        noise_seed=noise_seed, log_debug=False
                    )
                    
                    axes[i, 0].imshow(img, cmap="gray")
                    axes[i, 0].set_title(f"{lv}: Image")
                    axes[i, 0].axis("off")
                    
                    axes[i, 1].imshow(overlay(img, gt, alpha=0.4, color=(0, 255, 0)))
                    axes[i, 1].set_title(f"{lv}: GT")
                    axes[i, 1].axis("off")
                    
                    if pred_result.found:
                        pred = _safe_read_mask(str(pred_result.path))
                        if pred is not None:
                            axes[i, 2].imshow(overlay(img, pred, alpha=0.4, color=(255, 100, 0)))
                            axes[i, 2].set_title(f"{lv}: Pred ({noise_name})")
                        else:
                            _create_missing_panel(axes[i, 2], f"{lv}: Pred")
                    else:
                        _create_missing_panel(axes[i, 2], f"{lv}: Pred [missing]")
                    axes[i, 2].axis("off")
                
                plt.suptitle(f"Random Sample: {sid} | Mode: {mode} | Noise: {sample_noise}")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight', facecolor='white')
                plt.close(fig)
    
    exported.append(str(combined_pdf_path))
    logger.info(f"Exported random samples PDF: {combined_pdf_path}")
    return exported


def create_failure_summary(
    df: pd.DataFrame,
    out_path: Path,
    top_k: int = 20,
    per_mode: bool = True,
    metric: str = "dice"
) -> None:
    """
    Create CSV summary of top failure cases with mode breakdown.
    
    Args:
        df: Results DataFrame
        out_path: Output CSV path
        top_k: Number of cases to include
        per_mode: Include breakdown by mode
        metric: Metric for ranking
    """
    failures = identify_top_failures(df, top_k=top_k, metric=metric, per_mode=per_mode)
    
    if len(failures) > 0:
        # Add additional columns for analysis
        failures = failures.sort_values("drop", ascending=False)
        failures.to_csv(out_path, index=False)
        logger.info(f"Created failure summary: {out_path} ({len(failures)} cases)")
    else:
        logger.warning("No failures to summarize")


def analyze_failure_patterns(
    df: pd.DataFrame,
    top_k: int = 100,
    metric: str = "dice"
) -> Dict[str, Any]:
    """
    Analyze patterns in failure cases with mode and noise breakdown.
    
    Returns:
        Dict with analysis results:
          - worst_noise: Noise type causing most failures
          - worst_model: Model with most failures
          - worst_mode: Mode with most failures  
          - avg_drop_by_noise: Average drop per noise type
          - avg_drop_by_model: Average drop per model
          - avg_drop_by_mode: Average drop per mode
    """
    failures = identify_top_failures(df, top_k=top_k, metric=metric)
    
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
    
    # Count failures by mode
    mode_counts = failures["mode"].value_counts()
    result["failure_counts_by_mode"] = mode_counts.to_dict()
    result["worst_mode"] = mode_counts.idxmax() if len(mode_counts) > 0 else None
    
    # Average drop by noise
    avg_by_noise = failures.groupby("noise")["drop"].mean()
    result["avg_drop_by_noise"] = avg_by_noise.to_dict()
    
    # Average drop by model
    avg_by_model = failures.groupby("model")["drop"].mean()
    result["avg_drop_by_model"] = avg_by_model.to_dict()
    
    # Average drop by mode
    avg_by_mode = failures.groupby("mode")["drop"].mean()
    result["avg_drop_by_mode"] = avg_by_mode.to_dict()
    
    # Cross-tabulation: mode x noise
    if "mode" in failures.columns and "noise" in failures.columns:
        crosstab = pd.crosstab(failures["mode"], failures["noise"], values=failures["drop"], aggfunc="mean")
        result["drop_by_mode_noise"] = crosstab.to_dict()
    
    # Summary stats
    result["summary"] = {
        "total_failures_analyzed": len(failures),
        "mean_drop": float(failures["drop"].mean()),
        "std_drop": float(failures["drop"].std()),
        "max_drop": float(failures["drop"].max()),
        "min_drop": float(failures["drop"].min())
    }
    
    return result
