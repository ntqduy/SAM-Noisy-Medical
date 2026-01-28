from pathlib import Path
from turtle import title
from typing import List, Optional, Dict, Any
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation safely, handling constant arrays.
    
    Returns:
        Correlation coefficient, or 0.0 if computation fails (e.g., zero stddev)
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    
    # Check for constant arrays (stddev = 0)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr) or np.isinf(corr):
                return 0.0
            return corr
        except Exception:
            return 0.0


def _get_intensity_scalars_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract intensity_scalars from config.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Dict mapping level names to intensity scalars
    """
    if cfg is None:
        return {"L0": 0.0, "L1": 0.25, "L2": 0.5, "L3": 0.75, "L4": 1.0}
    
    levels_cfg = cfg.get("levels", {}) or {}
    intensity_scalars = levels_cfg.get("intensity_scalars", {})
    
    if not intensity_scalars:
        # Fallback default
        return {"L0": 0.0, "L1": 0.25, "L2": 0.5, "L3": 0.75, "L4": 1.0}
    
    return intensity_scalars


def _add_intensity_scalars_to_df(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Add intensity_scalar column to DataFrame using config mapping.
    
    Args:
        df: Results DataFrame
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with intensity_scalar column
    """
    df = df.copy()
    intensity_scalars = _get_intensity_scalars_from_cfg(cfg)
    
    if "intensity_scalar" not in df.columns:
        df["intensity_scalar"] = df["level"].map(intensity_scalars).fillna(0.5)
    else:
        # Fill missing values using config
        mask = df["intensity_scalar"].isna()
        if mask.any():
            df.loc[mask, "intensity_scalar"] = df.loc[mask, "level"].map(intensity_scalars).fillna(0.5)
    
    return df


def plot_metric_vs_level(df: pd.DataFrame, out_dir: Path, protocols: List[str], metrics: List[str]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)

    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    sub = df[df["protocol"].isin(protocols)].copy()
    if len(sub) == 0:
        return paths

    # normalize level order for L0..L4 only
    level_order = ["L0", "L1", "L2", "L3", "L4"]

    for metric in metrics:
        if metric not in sub.columns:
            continue
        gcols = ["dataset", "model", "weight", "mode", "noise", "protocol"]
        agg = sub.groupby(gcols + ["level"])[metric].mean().reset_index()

        for (dataset, model, weight, mode, noise, protocol), g in agg.groupby(gcols):
            # keep only L0..L4
            g = g[g["level"].isin(level_order)].copy()
            if len(g) == 0:
                continue
            g["level"] = pd.Categorical(g["level"], categories=level_order, ordered=True)
            g = g.sort_values("level")

            plt.figure()
            plt.plot(g["level"].astype(str), g[metric].values, marker="o")
            # Set title with size 9 and padding 6
            plt.title(
                f"{metric} vs level — {dataset} | {model}/{weight} | {mode} | {protocol} | {noise}",
                fontsize=9,   # nhỏ hơn (thử 8–10)
                pad=6
            )
            plt.xlabel("Level")
            plt.ylabel(metric)
            out_png = out_dir_png / f"{metric}_vs_level_{dataset}_{model}-{weight}_{mode}_{protocol}_{noise}.png"
            out_pdf = out_dir_pdf / f"{metric}_vs_level_{dataset}_{model}-{weight}_{mode}_{protocol}_{noise}.pdf"
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.savefig(out_pdf, bbox_inches="tight")

            plt.close()
            paths.append(str(out_png))
            paths.append(str(out_pdf))
    return paths


def plot_metric_vs_level_by_mode(
    df: pd.DataFrame, 
    out_dir: Path, 
    metrics: List[str] = None,
    protocols: List[str] = None,
    cfg: Dict[str, Any] = None
) -> List[str]:
    """
    Plot metric vs level comparing different modes (automatic vs prompt_bbox).
    
    Generates 4 comparison plots:
      - Dice vs Level (automatic mode, all noises)
      - Dice vs Level (prompt_bbox mode, all noises)  
      - IoU vs Level (automatic mode, all noises)
      - IoU vs Level (prompt_bbox mode, all noises)
      
    Args:
        df: Results DataFrame
        out_dir: Output directory
        metrics: List of metrics to plot (default: ["dice", "iou"])
        protocols: Protocols to include (default: ["P1"])
        cfg: Configuration dictionary (for intensity_scalars)
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)
    
    paths = []
    
    if metrics is None:
        metrics = ["dice", "iou"]
    if protocols is None:
        protocols = ["P1"]
    
    # Add intensity_scalar from config
    df = _add_intensity_scalars_to_df(df, cfg)
    intensity_scalars = _get_intensity_scalars_from_cfg(cfg)
    
    # Get P1 data (L1-L4) and P0 data (L0 baseline)
    sub_p1 = df[df["protocol"].isin(protocols)].copy()
    sub_p0 = df[df["protocol"] == "P0"].copy()
    
    if len(sub_p1) == 0:
        return paths
    
    level_order = ["L0", "L1", "L2", "L3", "L4"]
    modes = sub_p1["mode"].unique()
    
    # Generate comparison plots for each mode separately
    for metric in metrics:
        if metric not in sub_p1.columns:
            continue
        
        for mode in modes:
            mode_data_p1 = sub_p1[sub_p1["mode"] == mode]
            mode_data_p0 = sub_p0[sub_p0["mode"] == mode]
            
            if len(mode_data_p1) == 0:
                continue
            
            # Group by dataset, model, weight
            for (dataset, model, weight), g in mode_data_p1.groupby(["dataset", "model", "weight"]):
                # Get corresponding P0 baseline for this group
                g_p0 = mode_data_p0[
                    (mode_data_p0["dataset"] == dataset) &
                    (mode_data_p0["model"] == model) &
                    (mode_data_p0["weight"] == weight)
                ]
                
                # Calculate L0 baseline metric (average across all samples)
                l0_metric_value = None
                if len(g_p0) > 0 and metric in g_p0.columns:
                    l0_metric_value = g_p0[metric].mean()
                
                # Plot all noise types on one chart
                plt.figure(figsize=(12, 7))
                
                noise_types = g["noise"].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, min(len(noise_types), 10)))
                
                for noise, color in zip(noise_types, colors):
                    noise_data = g[g["noise"] == noise]
                    agg = noise_data.groupby("level")[metric].mean().reset_index()
                    
                    # Keep only standard levels (L1-L4 from P1)
                    agg = agg[agg["level"].isin(level_order)].copy()
                    
                    # Add L0 baseline point if available
                    if l0_metric_value is not None:
                        l0_row = pd.DataFrame({"level": ["L0"], metric: [l0_metric_value]})
                        agg = pd.concat([l0_row, agg], ignore_index=True)
                    
                    if len(agg) == 0:
                        continue
                    
                    agg["level"] = pd.Categorical(agg["level"], categories=level_order, ordered=True)
                    agg = agg.sort_values("level")
                    
                    # Map level to intensity_scalar for x-axis
                    agg["intensity"] = agg["level"].map(intensity_scalars).fillna(0.5)
                    
                    plt.plot(agg["intensity"], agg[metric].values, 
                             marker="o", label=noise, color=color, linewidth=2, markersize=6)
                
                plt.xlabel("Intensity Scalar (from config)", fontsize=11)
                plt.ylabel(metric.upper(), fontsize=11)
                title = f"Compare {metric.upper()} vs Level ({mode})\n{dataset} | {model}/{weight}"
                plt.title(title, fontsize=10, pad=8)
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
                plt.grid(True, alpha=0.3)
                plt.xlim(-0.05, 1.05)
                plt.ylim(0, 1.05)
                
                # Add level labels on x-axis
                level_positions = [intensity_scalars.get(lv, 0) for lv in level_order]
                plt.xticks(level_positions, [f"{lv}\n({intensity_scalars.get(lv, 0):.2f})" for lv in level_order])
                
                out_png = out_dir_png / f"compare_{metric}_vs_level_{mode}_{dataset}_{model}-{weight}.png"
                out_pdf = out_dir_pdf / f"compare_{metric}_vs_level_{mode}_{dataset}_{model}-{weight}.pdf"
                plt.tight_layout()
                plt.savefig(out_png, dpi=160, bbox_inches='tight')
                plt.savefig(out_pdf, bbox_inches='tight')
                plt.close()
                paths.append(str(out_png))
                paths.append(str(out_pdf))
    
    return paths


def plot_mode_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "dice",
    protocols: List[str] = None,
    cfg: Dict[str, Any] = None
) -> List[str]:
    """
    Plot mode comparison (automatic vs prompt_bbox) for each noise type.
    
    Args:
        df: Results DataFrame
        out_dir: Output directory
        metric: Metric to compare (default: "dice")
        protocols: Protocols to include (default: ["P1"])
        cfg: Configuration dictionary
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)
    
    paths = []
    
    if protocols is None:
        protocols = ["P1"]
    
    df = _add_intensity_scalars_to_df(df, cfg)
    intensity_scalars = _get_intensity_scalars_from_cfg(cfg)
    
    # Get P1 data and P0 baseline
    sub = df[df["protocol"].isin(protocols)].copy()
    sub_p0 = df[df["protocol"] == "P0"].copy()
    
    if len(sub) == 0 or metric not in sub.columns:
        return paths
    
    level_order = ["L0", "L1", "L2", "L3", "L4"]
    modes = sorted(sub["mode"].unique())
    
    if len(modes) < 2:
        return paths  # Need at least 2 modes to compare
    
    # Plot comparison for each dataset, model, noise
    for (dataset, model, weight, noise), g in sub.groupby(["dataset", "model", "weight", "noise"]):
        plt.figure(figsize=(10, 6))
        
        mode_colors = {"automatic": "blue", "prompt_bbox": "red", "prompt_point": "green"}
        
        for mode in modes:
            mode_data = g[g["mode"] == mode]
            if len(mode_data) == 0:
                continue
            
            agg = mode_data.groupby("level")[metric].mean().reset_index()
            agg = agg[agg["level"].isin(level_order)].copy()
            
            # Add L0 baseline from P0
            p0_mode_data = sub_p0[
                (sub_p0["dataset"] == dataset) &
                (sub_p0["model"] == model) &
                (sub_p0["weight"] == weight) &
                (sub_p0["mode"] == mode)
            ]
            if len(p0_mode_data) > 0 and metric in p0_mode_data.columns:
                l0_value = p0_mode_data[metric].mean()
                l0_row = pd.DataFrame({"level": ["L0"], metric: [l0_value]})
                agg = pd.concat([l0_row, agg], ignore_index=True)
            
            if len(agg) == 0:
                continue
            
            agg["level"] = pd.Categorical(agg["level"], categories=level_order, ordered=True)
            agg = agg.sort_values("level")
            agg["intensity"] = agg["level"].map(intensity_scalars).fillna(0.5)
            
            color = mode_colors.get(mode, "gray")
            plt.plot(agg["intensity"], agg[metric].values, 
                     marker="o", label=mode, color=color, linewidth=2, markersize=8)
        
        plt.xlabel("Intensity Scalar", fontsize=11)
        plt.ylabel(metric.upper(), fontsize=11)
        plt.title(f"Mode Comparison: {metric.upper()} — {dataset} | {model}/{weight} | {noise}", fontsize=10)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1.05)
        
        level_positions = [intensity_scalars.get(lv, 0) for lv in level_order]
        plt.xticks(level_positions, level_order)
        
        out_png = out_dir_png / f"mode_comparison_{metric}_{dataset}_{model}-{weight}_{noise}.png"
        out_pdf = out_dir_pdf / f"mode_comparison_{metric}_{dataset}_{model}-{weight}_{noise}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_ofat_sensitivity(df: pd.DataFrame, out_dir: Path, metrics: List[str], protocols: List[str]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    sub = df[df["protocol"].isin(protocols)].copy()
    if len(sub) == 0:
        return paths

    level_order = ["L1", "L2", "L3", "L4"]

    for metric in metrics:
        if metric not in sub.columns:
            continue
        gcols = ["dataset", "model", "weight", "mode", "noise", "protocol"]
        agg = sub.groupby(gcols + ["level"])[metric].mean().reset_index()

        for (dataset, model, weight, mode, noise, protocol), g in agg.groupby(gcols):
            g = g[g["level"].isin(level_order)].copy()
            if len(g) == 0:
                continue
            g["level"] = pd.Categorical(g["level"], categories=level_order, ordered=True)
            g = g.sort_values("level")

            plt.figure()
            plt.plot(g["level"].astype(str), g[metric].values, marker="o")
            title = f"OFAT {protocol}: {metric} — {dataset}\n{model}/{weight} | {mode} | {noise}"
            plt.title(title, fontsize=9, pad=6)
            plt.xlabel("Level (sweep)")
            plt.ylabel(metric)
            out_png = out_dir_png / f"OFAT_{protocol}_{metric}_{dataset}_{model}-{weight}_{mode}_{noise}.png"
            out_pdf = out_dir_pdf / f"OFAT_{protocol}_{metric}_{dataset}_{model}-{weight}_{mode}_{noise}.pdf"
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.savefig(out_pdf, bbox_inches="tight")
            plt.close()
            paths.append(str(out_png))
            paths.append(str(out_pdf))
    return paths


def plot_grid_heatmap(df: pd.DataFrame, out_dir: Path) -> List[str]:
    """
    Plot P3 grid results as heatmaps (p × severity).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    sub = df[df["protocol"] == "P3"].copy()
    if len(sub) == 0:
        return paths

    # level looks like "L2_p0.5"
    def parse_level(s: str):
        try:
            a, b = s.split("_p")
            return a, float(b)
        except Exception:
            return None, None
    
    sub["severity"] = sub["level"].apply(lambda x: parse_level(x)[0])
    sub["prob"] = sub["level"].apply(lambda x: parse_level(x)[1])
    
    # Filter valid rows
    sub = sub[sub["severity"].notna() & sub["prob"].notna()]
    if len(sub) == 0:
        return paths
    
    gcols = ["dataset", "model", "weight", "mode", "noise"]
    
    for keys, g in sub.groupby(gcols):
        dataset, model, weight, mode, noise = keys
        
        # Pivot for dice
        pivot = g.pivot_table(index="prob", columns="severity", values="dice", aggfunc="mean")
        
        if pivot.empty:
            continue
        
        # Reorder columns
        col_order = [c for c in ["L1", "L2", "L3", "L4"] if c in pivot.columns]
        pivot = pivot[col_order]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="Dice")
        
        plt.xticks(range(len(col_order)), col_order)
        plt.yticks(range(len(pivot.index)), [f"{p:.1f}" for p in pivot.index])
        
        plt.xlabel("Severity Level")
        plt.ylabel("Probability p")
        plt.title(f"P3 Grid: Dice — {dataset} | {model}/{weight} | {mode} | {noise}")
        
        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(col_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        
        out_png = out_dir_png / f"P3_grid_{dataset}_{model}-{weight}_{mode}_{noise}.png"
        out_pdf = out_dir_pdf / f"P3_grid_{dataset}_{model}-{weight}_{mode}_{noise}.pdf"        
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_model_comparison(df: pd.DataFrame, out_dir: Path, metric: str = "dice") -> List[str]:
    """
    Plot model comparison across noise types and levels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    
    sub = df[df["protocol"] == "P1"].copy()
    if len(sub) == 0:
        return paths
    
    level_order = ["L1", "L2", "L3", "L4"]
    
    # Group by dataset and noise
    for (dataset, noise), g in sub.groupby(["dataset", "noise"]):
        # Aggregate by model
        agg = g.groupby(["model", "weight", "level"])[metric].mean().reset_index()
        agg["model_weight"] = agg["model"] + "/" + agg["weight"]
        
        # Pivot
        pivot = agg.pivot(index="level", columns="model_weight", values=metric)
        pivot = pivot.reindex([lv for lv in level_order if lv in pivot.index])
        
        if pivot.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        for col in pivot.columns:
            plt.plot(pivot.index.tolist(), pivot[col].values, marker="o", label=col)
        
        plt.xlabel("Noise Level")
        plt.ylabel(metric.capitalize())
        plt.title(f"Model Comparison: {metric} — {dataset} | {noise}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        
        out_png = out_dir_png / f"model_comparison_{metric}_{dataset}_{noise}.png"
        out_pdf = out_dir_pdf / f"model_comparison_{metric}_{dataset}_{noise}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_noise_comparison(df: pd.DataFrame, out_dir: Path, metric: str = "dice") -> List[str]:
    """
    Plot noise type comparison for each model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    
    sub = df[df["protocol"] == "P1"].copy()
    if len(sub) == 0:
        return paths
    
    level_order = ["L1", "L2", "L3", "L4"]
    
    # Group by dataset, model, weight
    for (dataset, model, weight), g in sub.groupby(["dataset", "model", "weight"]):
        # Aggregate by noise
        agg = g.groupby(["noise", "level"])[metric].mean().reset_index()
        
        # Pivot
        pivot = agg.pivot(index="level", columns="noise", values=metric)
        pivot = pivot.reindex([lv for lv in level_order if lv in pivot.index])
        
        if pivot.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        for col in pivot.columns:
            plt.plot(pivot.index.tolist(), pivot[col].values, marker="o", label=col)
        
        plt.xlabel("Noise Level")
        plt.ylabel(metric.capitalize())
        plt.title(f"Noise Comparison: {metric} — {dataset} | {model}/{weight}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        
        out_png = out_dir_png / f"noise_comparison_{metric}_{dataset}_{model}-{weight}.png"
        out_pdf = out_dir_pdf / f"noise_comparison_{metric}_{dataset}_{model}-{weight}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_stability_summary(stability_df: pd.DataFrame, out_dir: Path) -> List[str]:
    """
    Plot stability summary (perf_drop distribution).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    
    if "perf_drop_mean" not in stability_df.columns:
        return paths
    
    # Bar chart: perf_drop by noise
    for (dataset, model, weight), g in stability_df.groupby(["dataset", "model", "weight"]):
        if len(g) == 0:
            continue
        
        plt.figure(figsize=(10, 5))
        x = g["noise"].tolist()
        y = g["perf_drop_mean"].values
        yerr = g.get("perf_drop_std", pd.Series([0]*len(g))).values
        
        bars = plt.bar(range(len(x)), y, yerr=yerr, capsize=3)
        plt.xticks(range(len(x)), x, rotation=45, ha="right")
        plt.xlabel("Noise Type")
        plt.ylabel("Performance Drop (Dice)")
        plt.title(f"Stability: Perf Drop — {dataset} | {model}/{weight}")
        
        # Color bars by severity
        for bar, val in zip(bars, y):
            if val > 0.3:
                bar.set_color("red")
            elif val > 0.15:
                bar.set_color("orange")
            else:
                bar.set_color("green")
        
        out_png = out_dir_png / f"stability_perf_drop_{dataset}_{model}-{weight}.png"
        out_pdf = out_dir_pdf / f"stability_perf_drop_{dataset}_{model}-{weight}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_global_sensitivity(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "dice",
    protocols: List[str] = None,
    cfg: Dict[str, Any] = None
) -> List[str]:
    """
    Create global sensitivity analysis plots comparing all noise types.
    
    Generates:
      1. Sensitivity heatmap: noise types × levels
      2. Sensitivity curves: metric vs intensity_scalar (using config values)
      3. Ranking plots: noise types sorted by impact
    
    Args:
        df: Results DataFrame
        out_dir: Output directory
        metric: Metric to analyze (default: "dice")
        protocols: Protocols to include (default: ["P1"])
        cfg: Configuration dictionary (for intensity_scalars)
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)
    paths = []
    
    if protocols is None:
        protocols = ["P1"]
    
    # Add intensity_scalar from config
    df = _add_intensity_scalars_to_df(df, cfg)
    intensity_scalars = _get_intensity_scalars_from_cfg(cfg)
    
    sub = df[df["protocol"].isin(protocols)].copy()
    if len(sub) == 0 or metric not in sub.columns:
        return paths
    
    # Get baseline performance from P0
    baseline = df[df["protocol"] == "P0"].groupby(
        ["dataset", "model", "weight", "mode"]
    )[metric].mean().reset_index()
    baseline = baseline.rename(columns={metric: f"{metric}_baseline"})
    
    # Merge baseline
    sub = sub.merge(baseline, on=["dataset", "model", "weight", "mode"], how="left")
    
    # Calculate relative performance
    if f"{metric}_baseline" in sub.columns:
        sub[f"{metric}_rel"] = sub[metric] / sub[f"{metric}_baseline"].clip(lower=1e-6)
    
    # 1. Global sensitivity heatmap (noise × level)
    level_order = ["L1", "L2", "L3", "L4"]
    noise_types = sorted(sub["noise"].unique())
    
    for (dataset, model, weight, mode), g in sub.groupby(["dataset", "model", "weight", "mode"]):
        pivot = g.groupby(["noise", "level"])[metric].mean().unstack(fill_value=np.nan)
        
        # Reorder columns
        cols = [c for c in level_order if c in pivot.columns]
        if len(cols) == 0:
            continue
        pivot = pivot[cols]
        
        plt.figure(figsize=(10, max(6, len(pivot) * 0.4)))
        im = plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, label=metric.capitalize())
        
        plt.xticks(range(len(cols)), cols)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("Noise Level")
        plt.ylabel("Noise Type")
        plt.title(f"Global Sensitivity: {metric} — {dataset} | {model}/{weight} | {mode}")
        
        # Add annotations
        for i in range(len(pivot.index)):
            for j in range(len(cols)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = 'white' if val < 0.5 else 'black'
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
        
        out_png = out_dir_png / f"global_sensitivity_heatmap_{dataset}_{model}-{weight}_{mode}.png"
        out_pdf = out_dir_pdf / f"global_sensitivity_heatmap_{dataset}_{model}-{weight}_{mode}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    # 2. Sensitivity curves: metric vs intensity_scalar (from config)
    level_order_all = ["L0", "L1", "L2", "L3", "L4"]
    for (dataset, model, weight, mode), g in sub.groupby(["dataset", "model", "weight", "mode"]):
        plt.figure(figsize=(12, 6))
        
        for noise in g["noise"].unique():
            noise_data = g[g["noise"] == noise]
            # Group by level and map to intensity_scalar from config
            agg = noise_data.groupby("level")[metric].mean().reset_index()
            agg = agg[agg["level"].isin(level_order_all)].copy()
            
            if len(agg) > 1:
                # Map level to intensity scalar from config
                agg["intensity_scalar"] = agg["level"].map(intensity_scalars).fillna(0.5)
                agg = agg.sort_values("intensity_scalar")
                plt.plot(agg["intensity_scalar"], agg[metric], 
                        marker="o", label=noise, linewidth=2, markersize=6)
        
        plt.xlabel("Intensity Scalar (from config)", fontsize=11)
        plt.ylabel(metric.capitalize(), fontsize=11)
        plt.title(f"Sensitivity Curves: {metric} vs Intensity — {dataset} | {model}/{weight} | {mode}", fontsize=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1.05)
        
        # Add level labels on x-axis
        level_positions = [intensity_scalars.get(lv, 0) for lv in level_order_all]
        plt.xticks(level_positions, [f"{lv}\n({intensity_scalars.get(lv, 0):.2f})" for lv in level_order_all])
        
        # Add baseline reference line
        if f"{metric}_baseline" in g.columns and not g[f"{metric}_baseline"].isna().all():
            baseline_val = g[f"{metric}_baseline"].iloc[0]
            plt.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.7, label='Baseline (P0)')
        
        out_png = out_dir_png / f"sensitivity_curves_{dataset}_{model}-{weight}_{mode}.png"
        out_pdf = out_dir_pdf / f"sensitivity_curves_{dataset}_{model}-{weight}_{mode}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160, bbox_inches='tight')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    # 3. Impact ranking (sorted bar chart)
    impact = sub.groupby(["dataset", "model", "weight", "mode", "noise"]).agg({
        metric: ["mean", "std"]
    }).reset_index()
    impact.columns = ["dataset", "model", "weight", "mode", "noise", f"{metric}_mean", f"{metric}_std"]
    
    for (dataset, model, weight, mode), g in impact.groupby(["dataset", "model", "weight", "mode"]):
        # Sort by metric (ascending = worse noise first)
        g = g.sort_values(f"{metric}_mean", ascending=True)
        
        plt.figure(figsize=(10, max(5, len(g) * 0.4)))
        colors = plt.cm.RdYlGn(g[f"{metric}_mean"].values)
        
        bars = plt.barh(range(len(g)), g[f"{metric}_mean"].values, 
                       xerr=g[f"{metric}_std"].values, color=colors, capsize=3)
        
        plt.yticks(range(len(g)), g["noise"].values)
        plt.xlabel(f"Mean {metric.capitalize()}")
        plt.ylabel("Noise Type")
        plt.title(f"Noise Impact Ranking — {dataset} | {model}/{weight} | {mode}")
        plt.xlim(0, 1)
        
        # Add value labels
        for i, (val, std) in enumerate(zip(g[f"{metric}_mean"].values, g[f"{metric}_std"].values)):
            plt.text(val + 0.02, i, f"{val:.3f}±{std:.3f}", va='center', fontsize=8)
        
        out_png = out_dir_png / f"impact_ranking_{dataset}_{model}-{weight}_{mode}.png"
        out_pdf = out_dir_pdf / f"impact_ranking_{dataset}_{model}-{weight}_{mode}.pdf"

        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_summary_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    stability_df: pd.DataFrame = None,
    metrics: List[str] = None
) -> List[str]:
    """
    Create summary heatmaps combining segmentation and stability metrics.
    
    Generates:
      1. Metric heatmap: models × noise types
      2. Stability heatmap: models × noise types (drop/slope/AUC)
      3. Combined summary heatmap
    
    Args:
        df: Results DataFrame
        out_dir: Output directory
        stability_df: Stability metrics DataFrame
        metrics: Metrics to include (default: ["dice", "iou"])
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    
    if metrics is None:
        metrics = ["dice", "iou"]
    
    # Use P1 protocol for noise comparison
    sub = df[df["protocol"] == "P1"].copy()
    if len(sub) == 0:
        return paths
    
    # 1. Metric heatmap per dataset
    for metric in metrics:
        if metric not in sub.columns:
            continue
        
        for dataset, g in sub.groupby("dataset"):
            # Aggregate: model/weight × noise
            agg = g.groupby(["model", "weight", "noise"])[metric].mean().reset_index()
            agg["model_weight"] = agg["model"] + "/" + agg["weight"]
            
            pivot = agg.pivot(index="model_weight", columns="noise", values=metric)
            
            if pivot.empty:
                continue
            
            plt.figure(figsize=(max(10, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.8)))
            im = plt.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
            plt.colorbar(im, label=metric.capitalize())
            
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xlabel("Noise Type")
            plt.ylabel("Model")
            plt.title(f"Summary: {metric} — {dataset}")
            
            # Annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = 'white' if val < 0.5 else 'black'
                        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
            
            out_png = out_dir_png / f"summary_heatmap_{metric}_{dataset}.png"
            out_pdf = out_dir_pdf / f"summary_heatmap_{metric}_{dataset}.pdf"
            
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.savefig(out_pdf, bbox_inches="tight")
            plt.close()
            paths.append(str(out_png))
            paths.append(str(out_pdf))
    
    # 2. Stability heatmap (if provided)
    if stability_df is not None and len(stability_df) > 0:
        stability_metrics = ["drop_Lmax", "slope", "auc", "cv"]
        available = [m for m in stability_metrics if m in stability_df.columns]
        
        for stab_metric in available:
            for dataset, g in stability_df.groupby("dataset"):
                if "model" not in g.columns or "noise" not in g.columns:
                    continue
                
                agg = g.groupby(["model", "weight", "noise"])[stab_metric].mean().reset_index()
                agg["model_weight"] = agg["model"] + "/" + agg["weight"]
                
                pivot = agg.pivot(index="model_weight", columns="noise", values=stab_metric)
                
                if pivot.empty:
                    continue
                
                # Choose colormap based on metric
                if stab_metric == "drop_Lmax":
                    cmap = "Reds"  # Higher drop = worse
                    vmin, vmax = 0, pivot.max().max()
                elif stab_metric == "slope":
                    cmap = "RdBu_r"  # Negative slope = worse
                    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
                    vmin, vmax = -max_abs, max_abs
                elif stab_metric == "auc":
                    cmap = "RdYlGn"  # Higher AUC = better
                    vmin, vmax = 0, 1
                else:  # cv
                    cmap = "Reds"  # Higher CV = worse
                    vmin, vmax = 0, pivot.max().max()
                
                plt.figure(figsize=(max(10, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.8)))
                im = plt.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
                plt.colorbar(im, label=stab_metric)
                
                plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
                plt.yticks(range(len(pivot.index)), pivot.index)
                plt.xlabel("Noise Type")
                plt.ylabel("Model")
                plt.title(f"Stability: {stab_metric} — {dataset}")
                
                # Annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
                
                out_png = out_dir_png / f"stability_heatmap_{stab_metric}_{dataset}.png"
                out_pdf = out_dir_pdf / f"stability_heatmap_{stab_metric}_{dataset}.pdf"
                plt.tight_layout()
                plt.savefig(out_png, dpi=160)
                plt.savefig(out_pdf, bbox_inches="tight")
                plt.close()
                paths.append(str(out_png))
                paths.append(str(out_pdf))
    
    return paths


def plot_uncertainty_vs_performance(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "dice"
) -> List[str]:
    """
    Plot uncertainty metrics vs segmentation performance.
    
    Shows correlation between:
      - Confidence and Dice
      - Entropy and Dice
    
    Args:
        df: Results DataFrame with uncertainty columns
        out_dir: Output directory
        metric: Performance metric (default: "dice")
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    
    uncertainty_cols = ["mean_confidence", "mean_entropy", "boundary_entropy", 
                       "mean_confidence_proxy", "mask_consistency_iou"]
    available = [c for c in uncertainty_cols if c in df.columns]
    
    if len(available) == 0 or metric not in df.columns:
        return paths
    
    for uncert in available:
        valid = df[[metric, uncert]].dropna()
        if len(valid) < 10:
            continue
        
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.scatter(valid[metric], valid[uncert], alpha=0.3, s=20)
        
        # Add correlation (safe)
        corr = _safe_corrcoef(valid[metric].values, valid[uncert].values)
        
        # Trend line
        z = np.polyfit(valid[metric], valid[uncert], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
        plt.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend (r={corr:.3f})')
        
        plt.xlabel(metric.capitalize())
        plt.ylabel(uncert.replace("_", " ").title())
        plt.title(f"{uncert} vs {metric}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_png = out_dir_png / f"uncertainty_{uncert}_vs_{metric}.png"
        out_pdf = out_dir_pdf / f"uncertainty_{uncert}_vs_{metric}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths


def plot_psnr_vs_performance(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "dice"
) -> List[str]:
    """
    Plot PSNR/SSIM vs segmentation performance.
    
    Args:
        df: Results DataFrame with psnr, ssim columns
        out_dir: Output directory
        metric: Performance metric (default: "dice")
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir_png = out_dir / "plot_png"
    out_dir_png.mkdir(parents=True, exist_ok=True)
    out_dir_pdf = out_dir / "plot_pdf"
    out_dir_pdf.mkdir(parents=True, exist_ok=True)

    paths = []
    
    for qual_metric in ["psnr", "ssim"]:
        if qual_metric not in df.columns or metric not in df.columns:
            continue
        
        valid = df[[metric, qual_metric, "noise"]].dropna()
        if len(valid) < 10:
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Color by noise type
        noise_types = valid["noise"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(noise_types)))
        
        for noise, color in zip(noise_types, colors):
            subset = valid[valid["noise"] == noise]
            plt.scatter(subset[qual_metric], subset[metric], alpha=0.5, s=30, 
                       label=noise, color=color)
        
        # Overall trend line (safe correlation)
        corr = _safe_corrcoef(valid[qual_metric].values, valid[metric].values)
        z = np.polyfit(valid[qual_metric], valid[metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[qual_metric].min(), valid[qual_metric].max(), 100)
        plt.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=2, label=f'Trend (r={corr:.3f})')
        
        plt.xlabel(qual_metric.upper())
        plt.ylabel(metric.capitalize())
        plt.title(f"{qual_metric.upper()} vs {metric.capitalize()}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        
        out_png = out_dir_png / f"{qual_metric}_vs_{metric}.png"
        out_pdf = out_dir_pdf / f"{qual_metric}_vs_{metric}.pdf"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()
        paths.append(str(out_png))
        paths.append(str(out_pdf))
    
    return paths