from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_metric_vs_level(df: pd.DataFrame, out_dir: Path, protocols: List[str], metrics: List[str]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
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
            plt.title(f"{metric} vs level — {dataset} | {model}/{weight} | {mode} | {protocol} | {noise}")
            plt.xlabel("Level")
            plt.ylabel(metric)
            out = out_dir / f"{metric}_vs_level__{dataset}__{model}-{weight}__{mode}__{protocol}__{noise}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=160)
            plt.close()
            paths.append(str(out))
    return paths


def plot_ofat_sensitivity(df: pd.DataFrame, out_dir: Path, metrics: List[str], protocols: List[str]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
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
            plt.title(f"OFAT {protocol}: {metric} — {dataset} | {model}/{weight} | {mode} | {noise}")
            plt.xlabel("Level (sweep)")
            plt.ylabel(metric)
            out = out_dir / f"OFAT_{protocol}__{metric}__{dataset}__{model}-{weight}__{mode}__{noise}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=160)
            plt.close()
            paths.append(str(out))
    return paths


def plot_grid_heatmap(df: pd.DataFrame, out_dir: Path) -> List[str]:
    """
    Plot P3 grid results as heatmaps (p × severity).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
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
        
        out = out_dir / f"P3_grid__{dataset}__{model}-{weight}__{mode}__{noise}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
    return paths


def plot_model_comparison(df: pd.DataFrame, out_dir: Path, metric: str = "dice") -> List[str]:
    """
    Plot model comparison across noise types and levels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
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
        
        out = out_dir / f"model_comparison__{metric}__{dataset}__{noise}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
    return paths


def plot_noise_comparison(df: pd.DataFrame, out_dir: Path, metric: str = "dice") -> List[str]:
    """
    Plot noise type comparison for each model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
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
        
        out = out_dir / f"noise_comparison__{metric}__{dataset}__{model}-{weight}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
    return paths


def plot_stability_summary(stability_df: pd.DataFrame, out_dir: Path) -> List[str]:
    """
    Plot stability summary (perf_drop distribution).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
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
        
        out = out_dir / f"stability_perf_drop__{dataset}__{model}-{weight}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
    return paths


def plot_global_sensitivity(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str = "dice",
    protocols: List[str] = None
) -> List[str]:
    """
    Create global sensitivity analysis plots comparing all noise types.
    
    Generates:
      1. Sensitivity heatmap: noise types × levels
      2. Sensitivity curves: metric vs intensity_scalar
      3. Ranking plots: noise types sorted by impact
    
    Args:
        df: Results DataFrame
        out_dir: Output directory
        metric: Metric to analyze (default: "dice")
        protocols: Protocols to include (default: ["P1"])
        
    Returns:
        List of saved plot paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    if protocols is None:
        protocols = ["P1"]
    
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
        
        out = out_dir / f"global_sensitivity_heatmap__{dataset}__{model}-{weight}__{mode}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
    # 2. Sensitivity curves: metric vs intensity_scalar
    if "intensity_scalar" in sub.columns:
        for (dataset, model, weight, mode), g in sub.groupby(["dataset", "model", "weight", "mode"]):
            plt.figure(figsize=(12, 6))
            
            for noise in g["noise"].unique():
                noise_data = g[g["noise"] == noise].groupby("intensity_scalar")[metric].mean().reset_index()
                if len(noise_data) > 1:
                    noise_data = noise_data.sort_values("intensity_scalar")
                    plt.plot(noise_data["intensity_scalar"], noise_data[metric], 
                            marker="o", label=noise, linewidth=2, markersize=6)
            
            plt.xlabel("Intensity Scalar (normalized severity)")
            plt.ylabel(metric.capitalize())
            plt.title(f"Sensitivity Curves: {metric} vs Intensity — {dataset} | {model}/{weight} | {mode}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            
            # Add baseline reference line
            if f"{metric}_baseline" in g.columns:
                baseline_val = g[f"{metric}_baseline"].iloc[0]
                plt.axhline(y=baseline_val, color='gray', linestyle='--', label='Baseline (P0)')
            
            out = out_dir / f"sensitivity_curves__{dataset}__{model}-{weight}__{mode}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=160)
            plt.close()
            paths.append(str(out))
    
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
        
        out = out_dir / f"impact_ranking__{dataset}__{model}-{weight}__{mode}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
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
            
            out = out_dir / f"summary_heatmap_{metric}__{dataset}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=160)
            plt.close()
            paths.append(str(out))
    
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
                
                out = out_dir / f"stability_heatmap_{stab_metric}__{dataset}.png"
                plt.tight_layout()
                plt.savefig(out, dpi=160)
                plt.close()
                paths.append(str(out))
    
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
        
        # Add correlation
        corr = np.corrcoef(valid[metric], valid[uncert])[0, 1]
        
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
        
        out = out_dir / f"uncertainty_{uncert}_vs_{metric}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
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
        
        # Overall trend line
        corr = np.corrcoef(valid[qual_metric], valid[metric])[0, 1]
        z = np.polyfit(valid[qual_metric], valid[metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[qual_metric].min(), valid[qual_metric].max(), 100)
        plt.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=2, label=f'Trend (r={corr:.3f})')
        
        plt.xlabel(qual_metric.upper())
        plt.ylabel(metric.capitalize())
        plt.title(f"{qual_metric.upper()} vs {metric.capitalize()}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        
        out = out_dir / f"{qual_metric}_vs_{metric}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        paths.append(str(out))
    
    return paths