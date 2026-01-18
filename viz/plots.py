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
