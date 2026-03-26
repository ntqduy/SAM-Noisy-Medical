"""
Comprehensive Visualization Suite for SAM Robustness Benchmark.

Generates all required figures for academic reports:
1. Prompt mode illustration
2. Noise gallery (all L0-L9 levels)
3. Line plots per mode (all metrics, all levels)
4. Mode comparison plots
5. Robustness heatmaps
6. Ranking/summary plots
7. Segmentation quality gallery

Key features:
- NO TITLE on figures (filename serves as description)
- Handles HD metric direction correctly
- Preserves ALL levels L0-L9
- Exports to PDF for publication
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

METRICS = ["IoU", "Dice", "Recall", "Precision", "F1", "HD"]

METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "IoU": True,
    "Dice": True,
    "Recall": True,
    "Precision": True,
    "F1": True,
    "HD": False,
}

# Prompt mode display names
PROMPT_DISPLAY = {
    "prompt_point": "point",
    "prompt_bbox": "bbox",
    "prompt_point_box": "point+bbox",
}

# Color palettes
MODEL_PALETTE = sns.color_palette("colorblind", 12)
MODE_PALETTE = sns.color_palette("Set1", 5)
NOISE_PALETTE = sns.color_palette("husl", 15)


def _level_key(level: str) -> int:
    """Extract numeric index from level string."""
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _sorted_levels(levels: List[str]) -> List[str]:
    """Sort levels L0-L9 in correct order."""
    return sorted(set(str(lv) for lv in levels), key=_level_key)


def _slugify(name: str) -> str:
    """Convert name to filename-safe slug."""
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower())
    return out.strip("_") or "item"


def _prompt_display(mode: str) -> str:
    """Convert canonical prompt mode to display name."""
    return PROMPT_DISPLAY.get(mode, mode)


def _apply_paper_style() -> None:
    """Apply publication-ready matplotlib style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 0,  # No title
    })


def _metric_ylabel(metric: str) -> str:
    """Get y-axis label for metric with direction hint."""
    if METRIC_HIGHER_IS_BETTER.get(metric, True):
        return f"{metric} (↑ better)"
    else:
        return f"{metric} (↓ better)"


# ═══════════════════════════════════════════════════════════════════════════
#  COMPREHENSIVE VISUALIZATION SUITE
# ═══════════════════════════════════════════════════════════════════════════

class ComprehensiveVisualization:
    """
    Generate all required visualizations from benchmark statistics.

    Parameters
    ----------
    csv_path : Path
        Path to statistics CSV (merged or raw).
    output_dir : Path
        Base directory for output figures.
    artifact_root : Path, optional
        Root directory containing saved images/masks for galleries.
    """

    def __init__(
        self,
        csv_path: Path,
        output_dir: Path,
        artifact_root: Optional[Path] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root) if artifact_root else None

        self.df = self._load_and_preprocess()
        _apply_paper_style()

    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load CSV and preprocess columns."""
        df = pd.read_csv(self.csv_path)

        # Standardize string columns
        for col in ["dataset", "model", "prompt_mode", "noise_type", "noise_level"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Add derived columns
        if "noise_level" in df.columns:
            df["level_idx"] = df["noise_level"].map(_level_key)
            df["is_clean"] = df["noise_level"].map(lambda x: str(x).upper() == "L0")

        if "prompt_mode" in df.columns:
            df["prompt_display"] = df["prompt_mode"].map(_prompt_display)

        # Coerce metrics
        for m in METRICS:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")

        return df

    def _dataset_dir(self, dataset: str) -> Path:
        """Get output directory for a dataset."""
        d = self.output_dir / _slugify(dataset)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _datasets(self) -> List[str]:
        """Get unique datasets."""
        return sorted(self.df["dataset"].dropna().unique().tolist())

    def _models(self) -> List[str]:
        """Get unique models."""
        return sorted(self.df["model"].dropna().unique().tolist())

    def _levels(self) -> List[str]:
        """Get all levels in order (L0-L9 complete)."""
        if "noise_level" not in self.df.columns:
            return []
        return _sorted_levels(self.df["noise_level"].dropna().unique().tolist())

    def _noises(self) -> List[str]:
        """Get unique noise types."""
        return sorted(self.df["noise_type"].dropna().unique().tolist())

    def _modes(self) -> List[str]:
        """Get prompt modes in standard order."""
        order = ["prompt_point", "prompt_bbox", "prompt_point_box"]
        present = self.df["prompt_mode"].dropna().unique().tolist()
        return [m for m in order if m in present] + [m for m in present if m not in order]

    def _metrics(self) -> List[str]:
        """Get available metrics."""
        return [m for m in METRICS if m in self.df.columns]

    # ─────────────────────────────────────────────────────────────────────────
    #  1. PROMPT MODE SCHEMATIC
    # ─────────────────────────────────────────────────────────────────────────

    def plot_prompt_schematic(self) -> Path:
        """
        Generate schematic illustration of 3 prompt modes.
        Uses synthetic example to show point, bbox, point+bbox.
        """
        out_dir = self.output_dir / "schematics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / "prompt_modes_illustration_point_bbox_pointbox.pdf"

        size = 256
        bg = np.full((size, size), 60, dtype=np.uint8)

        # Synthetic foreground mask (off-center ellipse)
        yy, xx = np.ogrid[:size, :size]
        fg = (((xx - 100)**2) / 50**2 + ((yy - 130)**2) / 35**2) <= 1.0
        bg[fg] = 160

        # Get bounding box
        ys, xs = np.where(fg)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()

        # Center point
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        for idx, (ax, mode_name) in enumerate(zip(axes, ["point", "bbox", "point+bbox"])):
            ax.imshow(bg, cmap="gray", vmin=0, vmax=255)

            # Draw bbox for bbox and point+bbox modes
            if mode_name in ["bbox", "point+bbox"]:
                rect = plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    fill=False, linewidth=2, edgecolor="#E67E22"
                )
                ax.add_patch(rect)

            # Draw point for point and point+bbox modes
            if mode_name in ["point", "point+bbox"]:
                ax.scatter([cx], [cy], c="#1F77B4", s=50, marker="o",
                           edgecolors="white", linewidths=0.8, zorder=5)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(mode_name, fontsize=11, fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        return out_pdf

    # ─────────────────────────────────────────────────────────────────────────
    #  2. NOISE GALLERY (ALL L0-L9 LEVELS)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_noise_gallery(self, dataset: Optional[str] = None) -> Dict[str, Path]:
        """
        Generate noise gallery showing all noise types × all levels (L0-L9).

        If images are not available, creates placeholder grid.
        """
        output_paths = {}
        datasets = [dataset] if dataset else self._datasets()

        for ds in datasets:
            ds_dir = self._dataset_dir(ds)
            noises = self._noises()
            levels = self._levels()

            if not noises or not levels:
                continue

            n_rows = len(noises)
            n_cols = len(levels)

            # Create figure
            fig_w = max(10, n_cols * 1.2)
            fig_h = max(6, n_rows * 1.2)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
            axes = np.array(axes).reshape(n_rows, n_cols)

            for r, noise in enumerate(noises):
                for c, level in enumerate(levels):
                    ax = axes[r, c]
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Try to load actual image from artifacts
                    img_loaded = False
                    if self.artifact_root:
                        # Pattern: artifacts/_shared/{dataset}/{noise}/{level}/seed*/xxx_noisy.png
                        pattern = (
                            self.artifact_root / "_shared" / _slugify(ds) /
                            _slugify(noise) / level / "seed*" / "*_noisy.png"
                        )
                        matches = list(self.artifact_root.glob(
                            f"_shared/{_slugify(ds)}/{_slugify(noise)}/{level}/seed*/*_noisy.png"
                        ))
                        if matches:
                            try:
                                img = plt.imread(matches[0])
                                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                                img_loaded = True
                            except Exception:
                                pass

                    if not img_loaded:
                        # Placeholder
                        ax.imshow(np.full((64, 64), 200, dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
                        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=6)

                    # Labels
                    if r == n_rows - 1:
                        ax.set_xlabel(level, fontsize=8)
                    if c == 0:
                        ax.set_ylabel(noise, fontsize=8, rotation=90, va="center")

            fig.tight_layout()
            out_pdf = ds_dir / f"{_slugify(ds)}_noise_gallery_all_types_L0_to_L9.pdf"
            fig.savefig(out_pdf, format="pdf")
            plt.close(fig)
            output_paths[ds] = out_pdf

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    #  3. LINE PLOTS PER MODE (ALL METRICS, ALL LEVELS)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_metric_vs_level_by_mode(self) -> Dict[str, List[Path]]:
        """
        Generate line plots: metric vs level (L0-L9 complete).
        One figure per (dataset, metric, mode) with lines for each model.
        """
        output_paths: Dict[str, List[Path]] = {}

        for ds in self._datasets():
            ds_df = self.df[self.df["dataset"] == ds]
            ds_dir = self._dataset_dir(ds)
            output_paths[ds] = []

            for metric in self._metrics():
                metric_slug = _slugify(metric)

                for mode in self._modes():
                    mode_df = ds_df[ds_df["prompt_mode"] == mode]
                    if mode_df.empty:
                        continue

                    mode_slug = _slugify(_prompt_display(mode))

                    # Aggregate: mean per (model, level)
                    agg = (
                        mode_df.groupby(["model", "noise_level", "level_idx"])[metric]
                        .mean()
                        .reset_index()
                    )

                    levels = _sorted_levels(agg["noise_level"].unique().tolist())
                    models = sorted(agg["model"].unique().tolist())
                    level_to_x = {lv: i for i, lv in enumerate(levels)}

                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 5))

                    for i, model in enumerate(models):
                        g = agg[agg["model"] == model].sort_values("level_idx")
                        if g.empty:
                            continue
                        xs = [level_to_x[str(lv)] for lv in g["noise_level"]]
                        ys = g[metric].tolist()
                        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4,
                                color=MODEL_PALETTE[i % len(MODEL_PALETTE)], label=model)

                    ax.set_xlabel("Noise Level")
                    ax.set_ylabel(_metric_ylabel(metric))
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels(levels, rotation=30, ha="right")
                    ax.legend(loc="best", frameon=True, fontsize=7, ncol=2)
                    ax.grid(True, alpha=0.3)

                    fig.tight_layout()
                    out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_lineplot_by_level_{mode_slug}.pdf"
                    fig.savefig(out_pdf, format="pdf")
                    plt.close(fig)
                    output_paths[ds].append(out_pdf)

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    #  4. MODE COMPARISON PLOTS
    # ─────────────────────────────────────────────────────────────────────────

    def plot_mode_comparison(self) -> Dict[str, List[Path]]:
        """
        Compare all 3 prompt modes in single figures.
        One line per mode, x-axis = levels L0-L9.
        """
        output_paths: Dict[str, List[Path]] = {}

        for ds in self._datasets():
            ds_df = self.df[self.df["dataset"] == ds]
            ds_dir = self._dataset_dir(ds)
            output_paths[ds] = []

            for metric in self._metrics():
                metric_slug = _slugify(metric)

                # Aggregate: mean per (mode, level) across all models
                agg = (
                    ds_df.groupby(["prompt_mode", "noise_level", "level_idx"])[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                agg.columns = ["prompt_mode", "noise_level", "level_idx", "mean", "std"]

                levels = _sorted_levels(agg["noise_level"].unique().tolist())
                modes = self._modes()
                level_to_x = {lv: i for i, lv in enumerate(levels)}

                fig, ax = plt.subplots(figsize=(9, 5))

                for i, mode in enumerate(modes):
                    g = agg[agg["prompt_mode"] == mode].sort_values("level_idx")
                    if g.empty:
                        continue
                    xs = [level_to_x[str(lv)] for lv in g["noise_level"]]
                    ys = g["mean"].tolist()
                    stds = g["std"].tolist()

                    color = MODE_PALETTE[i % len(MODE_PALETTE)]
                    display = _prompt_display(mode)

                    ax.plot(xs, ys, marker="o", linewidth=2, markersize=5,
                            color=color, label=display)
                    # Add error band
                    ax.fill_between(
                        xs,
                        [y - s for y, s in zip(ys, stds)],
                        [y + s for y, s in zip(ys, stds)],
                        alpha=0.15, color=color
                    )

                ax.set_xlabel("Noise Level")
                ax.set_ylabel(_metric_ylabel(metric))
                ax.set_xticks(range(len(levels)))
                ax.set_xticklabels(levels, rotation=30, ha="right")
                ax.legend(loc="best", frameon=True)
                ax.grid(True, alpha=0.3)

                fig.tight_layout()
                out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_mode_comparison_all_levels.pdf"
                fig.savefig(out_pdf, format="pdf")
                plt.close(fig)
                output_paths[ds].append(out_pdf)

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    #  5. ROBUSTNESS HEATMAPS
    # ─────────────────────────────────────────────────────────────────────────

    def plot_heatmaps(self) -> Dict[str, List[Path]]:
        """
        Generate heatmaps:
        - Model × Noise (mean score)
        - Model × Level (degradation progression)
        """
        output_paths: Dict[str, List[Path]] = {}

        for ds in self._datasets():
            ds_df = self.df[self.df["dataset"] == ds]
            ds_dir = self._dataset_dir(ds)
            output_paths[ds] = []

            for metric in self._metrics():
                metric_slug = _slugify(metric)
                is_lower_better = not METRIC_HIGHER_IS_BETTER.get(metric, True)

                # --- Heatmap 1: Model × Noise ---
                pivot_mn = (
                    ds_df.groupby(["model", "noise_type"])[metric]
                    .mean()
                    .reset_index()
                    .pivot(index="model", columns="noise_type", values=metric)
                )

                if not pivot_mn.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    cmap = "YlGnBu_r" if is_lower_better else "YlGnBu"
                    sns.heatmap(
                        pivot_mn, cmap=cmap, annot=True, fmt=".3f",
                        linewidths=0.5, ax=ax,
                        cbar_kws={"label": _metric_ylabel(metric)}
                    )
                    ax.set_xlabel("Noise Type")
                    ax.set_ylabel("Model")
                    fig.tight_layout()
                    out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_heatmap_model_vs_noise.pdf"
                    fig.savefig(out_pdf, format="pdf")
                    plt.close(fig)
                    output_paths[ds].append(out_pdf)

                # --- Heatmap 2: Model × Level ---
                pivot_ml = (
                    ds_df.groupby(["model", "noise_level", "level_idx"])[metric]
                    .mean()
                    .reset_index()
                    .sort_values("level_idx")
                    .pivot(index="model", columns="noise_level", values=metric)
                )

                if not pivot_ml.empty:
                    # Reorder columns by level
                    levels = _sorted_levels(pivot_ml.columns.tolist())
                    pivot_ml = pivot_ml[[lv for lv in levels if lv in pivot_ml.columns]]

                    fig, ax = plt.subplots(figsize=(12, 6))
                    cmap = "YlOrRd_r" if is_lower_better else "YlOrRd"
                    sns.heatmap(
                        pivot_ml, cmap=cmap, annot=True, fmt=".3f",
                        linewidths=0.5, ax=ax,
                        cbar_kws={"label": _metric_ylabel(metric)}
                    )
                    ax.set_xlabel("Noise Level")
                    ax.set_ylabel("Model")
                    fig.tight_layout()
                    out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_heatmap_model_vs_level.pdf"
                    fig.savefig(out_pdf, format="pdf")
                    plt.close(fig)
                    output_paths[ds].append(out_pdf)

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    #  6. RANKING / SUMMARY PLOTS
    # ─────────────────────────────────────────────────────────────────────────

    def plot_rankings(self) -> Dict[str, List[Path]]:
        """
        Generate ranking bar charts:
        - Overall model ranking
        - Noise difficulty ranking
        """
        output_paths: Dict[str, List[Path]] = {}

        for ds in self._datasets():
            ds_df = self.df[self.df["dataset"] == ds]
            ds_dir = self._dataset_dir(ds)
            output_paths[ds] = []

            for metric in self._metrics():
                metric_slug = _slugify(metric)
                is_lower_better = not METRIC_HIGHER_IS_BETTER.get(metric, True)

                # --- Model ranking ---
                model_means = ds_df.groupby("model")[metric].mean().reset_index()
                model_means = model_means.sort_values(
                    metric, ascending=is_lower_better
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                colors = [MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(len(model_means))]
                ax.barh(model_means["model"], model_means[metric], color=colors)
                ax.set_xlabel(_metric_ylabel(metric))
                ax.set_ylabel("Model")
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_ranking_models.pdf"
                fig.savefig(out_pdf, format="pdf")
                plt.close(fig)
                output_paths[ds].append(out_pdf)

                # --- Noise difficulty ranking ---
                # Compute drop from L0 to noisy mean for each noise type
                noise_drops = []
                for noise in self._noises():
                    noise_df = ds_df[ds_df["noise_type"] == noise]
                    clean = noise_df[noise_df["is_clean"] == True][metric].mean()
                    noisy = noise_df[noise_df["is_clean"] == False][metric].mean()

                    if METRIC_HIGHER_IS_BETTER.get(metric, True):
                        drop = clean - noisy  # Positive = degraded
                    else:
                        drop = noisy - clean  # Positive = degraded

                    noise_drops.append({"noise_type": noise, "degradation": drop})

                drop_df = pd.DataFrame(noise_drops).sort_values("degradation", ascending=False)

                fig, ax = plt.subplots(figsize=(10, 5))
                colors = [NOISE_PALETTE[i % len(NOISE_PALETTE)] for i in range(len(drop_df))]
                ax.barh(drop_df["noise_type"], drop_df["degradation"], color=colors)
                ax.set_xlabel("Degradation (positive = more destructive)")
                ax.set_ylabel("Noise Type")
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_ranking_noise_difficulty.pdf"
                fig.savefig(out_pdf, format="pdf")
                plt.close(fig)
                output_paths[ds].append(out_pdf)

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    #  7. SEGMENTATION QUALITY GALLERY
    # ─────────────────────────────────────────────────────────────────────────

    def plot_segmentation_gallery(
        self,
        dataset: Optional[str] = None,
        max_samples: int = 5,
    ) -> Dict[str, Path]:
        """
        Generate segmentation quality gallery showing:
        - Original image
        - Noisy image
        - GT mask
        - Prediction mask
        - Overlay

        Shows progression across levels L0-L9.
        """
        output_paths = {}

        if self.artifact_root is None or not self.artifact_root.exists():
            return output_paths

        datasets = [dataset] if dataset else self._datasets()

        for ds in datasets:
            ds_dir = self._dataset_dir(ds)
            levels = self._levels()

            # Find available samples
            shared_dir = self.artifact_root / "_shared" / _slugify(ds)
            if not shared_dir.exists():
                continue

            # Find one noise type with good coverage
            noise_dirs = sorted([d for d in shared_dir.iterdir() if d.is_dir()])
            if not noise_dirs:
                continue

            best_noise = None
            best_coverage = 0
            for noise_dir in noise_dirs:
                coverage = sum(1 for lv in levels if (noise_dir / lv).exists())
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_noise = noise_dir.name

            if best_noise is None:
                continue

            # Create gallery: rows = levels, columns = [original, noisy, GT, pred, overlay]
            n_rows = len(levels)
            n_cols = 4  # original, noisy, GT, pred

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2.5))
            axes = np.array(axes).reshape(n_rows, n_cols)

            for r, level in enumerate(levels):
                level_dir = shared_dir / best_noise / level / "seed0"
                if not level_dir.exists():
                    # Skip if level not available
                    for c in range(n_cols):
                        axes[r, c].text(0.5, 0.5, "N/A", ha="center", va="center")
                        axes[r, c].set_xticks([])
                        axes[r, c].set_yticks([])
                    axes[r, 0].set_ylabel(level, fontsize=10)
                    continue

                # Find first sample
                originals = list(level_dir.glob("*_original.png"))
                if not originals:
                    continue

                sample_id = originals[0].stem.replace("_original", "")

                img_paths = {
                    "original": level_dir / f"{sample_id}_original.png",
                    "noisy": level_dir / f"{sample_id}_noisy.png",
                    "gt": level_dir / f"{sample_id}_gt.png",
                }

                col_names = ["Original", "Noisy", "GT", "Prediction"]

                for c, key in enumerate(["original", "noisy", "gt"]):
                    ax = axes[r, c]
                    path = img_paths.get(key)
                    if path and path.exists():
                        img = plt.imread(path)
                        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r == 0:
                        ax.set_xlabel(col_names[c], fontsize=9)
                    if c == 0:
                        ax.set_ylabel(level, fontsize=10)

                # Prediction column - placeholder since we'd need to search model dirs
                axes[r, 3].text(0.5, 0.5, "See\nmodel\nartifacts", ha="center", va="center", fontsize=7)
                axes[r, 3].set_xticks([])
                axes[r, 3].set_yticks([])
                if r == 0:
                    axes[r, 3].set_xlabel("Prediction", fontsize=9)

            fig.tight_layout()
            out_pdf = ds_dir / f"{_slugify(ds)}_segmentation_gallery_{best_noise}_L0_to_L9.pdf"
            fig.savefig(out_pdf, format="pdf", dpi=150)
            plt.close(fig)
            output_paths[ds] = out_pdf

        return output_paths

    # ─────────────────────────────────────────────────────────────────────────
    #  GENERATE ALL
    # ─────────────────────────────────────────────────────────────────────────

    def generate_all(self) -> Dict[str, Any]:
        """Generate all visualizations and return paths."""
        results = {}

        # 1. Prompt schematic
        results["prompt_schematic"] = self.plot_prompt_schematic()

        # 2. Noise gallery
        results["noise_gallery"] = self.plot_noise_gallery()

        # 3. Line plots per mode
        results["line_plots_by_mode"] = self.plot_metric_vs_level_by_mode()

        # 4. Mode comparison
        results["mode_comparison"] = self.plot_mode_comparison()

        # 5. Heatmaps
        results["heatmaps"] = self.plot_heatmaps()

        # 6. Rankings
        results["rankings"] = self.plot_rankings()

        # 7. Segmentation gallery
        results["segmentation_gallery"] = self.plot_segmentation_gallery()

        return results


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def generate_comprehensive_visualizations(
    csv_path: Path,
    output_dir: Path,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate all visualizations from benchmark CSV.

    Parameters
    ----------
    csv_path : Path
        Path to statistics CSV.
    output_dir : Path
        Directory to save figures.
    artifact_root : Path, optional
        Root directory with saved images for galleries.

    Returns
    -------
    Dict
        Mapping from category to output paths.
    """
    viz = ComprehensiveVisualization(csv_path, output_dir, artifact_root)
    return viz.generate_all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python comprehensive_visualization.py <csv_path> <output_dir> [artifact_root]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    artifact_root = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    results = generate_comprehensive_visualizations(csv_path, output_dir, artifact_root)
    print(f"Generated visualizations in {output_dir}")
