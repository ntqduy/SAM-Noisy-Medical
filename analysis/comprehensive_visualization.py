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
from PIL import Image

from models.wrappers.prompt_utils import resolve_prompt


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


def _prompt_suffix(mode: str) -> str:
    """Match stage-1 artifact directory naming for prompt modes."""
    mapping = {
        "prompt_point": "point",
        "prompt_multi_point": "multipoint",
        "prompt_bbox": "bbox",
        "prompt_point_box": "pointbox",
        "autogen": "autogen",
    }
    return mapping.get(mode, _slugify(mode))


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

def _read_uint8_image(path: Path) -> np.ndarray:
    """Load an image and normalize it to uint8 for consistent plotting."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and float(arr.max()) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _read_binary_mask(path: Path) -> np.ndarray:
    """Load a binary mask from disk as uint8 {0,1}."""
    arr = _read_uint8_image(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


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

    def _shared_dataset_dir(self, dataset: str) -> Optional[Path]:
        """Return dataset-specific shared artifact directory when available."""
        if self.artifact_root is None:
            return None
        ds_dir = self.artifact_root / "_shared" / _slugify(dataset)
        return ds_dir if ds_dir.exists() else None

    def _find_prompt_sample(self, dataset: str) -> Optional[Dict[str, Any]]:
        """
        Pick a real evaluation sample from stage-1 shared artifacts.

        Selection policy is deterministic and tied to the benchmark output:
        prefer the first clean ``L0/seed0`` sample saved by stage 1, then fall
        back to the first available shared sample.
        """
        ds_dir = self._shared_dataset_dir(dataset)
        if ds_dir is None:
            return None

        patterns = [
            "*/L0/seed0/*_original.png",
            "*/L0/seed*/*_original.png",
            "*/*/seed*/*_original.png",
        ]
        for pattern in patterns:
            for original_path in sorted(ds_dir.glob(pattern)):
                sample_id = original_path.stem.replace("_original", "")
                gt_path = original_path.with_name(f"{sample_id}_gt.png")
                noisy_path = original_path.with_name(f"{sample_id}_noisy.png")
                if not gt_path.exists():
                    continue
                rel = original_path.relative_to(ds_dir)
                return {
                    "sample_id": sample_id,
                    "original_path": original_path,
                    "gt_path": gt_path,
                    "noisy_path": noisy_path if noisy_path.exists() else None,
                    "noise_type": rel.parts[0] if len(rel.parts) >= 1 else "",
                    "noise_level": rel.parts[1] if len(rel.parts) >= 2 else "",
                    "seed_dir": rel.parts[2] if len(rel.parts) >= 3 else "",
                }
        return None

    def _shared_noise_level_map(
        self,
        dataset: str,
        noise: str,
        *,
        suffix: str,
    ) -> Dict[str, Dict[str, Path]]:
        """Map sample_id -> level -> shared artifact path for one dataset/noise."""
        ds_dir = self._shared_dataset_dir(dataset)
        if ds_dir is None:
            return {}

        noise_dir = ds_dir / _slugify(noise)
        if not noise_dir.exists():
            return {}

        mapping: Dict[str, Dict[str, Path]] = {}
        for path in sorted(noise_dir.glob(f"*/seed*/*_{suffix}.png")):
            rel = path.relative_to(noise_dir)
            if len(rel.parts) < 3:
                continue
            level = rel.parts[0]
            sample_id = path.stem.replace(f"_{suffix}", "")
            mapping.setdefault(sample_id, {})[level] = path
        return mapping

    @staticmethod
    def _best_sample_id(level_map: Dict[str, Dict[str, Path]], levels: List[str]) -> Optional[str]:
        """Choose one consistent sample that maximizes level coverage."""
        best_sample_id: Optional[str] = None
        best_cov = -1
        for sample_id in sorted(level_map):
            coverage = sum(1 for lv in levels if lv in level_map[sample_id])
            if coverage > best_cov:
                best_cov = coverage
                best_sample_id = sample_id
        return best_sample_id

    def _prediction_artifact_path(
        self,
        dataset: str,
        model: str,
        prompt_mode: str,
        noise: str,
        level: str,
        sample_id: str,
        *,
        seed: int = 0,
    ) -> Optional[Path]:
        """Return prediction artifact path for one dataset/model/prompt/noise/level/sample."""
        if self.artifact_root is None:
            return None
        path = (
            self.artifact_root
            / _slugify(dataset)
            / _slugify(model)
            / _prompt_suffix(prompt_mode)
            / _slugify(noise)
            / level
            / f"seed{seed}"
            / f"{sample_id}_pred.png"
        )
        return path if path.exists() else None

    def plot_prompt_schematic(self) -> Path:
        """
        Generate schematic illustration of 3 prompt modes.
        Reuses a real stage-1 sample when shared artifacts are available.
        """
        out_dir = self.output_dir / "schematics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / "prompt_modes_illustration_point_bbox_pointbox.pdf"

        modes = [
            ("point", "prompt_point"),
            ("bbox", "prompt_bbox"),
            ("point+bbox", "prompt_point_box"),
        ]
        pages_written = 0

        with PdfPages(out_pdf) as pdf:
            for dataset in self._datasets():
                sample = self._find_prompt_sample(dataset)
                if sample is None:
                    continue

                image = _read_uint8_image(sample["original_path"])
                gt_mask = _read_binary_mask(sample["gt_path"])
                if image.shape[:2] != gt_mask.shape[:2]:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
                for ax, (display_label, prompt_mode) in zip(axes, modes):
                    resolved = resolve_prompt(
                        {"gt_mask": gt_mask},
                        image.shape[:2],
                        prompt_mode=prompt_mode,
                    )

                    if image.ndim == 2:
                        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                    else:
                        ax.imshow(image)

                    bbox = resolved.get("bbox")
                    if bbox is not None:
                        x0, y0, x1, y1 = [int(v) for v in bbox]
                        rect = plt.Rectangle(
                            (x0, y0),
                            max(1, x1 - x0 + 1),
                            max(1, y1 - y0 + 1),
                            fill=False,
                            linewidth=2.0,
                            edgecolor="#E67E22",
                        )
                        ax.add_patch(rect)

                    pts = resolved.get("points")
                    if pts is not None and np.asarray(pts).size > 0:
                        pts_arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                        ax.scatter(
                            pts_arr[:, 0],
                            pts_arr[:, 1],
                            c="#1F77B4",
                            s=40,
                            marker="o",
                            edgecolors="white",
                            linewidths=0.8,
                            zorder=5,
                        )

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(display_label, fontsize=11, fontweight="bold")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                pages_written += 1

            if pages_written == 0:
                size = 256
                bg = np.full((size, size), 60, dtype=np.uint8)
                yy, xx = np.ogrid[:size, :size]
                fg = (((xx - 100) ** 2) / 50**2 + ((yy - 130) ** 2) / 35**2) <= 1.0
                bg[fg] = 160
                gt_mask = fg.astype(np.uint8)
                fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
                for ax, (display_label, prompt_mode) in zip(axes, modes):
                    resolved = resolve_prompt(
                        {"gt_mask": gt_mask},
                        bg.shape[:2],
                        prompt_mode=prompt_mode,
                    )
                    ax.imshow(bg, cmap="gray", vmin=0, vmax=255)
                    bbox = resolved.get("bbox")
                    if bbox is not None:
                        x0, y0, x1, y1 = [int(v) for v in bbox]
                        rect = plt.Rectangle(
                            (x0, y0),
                            max(1, x1 - x0 + 1),
                            max(1, y1 - y0 + 1),
                            fill=False,
                            linewidth=2.0,
                            edgecolor="#E67E22",
                        )
                        ax.add_patch(rect)
                    pts = resolved.get("points")
                    if pts is not None and np.asarray(pts).size > 0:
                        pts_arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                        ax.scatter(
                            pts_arr[:, 0],
                            pts_arr[:, 1],
                            c="#1F77B4",
                            s=40,
                            marker="o",
                            edgecolors="white",
                            linewidths=0.8,
                            zorder=5,
                        )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(display_label, fontsize=11, fontweight="bold")
                fig.tight_layout()
                pdf.savefig(fig)
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
                noise_level_map = self._shared_noise_level_map(ds, noise, suffix="noisy")
                best_sample_id = self._best_sample_id(noise_level_map, levels)
                for c, level in enumerate(levels):
                    ax = axes[r, c]
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Try to load actual image from artifacts
                    img_loaded = False
                    if self.artifact_root:
                        path = None
                        if best_sample_id is not None:
                            path = noise_level_map.get(best_sample_id, {}).get(level)
                        if path is not None and path.exists():
                            try:
                                img = _read_uint8_image(path)
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
                    ax.set_ylim(bottom=0)

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
                ax.set_ylim(bottom=0)

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
        - Model × Noise (mean score), per prompt mode
        - Model × Level (degradation progression), per prompt mode
        """
        output_paths: Dict[str, List[Path]] = {}

        for ds in self._datasets():
            ds_df = self.df[self.df["dataset"] == ds]
            ds_dir = self._dataset_dir(ds)
            output_paths[ds] = []

            for metric in self._metrics():
                metric_slug = _slugify(metric)
                is_lower_better = not METRIC_HIGHER_IS_BETTER.get(metric, True)
                for prompt_mode in self._modes():
                    prompt_df = ds_df[ds_df["prompt_mode"] == prompt_mode]
                    if prompt_df.empty:
                        continue
                    prompt_slug = _slugify(_prompt_display(prompt_mode))

                    # --- Heatmap 1: Model × Noise ---
                    pivot_mn = (
                        prompt_df.groupby(["model", "noise_type"])[metric]
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
                        out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_heatmap_model_vs_noise_{prompt_slug}.pdf"
                        fig.savefig(out_pdf, format="pdf")
                        plt.close(fig)
                        output_paths[ds].append(out_pdf)

                    # --- Heatmap 2: Model × Level ---
                    pivot_ml = (
                        prompt_df.groupby(["model", "noise_level", "level_idx"])[metric]
                        .mean()
                        .reset_index()
                        .sort_values("level_idx")
                        .pivot(index="model", columns="noise_level", values=metric)
                    )

                    if not pivot_ml.empty:
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
                        out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_heatmap_model_vs_level_{prompt_slug}.pdf"
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
        target_level: str = "L9",
        seed: int = 0,
    ) -> Dict[str, Path]:
        """
        Generate prompt-specific qualitative galleries.

        Layout per PDF:
        - one PDF per (dataset, prompt_mode)
        - one row per noise type
        - columns: original, noisy at ``target_level``, GT, predictions for models
        """
        output_paths: Dict[str, Path] = {}

        if self.artifact_root is None or not self.artifact_root.exists():
            return output_paths

        datasets = [dataset] if dataset else self._datasets()

        for ds in datasets:
            ds_dir = self._dataset_dir(ds)
            shared_dir = self.artifact_root / "_shared" / _slugify(ds)
            if not shared_dir.exists():
                continue

            ds_df = self.df[self.df["dataset"] == ds]
            noises = sorted(ds_df["noise_type"].dropna().astype(str).unique().tolist())

            for prompt_mode in self._modes():
                mode_df = ds_df[ds_df["prompt_mode"] == prompt_mode]
                models = sorted(mode_df["model"].dropna().astype(str).unique().tolist())
                if not models or not noises:
                    continue
                prompt_slug = _slugify(_prompt_display(prompt_mode))

                rows: List[Tuple[str, str, Path, Path, Path, Dict[str, Optional[Path]]]] = []
                for noise in noises:
                    noisy_level_map = self._shared_noise_level_map(ds, noise, suffix="noisy")
                    sample_id = self._best_sample_id(noisy_level_map, [target_level])
                    if sample_id is None:
                        continue

                    noisy_path = noisy_level_map.get(sample_id, {}).get(target_level)
                    if noisy_path is None or not noisy_path.exists():
                        continue

                    noise_dir = shared_dir / _slugify(noise)
                    clean_dir = noise_dir / "L0" / f"seed{seed}"
                    level_dir = noise_dir / target_level / f"seed{seed}"

                    original_path = clean_dir / f"{sample_id}_original.png"
                    if not original_path.exists():
                        original_path = level_dir / f"{sample_id}_original.png"
                    gt_path = clean_dir / f"{sample_id}_gt.png"
                    if not gt_path.exists():
                        gt_path = level_dir / f"{sample_id}_gt.png"
                    if not original_path.exists() or not gt_path.exists():
                        continue

                    pred_paths = {
                        model: self._prediction_artifact_path(
                            ds,
                            model,
                            prompt_mode,
                            noise,
                            target_level,
                            sample_id,
                            seed=seed,
                        )
                        for model in models
                    }
                    rows.append((noise, sample_id, original_path, noisy_path, gt_path, pred_paths))

                if not rows:
                    continue

                headers = ["Original", f"Noisy {target_level}", "GT"] + models
                n_rows = len(rows)
                n_cols = len(headers)
                fig_w = max(12.0, 2.3 * n_cols)
                fig_h = max(5.0, 2.2 * n_rows)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

                for r, (noise, _, original_path, noisy_path, gt_path, pred_paths) in enumerate(rows):
                    cell_paths: List[Optional[Path]] = [original_path, noisy_path, gt_path] + [
                        pred_paths.get(model) for model in models
                    ]
                    for c, path in enumerate(cell_paths):
                        ax = axes[r, c]
                        if path is not None and path.exists():
                            img = _read_uint8_image(path)
                            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                        else:
                            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if r == 0:
                            ax.set_xlabel(headers[c], fontsize=9)
                        if c == 0:
                            ax.set_ylabel(str(noise), fontsize=9, rotation=90, va="center")

                fig.tight_layout()
                out_pdf = (
                    ds_dir
                    / f"{_slugify(ds)}_segmentation_gallery_{prompt_slug}_{target_level}_by_noise.pdf"
                )
                fig.savefig(out_pdf, format="pdf", dpi=150)
                plt.close(fig)
                output_paths[f"{ds}|{prompt_mode}"] = out_pdf

        return output_paths

    def plot_model_line_galleries(
        self,
        dataset: Optional[str] = None,
        metric: str = "Dice",
        max_cols: int = 3,
    ) -> Dict[str, Path]:
        """
        Generate one line-plot gallery per prompt mode.

        Each subplot corresponds to one noise type, and each line corresponds
        to one model across all levels.
        """
        output_paths: Dict[str, Path] = {}
        if metric not in self._metrics():
            return output_paths

        datasets = [dataset] if dataset else self._datasets()
        metric_slug = _slugify(metric)

        for ds in datasets:
            ds_dir = self._dataset_dir(ds)
            ds_df = self.df[self.df["dataset"] == ds]
            noises = sorted(ds_df["noise_type"].dropna().astype(str).unique().tolist())
            levels = self._levels()
            if not noises or not levels:
                continue

            for prompt_mode in self._modes():
                mode_df = ds_df[ds_df["prompt_mode"] == prompt_mode]
                models = sorted(mode_df["model"].dropna().astype(str).unique().tolist())
                if not models:
                    continue
                prompt_slug = _slugify(_prompt_display(prompt_mode))

                n_cols = max(1, min(max_cols, len(noises)))
                n_rows = (len(noises) + n_cols - 1) // n_cols
                fig_w = max(10.0, 4.2 * n_cols)
                fig_h = max(4.5, 3.2 * n_rows)
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(fig_w, fig_h),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )
                level_to_x = {lv: i for i, lv in enumerate(levels)}

                for idx, noise in enumerate(noises):
                    r = idx // n_cols
                    c = idx % n_cols
                    ax = axes[r, c]
                    noise_df = mode_df[mode_df["noise_type"] == noise]
                    agg = (
                        noise_df.groupby(["model", "noise_level", "level_idx"])[metric]
                        .mean()
                        .reset_index()
                    )

                    for model_idx, model in enumerate(models):
                        g = agg[agg["model"] == model].sort_values("level_idx")
                        if g.empty:
                            continue
                        xs = [level_to_x[str(lv)] for lv in g["noise_level"]]
                        ys = g[metric].astype(float).tolist()
                        ax.plot(
                            xs,
                            ys,
                            marker="o",
                            linewidth=1.5,
                            markersize=3.5,
                            color=MODEL_PALETTE[model_idx % len(MODEL_PALETTE)],
                            label=model,
                        )

                    ax.text(
                        0.03,
                        0.94,
                        str(noise),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
                    )
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels(levels, rotation=30, ha="right")
                    ax.grid(True, alpha=0.25)
                    ax.set_ylim(bottom=0)
                    if r == n_rows - 1:
                        ax.set_xlabel("Noise Level")
                    if c == 0:
                        ax.set_ylabel(_metric_ylabel(metric))

                for idx in range(len(noises), n_rows * n_cols):
                    r = idx // n_cols
                    c = idx % n_cols
                    axes[r, c].axis("off")

                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                        marker="o",
                        linewidth=1.5,
                        markersize=3.5,
                    )
                    for i, _ in enumerate(models)
                ]
                fig.legend(
                    handles,
                    models,
                    loc="upper center",
                    ncol=min(6, len(models)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 1.0),
                )
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_line_gallery_{prompt_slug}_by_noise.pdf"
                fig.savefig(out_pdf, format="pdf")
                plt.close(fig)
                output_paths[f"{ds}|{prompt_mode}"] = out_pdf

        return output_paths

    def plot_model_bar_galleries(
        self,
        dataset: Optional[str] = None,
        metric: str = "Dice",
        max_cols: int = 3,
    ) -> Dict[str, Path]:
        """
        Generate one grouped-bar gallery per prompt mode.

        Each subplot corresponds to one noise type, and bars show
        ``metric(L9) - metric(L0)`` for each model.
        """
        output_paths: Dict[str, Path] = {}
        if metric not in self._metrics():
            return output_paths

        datasets = [dataset] if dataset else self._datasets()
        metric_slug = _slugify(metric)

        for ds in datasets:
            ds_dir = self._dataset_dir(ds)
            ds_df = self.df[self.df["dataset"] == ds]
            noises = sorted(ds_df["noise_type"].dropna().astype(str).unique().tolist())
            levels = self._levels()
            if not noises or not levels:
                continue

            for prompt_mode in self._modes():
                mode_df = ds_df[ds_df["prompt_mode"] == prompt_mode]
                models = sorted(mode_df["model"].dropna().astype(str).unique().tolist())
                if not models:
                    continue
                prompt_slug = _slugify(_prompt_display(prompt_mode))

                n_cols = max(1, min(max_cols, len(noises)))
                n_rows = (len(noises) + n_cols - 1) // n_cols
                fig_w = max(10.0, 4.6 * n_cols)
                fig_h = max(4.5, 3.4 * n_rows)
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(fig_w, fig_h),
                    squeeze=False,
                    sharex=True,
                    sharey=True,
                )
                x = np.arange(len(models), dtype=np.float32)

                for idx, noise in enumerate(noises):
                    r = idx // n_cols
                    c = idx % n_cols
                    ax = axes[r, c]
                    noise_df = mode_df[mode_df["noise_type"] == noise]
                    agg = (
                        noise_df.groupby(["model", "noise_level", "level_idx"])[metric]
                        .mean()
                        .reset_index()
                    )

                    deltas: List[float] = []
                    for model_idx, model in enumerate(models):
                        g = agg[agg["model"] == model].copy()
                        if g.empty:
                            deltas.append(np.nan)
                            continue
                        value_map = {
                            str(lv): float(val)
                            for lv, val in zip(g["noise_level"].astype(str), g[metric].astype(float))
                        }
                        l0 = value_map.get("L0", np.nan)
                        l9 = value_map.get("L9", np.nan)
                        deltas.append(float(l9 - l0) if np.isfinite(l0) and np.isfinite(l9) else np.nan)

                    ax.bar(
                        x,
                        deltas,
                        color=[MODEL_PALETTE[i % len(MODEL_PALETTE)] for i in range(len(models))],
                    )
                    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

                    ax.text(
                        0.03,
                        0.94,
                        str(noise),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(models, rotation=25, ha="right")
                    ax.grid(True, axis="y", alpha=0.25)
                    if r == n_rows - 1:
                        ax.set_xlabel("Model")
                    if c == 0:
                        ax.set_ylabel(f"{metric} (L9 - L0)")

                for idx in range(len(noises), n_rows * n_cols):
                    r = idx // n_cols
                    c = idx % n_cols
                    axes[r, c].axis("off")

                fig.tight_layout()
                out_pdf = ds_dir / f"{_slugify(ds)}_{metric_slug}_delta_l9_vs_l0_bar_gallery_{prompt_slug}_by_noise.pdf"
                fig.savefig(out_pdf, format="pdf")
                plt.close(fig)
                output_paths[f"{ds}|{prompt_mode}"] = out_pdf

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

        # 4. Skip merged prompt-mode comparison per updated visualization requirement
        results["mode_comparison"] = {}

        # 5. Heatmaps
        results["heatmaps"] = self.plot_heatmaps()

        # 6. Rankings
        results["rankings"] = self.plot_rankings()

        # 7. Segmentation gallery
        results["segmentation_gallery"] = self.plot_segmentation_gallery()

        # 8. Prompt-specific line galleries by noise type
        results["model_line_galleries"] = self.plot_model_line_galleries()

        # 9. Prompt-specific grouped-bar galleries by noise type
        results["model_bar_galleries"] = self.plot_model_bar_galleries()

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
