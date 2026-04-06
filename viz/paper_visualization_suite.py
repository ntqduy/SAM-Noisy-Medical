"""
Publication-ready visualization suite for SAM benchmark research.

Key properties:
- Works with either long or wide CSV format.
- Auto-detects metrics and model names dynamically from input CSV.
- Normalizes prompt/noise/level fields to a consistent internal structure.
- Exports PDF figures under outputs/visualizations/{dataset_name}/...
- Uses only pandas, numpy, matplotlib, seaborn (plus Python stdlib).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from models.wrappers.prompt_utils import resolve_prompt, normalize_prompt_mode


def _slugify(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower())
    return out.strip("_") or "item"


def _level_key(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _safe_text(v: object) -> str:
    return str(v).strip() if v is not None else ""


def _coerce_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: x.strip())


def _canonical_prompt_mode(raw: object) -> str:
    """
    Normalize prompt mode using the same logic as model layer.

    Supports all standard modes: prompt_point, prompt_bbox, prompt_point_box, prompt_multi_point, autogen.
    Also supports display aliases: point, bbox, point+bbox.

    Uses normalize_prompt_mode() from models/wrappers/prompt_utils.py for consistency.
    """
    try:
        return normalize_prompt_mode(str(raw or ""))
    except ValueError:
        # For CSV data, return "prompt_unknown" as fallback instead of raising
        # This allows visualization to continue even with malformed CSV
        return "prompt_unknown"


def _prompt_display(canonical_prompt: str) -> str:
    mapping = {
        "prompt_point": "point",
        "prompt_bbox": "bbox",
        "prompt_point_box": "point+bbox",
    }
    return mapping.get(canonical_prompt, canonical_prompt)


def _sorted_prompt_modes(values: Iterable[str]) -> List[str]:
    vals = [str(v) for v in values]
    rank = {"prompt_point": 0, "prompt_bbox": 1, "prompt_point_box": 2}
    return sorted(vals, key=lambda x: (rank.get(x, 99), x))


LINE_PLOT_Y_MARGIN = 0.1


def _collect_finite_values(*series_groups: object) -> List[float]:
    values: List[float] = []
    for series in series_groups:
        arr = np.asarray(series, dtype=float).ravel()
        if arr.size == 0:
            continue
        values.extend(arr[np.isfinite(arr)].tolist())
    return values


def _line_y_lower_bound(*series_groups: object, margin: float = LINE_PLOT_Y_MARGIN) -> float:
    values = _collect_finite_values(*series_groups)
    if not values:
        return 0.0
    return max(0.0, min(values) - margin)


def _set_line_y_axis(
    ax: object,
    *series_groups: object,
    top: Optional[float] = None,
    margin: float = LINE_PLOT_Y_MARGIN,
) -> None:
    bottom = _line_y_lower_bound(*series_groups, margin=margin)
    if top is not None and top > bottom:
        ax.set_ylim(bottom=bottom, top=top)
    else:
        ax.set_ylim(bottom=bottom)


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    alias_to_canonical = {
        "dataset": "dataset",
        "data_set": "dataset",
        "model": "model",
        "model_name": "model",
        "noise_type": "noise_type",
        "noise": "noise_type",
        "noiselevel": "noise_level",
        "noise_level": "noise_level",
        "level": "noise_level",
        "prompt_mode": "prompt_mode",
        "prompt": "prompt_mode",
        "mode": "prompt_mode",
        "metric_name": "metric_name",
        "metric": "metric_name",
        "metric_value": "metric_value",
        "value": "metric_value",
        "score": "metric_value",
    }
    rename: Dict[str, str] = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9_]+", "_", str(c).strip().lower())
        if key in alias_to_canonical and alias_to_canonical[key] not in rename.values():
            rename[c] = alias_to_canonical[key]
    return df.rename(columns=rename)


def _detect_metric_columns_wide(df: pd.DataFrame) -> List[str]:
    id_cols = {"dataset", "model", "noise_type", "noise_level", "prompt_mode"}
    known_non_metrics = {
        "source_stats_file",
        "n_images",
        "n_rows",
        "noise_seed",
        "image_id",
        "gt_empty_rate",
        "pred_empty_rate",
        "n_gt_non_empty",
        "prompt_x",
        "prompt_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "gt_fg_pixels",
        "pred_fg_pixels",
        "is_gt_empty",
        "is_pred_empty",
    }

    metric_cols: List[str] = []
    for c in df.columns:
        lc = str(c).lower()
        if c in id_cols or c in known_non_metrics:
            continue
        if lc.endswith(("_std", "_cv_pct", "_n_valid")):
            continue
        s_num = pd.to_numeric(df[c], errors="coerce")
        if int(s_num.notna().sum()) == 0:
            continue
        metric_cols.append(c)
    return metric_cols


def normalize_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    """
    Normalize either long or wide CSV into canonical long format:
    dataset, model, noise_type, noise_level, prompt_mode, metric_name, metric_value
    """
    df_raw = pd.read_csv(csv_path)
    if df_raw.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "noise_type",
                "noise_level",
                "prompt_mode",
                "metric_name",
                "metric_value",
                "noise_level_idx",
                "prompt_mode_display",
            ]
        )

    df = _standardize_column_names(df_raw.copy())

    is_long = {"metric_name", "metric_value"}.issubset(df.columns)
    if is_long:
        out = df.copy()
    else:
        metric_cols = _detect_metric_columns_wide(df)
        if not metric_cols:
            raise ValueError(
                "Unable to detect metric columns from input CSV. "
                "Expected long format with metric_name/metric_value or wide format with numeric metric columns."
            )
        id_vars = [c for c in ["dataset", "model", "noise_type", "noise_level", "prompt_mode"] if c in df.columns]
        out = df.melt(
            id_vars=id_vars,
            value_vars=metric_cols,
            var_name="metric_name",
            value_name="metric_value",
        )

    for c in ["dataset", "model", "noise_type", "noise_level", "prompt_mode", "metric_name"]:
        if c not in out.columns:
            out[c] = "unknown"

    out["dataset"] = _coerce_str_series(out["dataset"])
    out["model"] = _coerce_str_series(out["model"])
    out["noise_type"] = _coerce_str_series(out["noise_type"])
    out["noise_level"] = _coerce_str_series(out["noise_level"])
    out["prompt_mode"] = out["prompt_mode"].map(_canonical_prompt_mode)
    out["prompt_mode_display"] = out["prompt_mode"].map(_prompt_display)
    out["metric_name"] = _coerce_str_series(out["metric_name"])
    out["metric_value"] = pd.to_numeric(out["metric_value"], errors="coerce")
    out = out.dropna(subset=["metric_value"])
    out["noise_level_idx"] = out["noise_level"].map(_level_key)
    out = out.sort_values(
        ["dataset", "metric_name", "noise_type", "prompt_mode", "model", "noise_level_idx", "noise_level"]
    ).reset_index(drop=True)
    return out[
        [
            "dataset",
            "model",
            "noise_type",
            "noise_level",
            "noise_level_idx",
            "prompt_mode",
            "prompt_mode_display",
            "metric_name",
            "metric_value",
        ]
    ]


def _apply_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def _read_gray_image(path: Path) -> np.ndarray:
    img = plt.imread(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        arr = arr.mean(axis=2)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max(initial=0.0) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _read_uint8_image(path: Path) -> np.ndarray:
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
    arr = _read_uint8_image(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


class PaperVisualizationSuite:
    """Generate publication-ready benchmark figures from normalized CSV data."""

    def __init__(
        self,
        csv_path: Path,
        *,
        output_root: Path = Path("outputs/visualizations"),
        artifact_root: Optional[Path] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.df = normalize_benchmark_csv(self.csv_path)
        self.artifact_root = Path(artifact_root) if artifact_root else self._infer_artifact_root()
        _apply_paper_style()

    def _infer_artifact_root(self) -> Optional[Path]:
        # Common project flow: statistics_merged.csv is under outputs/{exp}/
        # artifacts live at outputs/{exp}/artifacts
        parent = self.csv_path.parent
        cand = parent / "artifacts"
        return cand if cand.exists() else None

    def _dataset_dir(self, dataset_name: str) -> Path:
        d = self.output_root / _slugify(dataset_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _detected_metrics(self) -> List[str]:
        return sorted(self.df["metric_name"].dropna().astype(str).unique().tolist())

    def _detected_datasets(self) -> List[str]:
        return sorted(self.df["dataset"].dropna().astype(str).unique().tolist())

    def _detected_levels(self, sub_df: pd.DataFrame) -> List[str]:
        levels = sorted(sub_df["noise_level"].dropna().astype(str).unique().tolist(), key=_level_key)
        return levels

    def _find_prompt_sample(self, dataset: str) -> Optional[Dict[str, Any]]:
        if self.artifact_root is None or not self.artifact_root.exists():
            return None
        ds_dir = self.artifact_root / "_shared" / _slugify(dataset)
        if not ds_dir.exists():
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
                if not gt_path.exists():
                    continue
                return {
                    "sample_id": sample_id,
                    "original_path": original_path,
                    "gt_path": gt_path,
                }
        return None

    # -- 1) prompt schematic ---------------------------------------------------

    def plot_prompt_mode_schematic(self) -> Path:
        """
        Generate schematic diagrams for the 3 standard prompt modes:
        - prompt_point: single foreground click
        - prompt_bbox: ground-truth bounding box with adaptive margin
        - prompt_point_box: single click + ground-truth box combination

        All modes use resolve_prompt() to follow the exact model setup.
        """
        out_dir = self._dataset_dir("all_datasets")
        out_pdf = out_dir / "schematic_prompt_modes_point_bbox_pointplusbbox.pdf"

        modes = [
            ("point", "prompt_point"),
            ("bbox", "prompt_bbox"),
            ("point+bbox", "prompt_point_box"),
        ]
        pages_written = 0

        with PdfPages(out_pdf) as pdf:
            for dataset in self._detected_datasets():
                sample = self._find_prompt_sample(dataset)
                if sample is None:
                    continue

                image = _read_uint8_image(Path(sample["original_path"]))
                gt_mask = _read_binary_mask(Path(sample["gt_path"]))
                if image.shape[:2] != gt_mask.shape[:2]:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
                for ax, (display_label, prompt_mode) in zip(axes, modes):
                    resolved = resolve_prompt(
                        {"gt_mask": gt_mask},
                        image_shape=image.shape[:2],
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
                    ax.set_xlabel(display_label, fontsize=12, fontweight="bold")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                pages_written += 1

            if pages_written == 0:
                size = 256
                yy, xx = np.ogrid[:size, :size]
                bg = np.full((size, size), 58, dtype=np.uint8)
                fg = (
                    ((xx - 100) ** 2) / (52.0**2) + ((yy - 132) ** 2) / (34.0**2) <= 1.0
                ) | (
                    ((xx - 148) ** 2) / (28.0**2) + ((yy - 110) ** 2) / (24.0**2) <= 1.0
                )
                bg[fg] = 156
                gt_mask = fg.astype(np.uint8)
                fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))
                for ax, (display_label, prompt_mode) in zip(axes, modes):
                    resolved = resolve_prompt(
                        {"gt_mask": gt_mask},
                        image_shape=bg.shape[:2],
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
                    ax.set_xlabel(display_label, fontsize=12, fontweight="bold")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf

        size = 256
        yy, xx = np.ogrid[:size, :size]
        bg = np.full((size, size), 58, dtype=np.uint8)

        # Synthetic foreground is intentionally off-center to avoid accidental
        # alignment with image center in prompt visual examples.
        # Two ellipses to simulate disconnected structures (e.g., left/right lungs).
        fg = (
            ((xx - 100) ** 2) / (52.0**2) + ((yy - 132) ** 2) / (34.0**2) <= 1.0
        ) | (
            ((xx - 148) ** 2) / (28.0**2) + ((yy - 110) ** 2) / (24.0**2) <= 1.0
        )
        bg[fg] = 156
        gt_mask = fg.astype(np.uint8)

        # Define prompt modes: (display_label, normalized_prompt_mode)
        # All modes must match normalize_prompt_mode() output from models/wrappers/prompt_utils.py
        modes = [
            ("point", "prompt_point"),
            ("bbox", "prompt_bbox"),
            ("point+bbox", "prompt_point_box"),  # Combination mode with both point and box
        ]
        fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2))

        for ax, (display_label, prompt_mode) in zip(axes, modes):
            # Resolve prompt using model's exact logic
            resolved = resolve_prompt(
                {"gt_mask": gt_mask},
                image_shape=(size, size),
                prompt_mode=prompt_mode,  # Must be one of: prompt_point, prompt_bbox, prompt_point_box
            )
            ax.imshow(bg, cmap="gray", vmin=0, vmax=255)

            # ─── Render bounding box if present ──────────────────────────────
            # prompt_bbox: always has bbox, no points
            # prompt_point: no bbox
            # prompt_point_box: always has bbox + one point
            bbox = resolved.get("bbox")
            if bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                rect = plt.Rectangle(
                    (x0, y0),
                    max(1, x1 - x0 + 1),
                    max(1, y1 - y0 + 1),
                    fill=False,
                    linewidth=2.0,
                    edgecolor="#E67E22",  # Orange for bbox
                    label="bbox" if prompt_mode in ("prompt_bbox", "prompt_point_box") else None,
                )
                ax.add_patch(rect)

            # ─── Render point(s) if present ─────────────────────────────────
            # prompt_point: exactly one point (clamped by resolve_prompt)
            # prompt_bbox: no points
            # prompt_point_box: exactly one point + bbox
            point = resolved.get("point")
            if point is None:
                # Fallback: extract first point from points array if available
                pts = resolved.get("points")
                if pts is not None and np.asarray(pts).size >= 2:
                    p0 = np.asarray(pts).reshape(-1, 2)[0]
                    point = (int(p0[0]), int(p0[1]))

            if point is not None:
                ax.scatter(
                    [int(point[0])],
                    [int(point[1])],
                    c="#1F77B4",  # Blue for point
                    s=40,
                    marker="o",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                    label="point" if prompt_mode in ("prompt_point", "prompt_point_box") else None,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(display_label, fontsize=12, fontweight="bold")
            ax.set_title(f"[{prompt_mode}]", fontsize=8, color="gray", style="italic")

        fig.tight_layout()
        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        return out_pdf

    # -- 2) noise gallery ------------------------------------------------------

    def _collect_noisy_images(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Path]]]]:
        """
        Returns nested mapping:
        dataset -> noise_type -> image_id -> noise_level -> noisy_path
        """
        mapping: Dict[str, Dict[str, Dict[str, Dict[str, Path]]]] = {}
        if self.artifact_root is None or not self.artifact_root.exists():
            return mapping

        # Shared layout: artifacts/_shared/{dataset}/{noise}/{level}/seed*/{image}_noisy.png
        for p in self.artifact_root.glob("_shared/*/*/*/seed*/*_noisy.png"):
            rel = p.relative_to(self.artifact_root)
            if len(rel.parts) < 6:
                continue
            _, dataset, noise_type, noise_level = rel.parts[:4]
            image_id = p.stem.replace("_noisy", "")
            mapping.setdefault(dataset, {}).setdefault(noise_type, {}).setdefault(image_id, {})[noise_level] = p

        # Legacy layout: artifacts/{dataset}/{model}/{prompt}/{noise}/{level}/seed*/{image}_noisy.png
        for p in self.artifact_root.glob("*/*/*/*/*/seed*/*_noisy.png"):
            rel = p.relative_to(self.artifact_root)
            if len(rel.parts) < 8:
                continue
            dataset, _, _, noise_type, noise_level = rel.parts[:5]
            image_id = p.stem.replace("_noisy", "")
            mapping.setdefault(dataset, {}).setdefault(noise_type, {}).setdefault(image_id, {})[noise_level] = p

        return mapping

    def plot_noise_gallery(self) -> Dict[str, Path]:
        out_paths: Dict[str, Path] = {}
        image_map = self._collect_noisy_images()

        for dataset in self._detected_datasets():
            ds_dir = self._dataset_dir(dataset)
            ds_slug = _slugify(dataset)
            out_pdf = ds_dir / f"gallery_noise_types_by_levels_dataset_{ds_slug}.pdf"
            out_paths[dataset] = out_pdf

            subset = self.df[self.df["dataset"] == dataset]
            noise_types = sorted(subset["noise_type"].dropna().astype(str).unique().tolist())
            levels = self._detected_levels(subset)
            n_rows, n_cols = max(1, len(noise_types)), max(1, len(levels))
            fig_w = max(7.5, 1.8 * n_cols)
            fig_h = max(5.0, 1.8 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
            axes = np.array(axes, dtype=object).reshape(n_rows, n_cols)

            ds_map = image_map.get(_slugify(dataset), image_map.get(dataset, {}))
            for r, noise in enumerate(noise_types):
                noise_map = ds_map.get(noise, {})
                best_image_id = None
                best_cov = -1
                for image_id, level_map in noise_map.items():
                    cov = sum(1 for lv in levels if lv in level_map)
                    if cov > best_cov:
                        best_cov = cov
                        best_image_id = image_id

                for c, lv in enumerate(levels):
                    ax = axes[r, c]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    shown = False
                    if best_image_id is not None:
                        path = noise_map.get(best_image_id, {}).get(lv)
                        if path is not None and path.exists():
                            img = _read_gray_image(path)
                            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                            shown = True
                    if not shown:
                        ax.imshow(np.full((64, 64), 220, dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
                        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=7)

                    if r == n_rows - 1:
                        ax.set_xlabel(str(lv))
                    if c == 0:
                        ax.set_ylabel(str(noise))

            fig.tight_layout()
            fig.savefig(out_pdf, format="pdf")
            plt.close(fig)

        return out_paths

    # -- 3) line plots metric vs level ----------------------------------------

    def plot_metric_vs_noise_level(self) -> Dict[str, List[Path]]:
        outputs: Dict[str, List[Path]] = {}
        palette = sns.color_palette("colorblind")

        for dataset in self._detected_datasets():
            sub_ds = self.df[self.df["dataset"] == dataset]
            ds_dir = self._dataset_dir(dataset)
            ds_slug = _slugify(dataset)
            outputs[dataset] = []

            for metric in self._detected_metrics():
                sub = sub_ds[sub_ds["metric_name"] == metric]
                if sub.empty:
                    continue
                metric_slug = _slugify(metric)
                out_pdf = ds_dir / (
                    f"line_metric_{metric_slug}_vs_noise_level_"
                    f"facet_noise_by_prompt_hue_model_dataset_{ds_slug}.pdf"
                )

                noises = sorted(sub["noise_type"].dropna().astype(str).unique().tolist())
                prompts = _sorted_prompt_modes(sub["prompt_mode"].dropna().astype(str).unique().tolist())
                levels = self._detected_levels(sub)
                models = sorted(sub["model"].dropna().astype(str).unique().tolist())
                if not noises or not prompts or not levels or not models:
                    continue

                agg = (
                    sub.groupby(
                        ["noise_type", "prompt_mode", "noise_level", "noise_level_idx", "model"],
                        dropna=False,
                    )["metric_value"]
                    .mean()
                    .reset_index()
                )

                n_rows, n_cols = len(noises), len(prompts)
                fig_w = max(9.0, 3.2 * n_cols)
                fig_h = max(5.0, 2.2 * n_rows)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharex=True, sharey=True)

                model_colors = {m: palette[i % len(palette)] for i, m in enumerate(models)}
                x_idx = {lv: i for i, lv in enumerate(levels)}
                figure_y_values: List[float] = []
                for r, noise in enumerate(noises):
                    for c, prompt in enumerate(prompts):
                        ax = axes[r, c]
                        cell = agg[(agg["noise_type"] == noise) & (agg["prompt_mode"] == prompt)]
                        for model in models:
                            g = cell[cell["model"] == model].sort_values(["noise_level_idx", "noise_level"])
                            if g.empty:
                                continue
                            xs = [x_idx[str(v)] for v in g["noise_level"].astype(str).tolist()]
                            ys = g["metric_value"].astype(float).tolist()
                            figure_y_values.extend(ys)
                            ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=3.0, color=model_colors[model], label=model)
                        ax.text(
                            0.02,
                            0.96,
                            f"noise={noise}\nprompt={_prompt_display(prompt)}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=6.8,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6),
                        )
                        if r == n_rows - 1:
                            ax.set_xlabel("noise level")
                        if c == 0:
                            ax.set_ylabel(metric)
                        ax.set_xticks(np.arange(len(levels)))
                        ax.set_xticklabels(levels, rotation=30, ha="right")
                        ax.grid(True, alpha=0.22, linewidth=0.5)

                if figure_y_values:
                    _set_line_y_axis(axes[0, 0], figure_y_values)

                handles = [plt.Line2D([0], [0], color=model_colors[m], marker="o", linewidth=1.3, markersize=3.0) for m in models]
                fig.legend(handles, models, loc="upper center", ncol=min(6, len(models)), frameon=False, bbox_to_anchor=(0.5, 1.0))
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(out_pdf, format="pdf")
                plt.close(fig)
                outputs[dataset].append(out_pdf)

        return outputs

    # -- 4) heatmaps -----------------------------------------------------------

    def _draw_heatmap(
        self,
        pivot: pd.DataFrame,
        *,
        out_pdf: Path,
        x_label: str,
        y_label: str,
        cbar_label: str,
    ) -> Optional[Path]:
        if pivot.empty:
            return None
        fig_w = max(6.5, 0.9 * pivot.shape[1] + 2.5)
        fig_h = max(4.2, 0.55 * pivot.shape[0] + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            pivot,
            cmap="YlGnBu",
            annot=True,
            fmt=".3f",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": cbar_label},
            ax=ax,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        fig.savefig(out_pdf, format="pdf")
        plt.close(fig)
        return out_pdf

    def plot_heatmaps(self) -> Dict[str, List[Path]]:
        outputs: Dict[str, List[Path]] = {}

        for dataset in self._detected_datasets():
            ds_dir = self._dataset_dir(dataset)
            ds_slug = _slugify(dataset)
            outputs[dataset] = []
            ds_df = self.df[self.df["dataset"] == dataset]

            for metric in self._detected_metrics():
                mdf = ds_df[ds_df["metric_name"] == metric]
                if mdf.empty:
                    continue
                metric_slug = _slugify(metric)
                prompts = _sorted_prompt_modes(mdf["prompt_mode"].dropna().astype(str).unique().tolist())
                prompt_groups = [(p, p) for p in prompts]

                for prompt_token, prompt_filter in prompt_groups:
                    sub = mdf if prompt_filter is None else mdf[mdf["prompt_mode"] == prompt_filter]
                    if sub.empty:
                        continue

                    # model vs noise_type (avg across levels)
                    p1 = (
                        sub.groupby(["model", "noise_type"], dropna=False)["metric_value"]
                        .mean()
                        .reset_index()
                        .pivot(index="model", columns="noise_type", values="metric_value")
                        .sort_index(axis=0)
                        .sort_index(axis=1)
                    )
                    out1 = ds_dir / (
                        f"heatmap_metric_{metric_slug}_rows_model_cols_noise_type_"
                        f"agg_levels_prompt_{_slugify(prompt_token)}_dataset_{ds_slug}.pdf"
                    )
                    saved1 = self._draw_heatmap(
                        p1,
                        out_pdf=out1,
                        x_label="noise type",
                        y_label="model",
                        cbar_label=metric,
                    )
                    if saved1 is not None:
                        outputs[dataset].append(saved1)

                    # model vs noise_level (avg across noise types)
                    p2 = (
                        sub.groupby(["model", "noise_level", "noise_level_idx"], dropna=False)["metric_value"]
                        .mean()
                        .reset_index()
                        .sort_values(["noise_level_idx", "noise_level"])
                        .pivot(index="model", columns="noise_level", values="metric_value")
                        .sort_index(axis=0)
                    )
                    out2 = ds_dir / (
                        f"heatmap_metric_{metric_slug}_rows_model_cols_noise_level_"
                        f"agg_noise_types_prompt_{_slugify(prompt_token)}_dataset_{ds_slug}.pdf"
                    )
                    saved2 = self._draw_heatmap(
                        p2,
                        out_pdf=out2,
                        x_label="noise level",
                        y_label="model",
                        cbar_label=metric,
                    )
                    if saved2 is not None:
                        outputs[dataset].append(saved2)

        return outputs

    # -- 5) prompt mode comparison --------------------------------------------

    def plot_prompt_mode_comparisons(self) -> Dict[str, List[Path]]:
        outputs: Dict[str, List[Path]] = {}
        palette_prompts = sns.color_palette("Set1")

        for dataset in self._detected_datasets():
            ds_df = self.df[self.df["dataset"] == dataset]
            ds_dir = self._dataset_dir(dataset)
            ds_slug = _slugify(dataset)
            outputs[dataset] = []

            for metric in self._detected_metrics():
                mdf = ds_df[ds_df["metric_name"] == metric]
                if mdf.empty:
                    continue
                metric_slug = _slugify(metric)
                out_pdf = ds_dir / (
                    f"prompt_mode_comparison_metric_{metric_slug}_"
                    f"models_noise_types_levels_dataset_{ds_slug}.pdf"
                )

                with PdfPages(out_pdf) as pdf:
                    # Page 1: prompt x model (mean over noise type + levels)
                    p1 = (
                        mdf.groupby(["prompt_mode_display", "model"], dropna=False)["metric_value"]
                        .mean()
                        .reset_index()
                    )
                    model_order = sorted(p1["model"].dropna().astype(str).unique().tolist())
                    model_palette = sns.color_palette("Set2", n_colors=max(1, len(model_order)))
                    fig1, ax1 = plt.subplots(figsize=(8.4, 4.2))
                    sns.barplot(
                        data=p1,
                        x="prompt_mode_display",
                        y="metric_value",
                        hue="model",
                        hue_order=model_order,
                        palette=model_palette,
                        ax=ax1,
                    )
                    ax1.set_xlabel("prompt mode")
                    ax1.set_ylabel(metric)
                    ax1.legend(frameon=False, ncol=min(4, max(1, p1["model"].nunique())))
                    ax1.grid(True, axis="y", alpha=0.25)
                    fig1.tight_layout()
                    pdf.savefig(fig1)
                    plt.close(fig1)

                    # Page 2: prompt x noise_type heatmap (mean over models + levels)
                    p2 = (
                        mdf.groupby(["prompt_mode_display", "noise_type"], dropna=False)["metric_value"]
                        .mean()
                        .reset_index()
                        .pivot(index="prompt_mode_display", columns="noise_type", values="metric_value")
                        .sort_index(axis=0)
                        .sort_index(axis=1)
                    )
                    fig2, ax2 = plt.subplots(figsize=(8.0, 3.6))
                    sns.heatmap(
                        p2,
                        cmap="YlOrBr",
                        annot=True,
                        fmt=".3f",
                        linewidths=0.4,
                        linecolor="white",
                        cbar_kws={"label": metric},
                        ax=ax2,
                    )
                    ax2.set_xlabel("noise type")
                    ax2.set_ylabel("prompt mode")
                    fig2.tight_layout()
                    pdf.savefig(fig2)
                    plt.close(fig2)

                    # Page 3: per-model trend across noise level (hue=prompt)
                    p3 = (
                        mdf.groupby(["model", "prompt_mode_display", "noise_level", "noise_level_idx"], dropna=False)[
                            "metric_value"
                        ]
                        .mean()
                        .reset_index()
                    )
                    models = sorted(p3["model"].dropna().astype(str).unique().tolist())
                    levels = sorted(p3["noise_level"].dropna().astype(str).unique().tolist(), key=_level_key)
                    prompts = sorted(p3["prompt_mode_display"].dropna().astype(str).unique().tolist())
                    if models and levels and prompts:
                        n_rows = len(models)
                        fig_h = max(2.4 * n_rows, 3.2)
                        fig3, axes = plt.subplots(n_rows, 1, figsize=(8.4, fig_h), squeeze=False, sharex=True, sharey=True)
                        prompt_colors = {p: palette_prompts[i % len(palette_prompts)] for i, p in enumerate(prompts)}
                        x_idx = {lv: i for i, lv in enumerate(levels)}
                        figure_y_values: List[float] = []

                        for r, model in enumerate(models):
                            ax = axes[r, 0]
                            sm = p3[p3["model"] == model]
                            for prompt in prompts:
                                g = sm[sm["prompt_mode_display"] == prompt].sort_values(["noise_level_idx", "noise_level"])
                                if g.empty:
                                    continue
                                xs = [x_idx[str(v)] for v in g["noise_level"].astype(str).tolist()]
                                ys = g["metric_value"].astype(float).tolist()
                                figure_y_values.extend(ys)
                                ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=3.0, color=prompt_colors[prompt], label=prompt)
                            ax.text(
                                0.01,
                                0.94,
                                f"model={model}",
                                transform=ax.transAxes,
                                ha="left",
                                va="top",
                                fontsize=7,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6),
                            )
                            ax.set_ylabel(metric)
                            ax.grid(True, alpha=0.25)

                        if figure_y_values:
                            _set_line_y_axis(axes[0, 0], figure_y_values)

                        axes[-1, 0].set_xlabel("noise level")
                        axes[-1, 0].set_xticks(np.arange(len(levels)))
                        axes[-1, 0].set_xticklabels(levels, rotation=30, ha="right")
                        handles = [plt.Line2D([0], [0], color=prompt_colors[p], marker="o", linewidth=1.3, markersize=3.0) for p in prompts]
                        fig3.legend(handles, prompts, loc="upper center", ncol=min(4, len(prompts)), frameon=False, bbox_to_anchor=(0.5, 1.0))
                        fig3.tight_layout(rect=[0, 0, 1, 0.96])
                        pdf.savefig(fig3)
                        plt.close(fig3)

                outputs[dataset].append(out_pdf)

        return outputs

    # -- orchestration ---------------------------------------------------------

    def generate_all(self) -> Dict[str, Path]:
        """
        Generate all required figures.
        Returns a flat mapping of output keys -> output paths.
        """
        outputs: Dict[str, Path] = {}
        outputs["prompt_schematic"] = self.plot_prompt_mode_schematic()

        noise_gallery = self.plot_noise_gallery()
        for ds, p in noise_gallery.items():
            outputs[f"noise_gallery_{_slugify(ds)}"] = p

        line_outputs = self.plot_metric_vs_noise_level()
        for ds, paths in line_outputs.items():
            for i, p in enumerate(paths):
                outputs[f"line_{_slugify(ds)}_{i:03d}"] = p

        heatmap_outputs = self.plot_heatmaps()
        for ds, paths in heatmap_outputs.items():
            for i, p in enumerate(paths):
                outputs[f"heatmap_{_slugify(ds)}_{i:03d}"] = p

        prompt_outputs = self.plot_prompt_mode_comparisons()
        for ds, paths in prompt_outputs.items():
            for i, p in enumerate(paths):
                outputs[f"prompt_comp_{_slugify(ds)}_{i:03d}"] = p

        return outputs


def generate_paper_visualizations(
    csv_path: Path,
    *,
    output_root: Path = Path("outputs/visualizations"),
    artifact_root: Optional[Path] = None,
) -> Dict[str, Path]:
    suite = PaperVisualizationSuite(csv_path, output_root=output_root, artifact_root=artifact_root)
    return suite.generate_all()
