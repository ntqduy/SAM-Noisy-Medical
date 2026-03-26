"""
PromptComparisonPlotter - prompt diagrams and prompt-mode bar comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from models.wrappers.prompt_utils import resolve_prompt


def _synthetic_image(size: int = 256) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = size // 4
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img = np.full((size, size), 70, dtype=np.uint8)
    img[mask] = 150
    img = img + np.linspace(0, 35, size, dtype=np.uint8)[None, :]
    return np.clip(img, 0, 255)


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower()).strip("_") or "item"


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


class PromptComparisonPlotter:
    """Generates prompt-schematic diagrams and prompt-mode bar comparisons."""

    def __init__(
        self,
        figures_dir: Path,
        stats_csv: Path | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.stats_csv = Path(stats_csv) if stats_csv else None
        self._df = pd.read_csv(self.stats_csv) if self.stats_csv else None
        if artifact_root is not None:
            self.artifact_root = Path(artifact_root)
        elif self.stats_csv is not None:
            cand = self.stats_csv.parent / "artifacts"
            self.artifact_root = cand if cand.exists() else None
        else:
            self.artifact_root = None

    def _detected_datasets(self) -> List[str]:
        if self._df is None or "dataset" not in self._df.columns:
            return []
        return sorted(self._df["dataset"].dropna().astype(str).str.strip().unique().tolist())

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
                if gt_path.exists():
                    return {
                        "sample_id": sample_id,
                        "original_path": original_path,
                        "gt_path": gt_path,
                    }
        return None

    def plot_schematic(self, filename: str | None = None) -> Path:
        """One-page-per-dataset PDF with prompt-mode diagrams."""
        out_pdf = self.figures_dir / (filename or "prompt_visualization.pdf")
        modes = [
            ("point", "prompt_point"),
            ("bbox", "prompt_bbox"),
            ("point+bbox", "prompt_point_box"),
        ]

        with PdfPages(out_pdf) as pdf:
            pages_written = 0

            for dataset in self._detected_datasets():
                sample = self._find_prompt_sample(dataset)
                if sample is None:
                    continue

                image = _read_uint8_image(Path(sample["original_path"]))
                gt_mask = _read_binary_mask(Path(sample["gt_path"]))
                if image.shape[:2] != gt_mask.shape[:2]:
                    continue

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, (label, prompt_mode) in zip(axes, modes):
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
                            edgecolor="yellow",
                            linewidth=2,
                        )
                        ax.add_patch(rect)

                    pts = resolved.get("points")
                    if pts is not None and np.asarray(pts).size > 0:
                        pts_arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                        ax.scatter(
                            pts_arr[:, 0],
                            pts_arr[:, 1],
                            c="lime",
                            s=40,
                            marker="o",
                            edgecolors="white",
                            linewidths=0.8,
                            zorder=5,
                        )

                    ax.text(0.5, -0.06, label, transform=ax.transAxes, ha="center", va="top", fontsize=10)
                    ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                pages_written += 1

            if pages_written == 0:
                image = _synthetic_image()
                gt_mask = (image > image.mean()).astype(np.uint8)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, (label, prompt_mode) in zip(axes, modes):
                    resolved = resolve_prompt(
                        {"gt_mask": gt_mask},
                        image.shape[:2],
                        prompt_mode=prompt_mode,
                    )
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                    bbox = resolved.get("bbox")
                    if bbox is not None:
                        x0, y0, x1, y1 = [int(v) for v in bbox]
                        rect = plt.Rectangle(
                            (x0, y0),
                            max(1, x1 - x0 + 1),
                            max(1, y1 - y0 + 1),
                            fill=False,
                            edgecolor="yellow",
                            linewidth=2,
                        )
                        ax.add_patch(rect)
                    pts = resolved.get("points")
                    if pts is not None and np.asarray(pts).size > 0:
                        pts_arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                        ax.scatter(
                            pts_arr[:, 0],
                            pts_arr[:, 1],
                            c="lime",
                            s=40,
                            marker="o",
                            edgecolors="white",
                            linewidths=0.8,
                            zorder=5,
                        )
                    ax.text(0.5, -0.06, label, transform=ax.transAxes, ha="center", va="top", fontsize=10)
                    ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf

    def plot_comparison(self, metric: str = "Dice", filename: str | None = None) -> Path:
        """Bar chart comparing prompt modes on *metric*."""
        if self._df is None:
            raise RuntimeError("stats_csv required for prompt comparison.")
        df = self._df
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found.")

        out_pdf = self.figures_dir / (filename or f"prompt_comparison_{metric.lower()}.pdf")
        with PdfPages(out_pdf) as pdf:
            grouped = df.groupby(["dataset", "model"], dropna=False)
            for (dataset, model), sub in grouped:
                summary = (
                    sub.groupby("prompt_mode", dropna=False)[metric]
                    .mean()
                    .reset_index()
                    .sort_values(metric, ascending=False)
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(summary["prompt_mode"].astype(str), summary[metric].astype(float))
                ax.set_ylabel(metric)
                ax.set_xlabel("Prompt mode")
                ax.grid(True, axis="y", alpha=0.25)
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf


def generate_prompt_visualization(
    out_pdf: Path,
    *,
    stats_csv: Path | None = None,
    artifact_root: Path | None = None,
) -> Path:
    plotter = PromptComparisonPlotter(out_pdf.parent, stats_csv, artifact_root)
    return plotter.plot_schematic(out_pdf.name)


def generate_prompt_comparison(stats_csv: Path, out_pdf: Path, metric: str = "Dice") -> Path:
    plotter = PromptComparisonPlotter(out_pdf.parent, stats_csv)
    return plotter.plot_comparison(metric, out_pdf.name)
