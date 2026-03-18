"""
PromptComparisonPlotter – schematic prompt diagrams and prompt-mode bar comparisons.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _synthetic_image(size: int = 256) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    r = size // 4
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img = np.full((size, size), 70, dtype=np.uint8)
    img[mask] = 150
    img = img + np.linspace(0, 35, size, dtype=np.uint8)[None, :]
    return np.clip(img, 0, 255)


class PromptComparisonPlotter:
    """Generates prompt-schematic diagrams and prompt-mode bar comparisons."""

    def __init__(self, figures_dir: Path, stats_csv: Path | None = None) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.stats_csv = Path(stats_csv) if stats_csv else None
        self._df = pd.read_csv(self.stats_csv) if self.stats_csv else None

    def plot_schematic(self, filename: str | None = None) -> Path:
        """One-page PDF with schematic prompt-mode diagrams."""
        out_pdf = self.figures_dir / (filename or "prompt_visualization.pdf")
        img = _synthetic_image()
        h, w = img.shape
        cx, cy = w // 2, h // 2
        x0, y0, x1, y1 = w // 4, h // 4, (3 * w) // 4, (3 * h) // 4

        with PdfPages(out_pdf) as pdf:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            titles = ["prompt_point", "prompt_bbox", "prompt_point_box"]
            for ax, title in zip(axes, titles):
                ax.imshow(img, cmap="gray")
                if "point" in title:
                    ax.scatter([cx], [cy], c="lime", s=40, marker="o")
                if "bbox" in title:
                    rect = patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor="yellow", linewidth=2,
                    )
                    ax.add_patch(rect)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle("Prompt modes (schematic)", fontsize=12)
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
                ax.set_title(f"Prompt comparison ({metric}) | {dataset} | {model}")
                ax.set_ylabel(metric)
                ax.set_xlabel("Prompt mode")
                ax.grid(True, axis="y", alpha=0.25)
                ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf


# ── backwards-compat free functions ─────────────────────────────────────

def generate_prompt_visualization(out_pdf: Path) -> Path:
    plotter = PromptComparisonPlotter(out_pdf.parent)
    return plotter.plot_schematic(out_pdf.name)


def generate_prompt_comparison(stats_csv: Path, out_pdf: Path, metric: str = "Dice") -> Path:
    plotter = PromptComparisonPlotter(out_pdf.parent, stats_csv)
    return plotter.plot_comparison(metric, out_pdf.name)

