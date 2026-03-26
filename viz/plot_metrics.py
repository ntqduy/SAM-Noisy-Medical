"""
MetricPlotter – OOP wrapper for metric-vs-noise-level curves and robustness bar charts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from viz import DEFAULT_LEVEL_NAMES, format_level_label


def _level_key(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _sorted_levels(df: pd.DataFrame) -> List[str]:
    levels = sorted(df["noise_level"].dropna().astype(str).unique().tolist(), key=_level_key)
    return levels


class MetricPlotter:
    """Generates metric-vs-level curves and robustness drop bar charts as PDF."""

    def __init__(
        self,
        stats_csv: Path,
        figures_dir: Path,
        level_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self.stats_csv = Path(stats_csv)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._df = pd.read_csv(self.stats_csv)
        self._level_names = level_names or DEFAULT_LEVEL_NAMES

    def plot_metric_curves(self, metric: str = "Dice", filename: str | None = None) -> Path:
        """One page per (dataset, model, prompt_mode); lines per noise type."""
        df = self._df
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {self.stats_csv}.")

        levels = _sorted_levels(df)
        level_to_x = {lv: i for i, lv in enumerate(levels)}

        out_pdf = self.figures_dir / (filename or f"metric_curves_{metric.lower()}.pdf")
        with PdfPages(out_pdf) as pdf:
            grouped = df.groupby(["dataset", "model", "prompt_mode"], dropna=False)
            for (dataset, model, prompt_mode), sub in grouped:
                fig, ax = plt.subplots(figsize=(10, 5))
                for noise_type, g in sub.groupby("noise_type", dropna=False):
                    g = g.sort_values("noise_level", key=lambda s: s.map(_level_key))
                    xs = [level_to_x[str(v)] for v in g["noise_level"].astype(str)]
                    ys = g[metric].astype(float).tolist()
                    ax.plot(xs, ys, marker="o", linewidth=1.8, label=str(noise_type))
                tick_labels = [format_level_label(lv, self._level_names) for lv in levels]
                ax.set_xticks(list(level_to_x.values()))
                ax.set_xticklabels(tick_labels, rotation=0, fontsize=7)
                ax.set_xlabel("Noise level")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.25)
                ax.set_ylim(bottom=0)
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf

    def plot_robustness(self, metric: str = "Dice", filename: str | None = None) -> Path:
        """Bar chart of metric drop (L0 → Lmax) per noise type."""
        df = self._df
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {self.stats_csv}.")

        rows = []
        for keys, g in df.groupby(["dataset", "model", "prompt_mode", "noise_type"], dropna=False):
            dataset, model, prompt_mode, noise_type = keys
            g = g.sort_values("noise_level", key=lambda s: s.map(_level_key))
            if len(g) == 0:
                continue
            l0 = g.iloc[0][metric]
            lmax = g.iloc[-1][metric]
            rows.append({
                "dataset": dataset,
                "model": model,
                "prompt_mode": prompt_mode,
                "noise_type": noise_type,
                f"{metric}_drop": float(l0 - lmax),
            })

        drop_df = pd.DataFrame(rows)
        out_pdf = self.figures_dir / (filename or f"robustness_{metric.lower()}_drop.pdf")
        with PdfPages(out_pdf) as pdf:
            if drop_df.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, "No robustness rows available", ha="center", va="center")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
                return out_pdf

            grouped = drop_df.groupby(["dataset", "model", "prompt_mode"], dropna=False)
            for (dataset, model, prompt_mode), sub in grouped:
                sub = sub.sort_values(f"{metric}_drop", ascending=False)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(sub["noise_type"].astype(str), sub[f"{metric}_drop"].astype(float))
                ax.set_ylabel(f"{metric}_drop (L0 - Lmax)")
                ax.set_xlabel("Noise type")
                ax.tick_params(axis="x", rotation=40)
                ax.grid(True, axis="y", alpha=0.25)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf


# ── backwards-compat free functions ─────────────────────────────────────

def generate_metric_curves(stats_csv: Path, out_pdf: Path, metric: str = "Dice") -> Path:
    plotter = MetricPlotter(stats_csv, out_pdf.parent)
    return plotter.plot_metric_curves(metric, out_pdf.name)


def generate_robustness_plot(stats_csv: Path, out_pdf: Path, metric: str = "Dice") -> Path:
    plotter = MetricPlotter(stats_csv, out_pdf.parent)
    return plotter.plot_robustness(metric, out_pdf.name)
