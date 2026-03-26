"""
ModelComparisonPlotter – compares multiple models on the same noise type / dataset.
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
    return sorted(df["noise_level"].dropna().astype(str).unique().tolist(), key=_level_key)


class ModelComparisonPlotter:
    """One page per (dataset, prompt_mode, noise_type); one line per model."""

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

    def plot(self, metric: str = "Dice", filename: str | None = None) -> Path:
        df = self._df
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {self.stats_csv}.")

        levels = _sorted_levels(df)
        level_to_x = {lv: i for i, lv in enumerate(levels)}

        out_pdf = self.figures_dir / (filename or f"model_comparison_{metric.lower()}.pdf")
        with PdfPages(out_pdf) as pdf:
            groups = df.groupby(["dataset", "prompt_mode", "noise_type"], dropna=False)
            for (dataset, prompt_mode, noise_type), sub in groups:
                fig, ax = plt.subplots(figsize=(10, 5))
                for model, g in sub.groupby("model", dropna=False):
                    g = g.sort_values("noise_level", key=lambda s: s.map(_level_key))
                    xs = [level_to_x[str(v)] for v in g["noise_level"].astype(str)]
                    ys = g[metric].astype(float).tolist()
                    ax.plot(xs, ys, marker="o", linewidth=1.8, label=str(model))
                tick_labels = [format_level_label(lv, self._level_names) for lv in levels]
                ax.set_xticks(list(level_to_x.values()))
                ax.set_xticklabels(tick_labels, fontsize=7)
                ax.set_xlabel("Noise level")
                ax.set_ylabel(metric)
                ax.set_title(f"Model comparison | {dataset} | {prompt_mode} | {noise_type}")
                ax.grid(True, alpha=0.25)
                ax.set_ylim(bottom=0)
                ax.legend(loc="best")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf


# ── backwards-compat free function ──────────────────────────────────────

def generate_model_comparison(stats_csv: Path, out_pdf: Path, metric: str = "Dice") -> Path:
    plotter = ModelComparisonPlotter(stats_csv, out_pdf.parent)
    return plotter.plot(metric, out_pdf.name)
