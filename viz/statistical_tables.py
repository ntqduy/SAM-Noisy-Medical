"""
StatisticalTableGenerator – perturbation-robustness, prompt-comparison,
and dataset-performance table PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from viz import DEFAULT_LEVEL_NAMES


def _level_key(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(14, 0.35 * min(max_rows, len(df)) + 2.5))
    ax.set_title(title, loc="left")
    ax.axis("off")
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    show_df = df.head(max_rows).copy()
    show_df = show_df.round(4)
    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


_METRIC_COLS = ["Dice", "IoU", "Recall", "Precision", "F1", "HD"]


class StatisticalTableGenerator:
    """Generates paper-style statistical table PDFs from merged statistics."""

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

    def generate(self, filename: str | None = None) -> Path:
        """Create a multi-page PDF with three statistical tables."""
        out_pdf = self.figures_dir / (filename or "statistical_tables.pdf")
        df = self._df

        # Add descriptive level label column
        if "noise_level" in df.columns:
            df = df.copy()
            df["level_desc"] = df["noise_level"].astype(str).map(
                lambda lv: self._level_names.get(lv, "")
            )

        with PdfPages(out_pdf) as pdf:
            # Table 1: Perturbation robustness (Dice drop L0→Lmax).
            robust_rows = []
            for keys, g in df.groupby(
                ["dataset", "model", "prompt_mode", "noise_type"], dropna=False
            ):
                dataset, model, prompt_mode, noise_type = keys
                g = g.sort_values("noise_level", key=lambda s: s.map(_level_key))
                if len(g) < 2 or "Dice" not in g.columns:
                    continue
                robust_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "prompt_mode": prompt_mode,
                        "noise_type": noise_type,
                        "Dice_L0": float(g.iloc[0]["Dice"]),
                        "Dice_Lmax": float(g.iloc[-1]["Dice"]),
                        "Dice_drop": float(g.iloc[0]["Dice"] - g.iloc[-1]["Dice"]),
                    }
                )
            robust_df = pd.DataFrame(robust_rows).sort_values("Dice_drop", ascending=False)
            _add_table_page(pdf, "Perturbation Robustness Table", robust_df)

            # Table 2: Prompt comparison.
            prompt_cols = [c for c in _METRIC_COLS if c in df.columns]
            prompt_df = (
                df.groupby(["dataset", "model", "prompt_mode"], dropna=False)[prompt_cols]
                .mean()
                .reset_index()
                .sort_values(
                    ["dataset", "model", "Dice"],
                    ascending=[True, True, False] if "Dice" in prompt_cols else True,
                )
            )
            _add_table_page(pdf, "Prompt Comparison Table", prompt_df)

            # Table 3: Dataset performance.
            dataset_df = (
                df.groupby(["dataset", "model"], dropna=False)[prompt_cols]
                .mean()
                .reset_index()
                .sort_values(["dataset", "model"])
            )
            _add_table_page(pdf, "Dataset Performance Table", dataset_df)
        return out_pdf


# ── backwards-compat free function ──────────────────────────────────────

def generate_statistical_tables(stats_csv: Path, out_pdf: Path) -> Path:
    gen = StatisticalTableGenerator(stats_csv, out_pdf.parent)
    return gen.generate(out_pdf.name)

