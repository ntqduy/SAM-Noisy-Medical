"""
NoiseGalleryGenerator – shows how a single image degrades across noise levels.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

from viz import DEFAULT_LEVEL_NAMES, format_level_label


def _level_key(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _collect_artifacts(artifact_root: Path) -> Dict[Tuple[str, str, str, str, str], Dict[str, Path]]:
    mapping: Dict[Tuple[str, str, str, str, str], Dict[str, Path]] = {}
    if not artifact_root.exists():
        return mapping
    for noisy_path in artifact_root.glob("*/*/*/*/*/seed*/*_noisy.png"):
        rel = noisy_path.relative_to(artifact_root)
        if len(rel.parts) < 8:
            continue
        dataset, model, prompt_mode, noise_type, noise_level = rel.parts[:5]
        image_id = noisy_path.stem.replace("_noisy", "")
        key = (dataset, model, prompt_mode, noise_type, image_id)
        mapping.setdefault(key, {})[noise_level] = noisy_path
    return mapping


class NoiseGalleryGenerator:
    """Generates a PDF gallery showing progressive noise degradation."""

    def __init__(
        self,
        artifact_root: Path,
        figures_dir: Path,
        level_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._level_names = level_names or DEFAULT_LEVEL_NAMES

    def generate(self, max_rows: int = 12, filename: str | None = None) -> Path:
        gallery_map = _collect_artifacts(self.artifact_root)
        out_pdf = self.figures_dir / (filename or "noise_gallery.pdf")

        with PdfPages(out_pdf) as pdf:
            if not gallery_map:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.text(0.5, 0.5, "No stage1 artifacts found.\nEnable stage1.save_artifacts.",
                        ha="center", va="center")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
                return out_pdf

            shown = 0
            for key, level_map in sorted(gallery_map.items()):
                if shown >= max_rows:
                    break
                dataset, model, prompt_mode, noise_type, image_id = key
                levels = sorted(level_map.keys(), key=_level_key)
                if not levels:
                    continue

                fig, axes = plt.subplots(1, len(levels), figsize=(max(3 * len(levels), 8), 3))
                if len(levels) == 1:
                    axes = [axes]
                for ax, level in zip(axes, levels):
                    img = np.asarray(Image.open(level_map[level]).convert("L"))
                    ax.imshow(img, cmap="gray")
                    ax.set_title(format_level_label(level, self._level_names), fontsize=8)
                    ax.axis("off")
                fig.suptitle(
                    f"Noise gallery | {dataset} | {model} | {prompt_mode} | {noise_type} | {image_id}"
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                shown += 1
        return out_pdf


# ── backwards-compat free function ──────────────────────────────────────

def generate_noise_gallery(artifact_root: Path, out_pdf: Path, *, max_rows: int = 12) -> Path:
    gen = NoiseGalleryGenerator(artifact_root, out_pdf.parent)
    return gen.generate(max_rows=max_rows, filename=out_pdf.name)

