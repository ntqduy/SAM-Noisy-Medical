"""
PredictionVisualizer – side-by-side overlays of predictions vs ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages


def _collect_cases(artifact_root: Path) -> List[Tuple[Path, Path, Path, Path]]:
    cases: List[Tuple[Path, Path, Path, Path]] = []
    for pred_path in artifact_root.glob("*/*/*/*/*/seed*/*_pred.png"):
        base = pred_path.with_name(pred_path.name.replace("_pred.png", ""))
        original = base.with_name(base.name + "_original.png")
        noisy = base.with_name(base.name + "_noisy.png")
        gt = base.with_name(base.name + "_gt.png")
        if original.exists() and noisy.exists() and gt.exists():
            cases.append((original, noisy, gt, pred_path))
    return sorted(cases)


def _overlay_mask(
    gray_img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float = 0.35,
) -> np.ndarray:
    base = np.stack([gray_img, gray_img, gray_img], axis=-1).astype(np.float32) / 255.0
    m = (mask > 0).astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out = base * (1.0 - alpha * m) + color_arr * (alpha * m)
    return np.clip(out, 0, 1)


class PredictionVisualizer:
    """Generates a multi-page PDF showing original / noisy / GT overlay / pred overlay."""

    def __init__(self, artifact_root: Path, figures_dir: Path) -> None:
        self.artifact_root = Path(artifact_root)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, max_cases: int = 20, filename: str | None = None) -> Path:
        cases = _collect_cases(self.artifact_root)
        out_pdf = self.figures_dir / (filename or "prediction_overlay.pdf")

        with PdfPages(out_pdf) as pdf:
            if not cases:
                fig, ax = plt.subplots(figsize=(9, 3))
                ax.text(0.5, 0.5, "No prediction artifacts found.", ha="center", va="center")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
                return out_pdf

            for original_p, noisy_p, gt_p, pred_p in cases[:max_cases]:
                original = np.asarray(Image.open(original_p).convert("L"))
                noisy = np.asarray(Image.open(noisy_p).convert("L"))
                gt = np.asarray(Image.open(gt_p).convert("L"))
                pred = np.asarray(Image.open(pred_p).convert("L"))

                gt_overlay = _overlay_mask(noisy, gt, color=(1.0, 0.0, 0.0), alpha=0.4)
                pred_overlay = _overlay_mask(noisy, pred, color=(0.0, 1.0, 1.0), alpha=0.4)

                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(original, cmap="gray"); axes[0].set_title("Original")
                axes[1].imshow(noisy, cmap="gray"); axes[1].set_title("Noisy")
                axes[2].imshow(gt_overlay); axes[2].set_title("Ground Truth")
                axes[3].imshow(pred_overlay); axes[3].set_title("Prediction")
                for ax in axes:
                    ax.axis("off")
                fig.suptitle(str(pred_p.relative_to(self.artifact_root)))
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return out_pdf


# ── backwards-compat free function ──────────────────────────────────────

def generate_prediction_overlays(artifact_root: Path, out_pdf: Path, *, max_cases: int = 20) -> Path:
    vis = PredictionVisualizer(artifact_root, out_pdf.parent)
    return vis.generate(max_cases=max_cases, filename=out_pdf.name)

