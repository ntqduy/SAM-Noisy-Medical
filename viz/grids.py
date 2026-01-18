from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from viz.overlays import overlay


def _safe_read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).astype(np.uint8)


def _safe_read_mask(path: str) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L")).astype(np.uint8)
    return (m > 0).astype(np.uint8)


def save_preview_pdf(df: pd.DataFrame, cfg: dict, out_pdf: Path, num_samples: int, levels: List[str]):
    """
    preview: vài sample clean + noisy (L2/L4) overlay GT & prediction
    We pick rows from P1 for L2/L4 and P0 for clean.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    base = df[df["protocol"] == "P0"].copy()
    if len(base) == 0:
        return

    # sample ids
    ids = base["id"].dropna().unique().tolist()[:num_samples]

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H - 2*cm, "preview.pdf — Clean vs Noisy (L2/L4) with GT & Pred overlays")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H - 2.7*cm, f"Exp: {cfg['exp']['name']}")

    y = H - 4.0*cm

    def draw_panel(img_np, gt_np, pred_np, title: str, x0: float, y0: float, w: float, h: float):
        fig = plt.figure(figsize=(6, 2))
        ax = fig.add_subplot(1, 3, 1); ax.imshow(img_np, cmap="gray"); ax.axis("off"); ax.set_title("Image")
        ax = fig.add_subplot(1, 3, 2); ax.imshow(overlay(img_np, gt_np)); ax.axis("off"); ax.set_title("GT")
        ax = fig.add_subplot(1, 3, 3); ax.imshow(overlay(img_np, pred_np)); ax.axis("off"); ax.set_title("Pred")
        fig.suptitle(title, fontsize=10)
        tmp = out_pdf.parent / "_tmp_preview.png"
        fig.tight_layout()
        fig.savefig(tmp, dpi=160)
        plt.close(fig)
        c.drawImage(str(tmp), x0, y0, width=w, height=h, preserveAspectRatio=True, anchor="sw")

    panel_h = 4.2*cm
    panel_w = W - 4*cm

    for sid in ids:
        # pick baseline row
        r0 = base[base["id"] == sid].iloc[0]
        img = _safe_read_gray(r0["img_path"])
        gt = _safe_read_mask(r0["mask_path"])

        # pick a representative noisy row for L2/L4 (first found)
        rows_l2 = df[(df["id"] == sid) & (df["protocol"] == "P1") & (df["level"] == levels[1])]
        rows_l4 = df[(df["id"] == sid) & (df["protocol"] == "P1") & (df["level"] == levels[2])]

        if len(rows_l2) == 0 or len(rows_l4) == 0:
            continue

        r2 = rows_l2.iloc[0]
        r4 = rows_l4.iloc[0]

        # load pred masks saved on disk (if not found, skip)
        pred0_path = Path(cfg["exp"].get("out_root", "outputs")) / cfg["exp"]["name"] / "pred_masks" / r0["dataset"] / r0["model"] / r0["weight"] / r0["mode"] / r0["protocol"] / r0["noise"] / str(r0["level"]) / f"{sid}.png"
        pred2_path = Path(cfg["exp"].get("out_root", "outputs")) / cfg["exp"]["name"] / "pred_masks" / r2["dataset"] / r2["model"] / r2["weight"] / r2["mode"] / r2["protocol"] / r2["noise"] / str(r2["level"]) / f"{sid}.png"
        pred4_path = Path(cfg["exp"].get("out_root", "outputs")) / cfg["exp"]["name"] / "pred_masks" / r4["dataset"] / r4["model"] / r4["weight"] / r4["mode"] / r4["protocol"] / r4["noise"] / str(r4["level"]) / f"{sid}.png"

        if not pred0_path.exists() or not pred2_path.exists() or not pred4_path.exists():
            continue

        pred0 = _safe_read_mask(str(pred0_path))
        pred2 = _safe_read_mask(str(pred2_path))
        pred4 = _safe_read_mask(str(pred4_path))

        if y < 3.0*cm:
            c.showPage()
            y = H - 2.0*cm

        draw_panel(img, gt, pred0, f"{sid} — Clean (L0)", 2*cm, y - panel_h, panel_w, panel_h)
        y -= (panel_h + 0.3*cm)
        draw_panel(img, gt, pred2, f"{sid} — Noisy {levels[1]} (example)", 2*cm, y - panel_h, panel_w, panel_h)
        y -= (panel_h + 0.3*cm)
        draw_panel(img, gt, pred4, f"{sid} — Noisy {levels[2]} (example)", 2*cm, y - panel_h, panel_w, panel_h)
        y -= (panel_h + 0.6*cm)

    c.save()
    # Clean up temp file
    tmp = out_pdf.parent / "_tmp_preview.png"
    if tmp.exists():
        tmp.unlink()


def save_side_by_side_grid(
    df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    sample_ids: List[str],
    levels: List[str] = ["L0", "L2", "L4"],
    noise: str = "gaussian"
) -> List[str]:
    """
    Save side-by-side comparison grids: Clean vs L2 vs L4 for each sample.
    
    Returns list of saved image paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    exp_name = cfg["exp"]["name"]
    pred_root = Path(cfg["exp"].get("out_root", "outputs")) / exp_name / "pred_masks"
    
    for sid in sample_ids:
        # Find rows for this sample
        base = df[(df["id"] == sid) & (df["protocol"] == "P0")]
        if len(base) == 0:
            continue
        
        r0 = base.iloc[0]
        img = _safe_read_gray(r0["img_path"])
        gt = _safe_read_mask(r0["mask_path"])
        
        fig, axes = plt.subplots(len(levels), 4, figsize=(16, 4 * len(levels)))
        
        for i, lv in enumerate(levels):
            if lv == "L0":
                row = r0
                protocol = "P0"
                noise_name = "clean"
            else:
                rows = df[(df["id"] == sid) & (df["protocol"] == "P1") & (df["level"] == lv) & (df["noise"] == noise)]
                if len(rows) == 0:
                    continue
                row = rows.iloc[0]
                protocol = "P1"
                noise_name = noise
            
            # Load prediction
            pred_path = pred_root / row["dataset"] / row["model"] / row["weight"] / row["mode"] / protocol / noise_name / str(lv) / f"{sid}.png"
            if not pred_path.exists():
                continue
            
            pred = _safe_read_mask(str(pred_path))
            
            # Plot
            ax_row = axes[i] if len(levels) > 1 else axes
            ax_row[0].imshow(img, cmap="gray")
            ax_row[0].set_title(f"{lv}: Image")
            ax_row[0].axis("off")
            
            ax_row[1].imshow(overlay(img, gt, color=(0, 255, 0)))
            ax_row[1].set_title(f"{lv}: GT")
            ax_row[1].axis("off")
            
            ax_row[2].imshow(overlay(img, pred, color=(255, 0, 0)))
            ax_row[2].set_title(f"{lv}: Pred")
            ax_row[2].axis("off")
            
            # Overlay both
            combined = overlay(img, gt, alpha=0.3, color=(0, 255, 0))
            combined = overlay(combined[:, :, 1], pred, alpha=0.3, color=(255, 0, 0))
            ax_row[3].imshow(combined)
            ax_row[3].set_title(f"{lv}: GT (green) + Pred (red)")
            ax_row[3].axis("off")
        
        plt.suptitle(f"Sample: {sid} — {noise}")
        plt.tight_layout()
        
        out_path = out_dir / f"grid_{sid}_{noise}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        paths.append(str(out_path))
    
    return paths


def create_noise_comparison_grid(
    images: Dict[str, np.ndarray],
    gt: np.ndarray,
    predictions: Dict[str, np.ndarray],
    noise_levels: List[str],
    title: str = ""
) -> np.ndarray:
    """
    Create a comparison grid showing image and prediction across noise levels.
    
    Args:
        images: Dict of level -> noisy image
        gt: Ground truth mask
        predictions: Dict of level -> prediction mask
        noise_levels: List of level names
        title: Optional title
        
    Returns:
        RGB image array of the grid
    """
    n_levels = len(noise_levels)
    fig, axes = plt.subplots(2, n_levels, figsize=(4 * n_levels, 8))
    
    for i, lv in enumerate(noise_levels):
        if lv in images:
            axes[0, i].imshow(images[lv], cmap="gray")
            axes[0, i].set_title(f"{lv}: Image")
        axes[0, i].axis("off")
        
        if lv in predictions:
            combined = overlay(images.get(lv, np.zeros_like(gt)), predictions[lv], color=(255, 0, 0))
            combined = np.maximum(combined, overlay(images.get(lv, np.zeros_like(gt)), gt, alpha=0.2, color=(0, 255, 0)))
            axes[1, i].imshow(combined)
            axes[1, i].set_title(f"{lv}: Pred (red) + GT (green)")
        axes[1, i].axis("off")
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return img
