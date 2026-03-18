"""
MetricManager – single entry-point for computing all segmentation metrics.

Computes: IoU, Dice, Recall, Precision, F1, Hausdorff Distance.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def _binary_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    p, g = _as_bool(pred), _as_bool(gt)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


# ── individual metrics ──────────────────────────────────────────────────

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    c = _binary_counts(pred, gt)
    union = c["tp"] + c["fp"] + c["fn"]
    return 1.0 if union == 0 else float(c["tp"] / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    c = _binary_counts(pred, gt)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return 1.0 if denom == 0 else float(2 * c["tp"] / denom)


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    c = _binary_counts(pred, gt)
    denom = c["tp"] + c["fn"]
    return 1.0 if denom == 0 else float(c["tp"] / denom)


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    c = _binary_counts(pred, gt)
    denom = c["tp"] + c["fp"]
    return 1.0 if denom == 0 else float(c["tp"] / denom)


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = precision_score(pred, gt)
    r = recall_score(pred, gt)
    denom = p + r
    return 0.0 if denom == 0 else float(2.0 * p * r / denom)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        return None

    p, g = _as_bool(pred), _as_bool(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")

    p_s = np.logical_xor(p, binary_erosion(p))
    g_s = np.logical_xor(g, binary_erosion(g))
    if not p_s.any() or not g_s.any():
        return 0.0

    dt_p = distance_transform_edt(~p_s)
    dt_g = distance_transform_edt(~g_s)
    return float(max(float(dt_g[p_s].max(initial=0.0)), float(dt_p[g_s].max(initial=0.0))))


# ── MetricManager ────────────────────────────────────────────────────────

class MetricManager:
    """Compute all segmentation metrics in one call."""

    @staticmethod
    def compute(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Return a dict with keys:
        ``IoU``, ``Dice``, ``Recall``, ``Precision``, ``F1``, ``HD``.
        """
        hd = hausdorff_distance(pred_mask, gt_mask)
        return {
            "IoU": iou(pred_mask, gt_mask),
            "Dice": dice(pred_mask, gt_mask),
            "Recall": recall_score(pred_mask, gt_mask),
            "Precision": precision_score(pred_mask, gt_mask),
            "F1": f1_score(pred_mask, gt_mask),
            "HD": hd if hd is not None else float("nan"),
        }
