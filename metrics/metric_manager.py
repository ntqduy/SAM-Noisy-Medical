"""
MetricManager – single entry-point for computing all segmentation metrics.

Computes: IoU, Dice, Recall, Precision, F1, Hausdorff Distance.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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
    g = _as_bool(gt)
    if not g.any():
        return float("nan")
    c = _binary_counts(pred, gt)
    union = c["tp"] + c["fp"] + c["fn"]
    return 1.0 if union == 0 else float(c["tp"] / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    g = _as_bool(gt)
    if not g.any():
        return float("nan")
    c = _binary_counts(pred, gt)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return 1.0 if denom == 0 else float(2 * c["tp"] / denom)


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    g = _as_bool(gt)
    if not g.any():
        return float("nan")
    c = _binary_counts(pred, gt)
    denom = c["tp"] + c["fn"]
    return 1.0 if denom == 0 else float(c["tp"] / denom)


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    g = _as_bool(gt)
    if not g.any():
        return float("nan")
    c = _binary_counts(pred, gt)
    denom = c["tp"] + c["fp"]
    # When pred is empty (denom=0), return 0.0 (no correct predictions)
    # instead of 1.0 which would misleadingly suggest perfect precision
    return 0.0 if denom == 0 else float(c["tp"] / denom)


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    g = _as_bool(gt)
    if not g.any():
        return float("nan")
    p = precision_score(pred, gt)
    r = recall_score(pred, gt)
    denom = p + r
    return 0.0 if denom == 0 else float(2.0 * p * r / denom)


def _normalize_spacing(
    spacing: Optional[Sequence[float]],
    ndim: int,
) -> Optional[Tuple[float, ...]]:
    if spacing is None:
        return None
    try:
        vals = [float(v) for v in spacing]
    except (TypeError, ValueError):
        return None
    if not vals or any(not np.isfinite(v) or v <= 0 for v in vals):
        return None
    if len(vals) >= ndim:
        return tuple(vals[-ndim:])
    if len(vals) == 1:
        return tuple([vals[0]] * ndim)
    return None


def _surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    directed = _directed_surface_distances(pred, gt, spacing=spacing)
    if directed is None:
        return None
    d_pred_to_gt, d_gt_to_pred = directed
    return np.concatenate(
        [d_pred_to_gt.astype(np.float64), d_gt_to_pred.astype(np.float64)],
        axis=0,
    )


def _directed_surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        return None

    p, g = _as_bool(pred), _as_bool(gt)
    # Empty masks return NaN except the both-empty case, which is a perfect match.
    if p.sum() == 0 and g.sum() == 0:
        vals = np.asarray([0.0], dtype=np.float64)
        return vals, vals
    if p.sum() == 0 or g.sum() == 0:
        vals = np.asarray([float("nan")], dtype=np.float64)
        return vals, vals

    p_s = np.logical_xor(p, binary_erosion(p))
    g_s = np.logical_xor(g, binary_erosion(g))
    if not p_s.any() or not g_s.any():
        vals = np.asarray([0.0], dtype=np.float64)
        return vals, vals

    sampling = _normalize_spacing(spacing, p_s.ndim)
    dt_p = distance_transform_edt(~p_s, sampling=sampling)
    dt_g = distance_transform_edt(~g_s, sampling=sampling)
    d_pred_to_gt = dt_g[p_s]
    d_gt_to_pred = dt_p[g_s]
    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        vals = np.asarray([0.0], dtype=np.float64)
        return vals, vals
    return d_pred_to_gt.astype(np.float64), d_gt_to_pred.astype(np.float64)


def hausdorff_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Optional[float]:
    distances = _surface_distances(pred, gt, spacing=spacing)
    if distances is None:
        return None
    if not np.isfinite(distances).all():
        return float(distances[0])
    return float(distances.max(initial=0.0))


def hausdorff95_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Optional[float]:
    directed = _directed_surface_distances(pred, gt, spacing=spacing)
    if directed is None:
        return None
    d_pred_to_gt, d_gt_to_pred = directed
    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    if not np.isfinite(distances).all():
        return float(distances[0])
    # HD95 uses pooled bidirectional surface distances.
    return float(np.percentile(distances, 95))


# ── MetricManager ────────────────────────────────────────────────────────

class MetricManager:
    """Compute all segmentation metrics in one call."""

    @staticmethod
    def compute(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        *,
        spacing: Optional[Sequence[float]] = None,
        add_hd95: bool = False,
        add_physical_distance: bool = False,
        keep_legacy_hd: bool = True,
    ) -> Dict[str, float]:
        """
        Return a dict with keys:
        ``IoU``, ``Dice``, ``Recall``, ``Precision``, ``F1``, ``HD``.
        """
        hd = hausdorff_distance(pred_mask, gt_mask)
        hd_value = hd if hd is not None else float("nan")
        out = {
            "IoU": iou(pred_mask, gt_mask),
            "Dice": dice(pred_mask, gt_mask),
            "Recall": recall_score(pred_mask, gt_mask),
            "Precision": precision_score(pred_mask, gt_mask),
            "F1": f1_score(pred_mask, gt_mask),
            "HD": hd_value,
        }
        if not keep_legacy_hd:
            out.pop("HD", None)

        if add_hd95:
            hd95 = hausdorff95_distance(pred_mask, gt_mask)
            out["HD_px"] = hd_value
            out["HD95_px"] = hd95 if hd95 is not None else float("nan")

        if add_physical_distance:
            if spacing is None:
                out["HD_mm"] = float("nan")
                out["HD95_mm"] = float("nan")
            else:
                hd_mm = hausdorff_distance(pred_mask, gt_mask, spacing=spacing)
                hd95_mm = hausdorff95_distance(pred_mask, gt_mask, spacing=spacing)
                out["HD_mm"] = hd_mm if hd_mm is not None else float("nan")
                out["HD95_mm"] = hd95_mm if hd95_mm is not None else float("nan")

        return out
