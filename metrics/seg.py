from typing import Optional
import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    inter = (p & g).sum()
    uni = (p | g).sum()
    if uni == 0:
        return 1.0
    return float(inter / uni)


def hd95(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except Exception:
        return None

    p = pred.astype(bool); g = gt.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")

    p_s = p ^ binary_erosion(p)
    g_s = g ^ binary_erosion(g)

    dt_p = distance_transform_edt(~p_s)
    dt_g = distance_transform_edt(~g_s)

    d1 = dt_g[p_s]
    d2 = dt_p[g_s]
    d = np.concatenate([d1, d2]).astype(np.float32)
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, 95.0))
