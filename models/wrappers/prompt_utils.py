"""
Prompt helper utilities – build standard prompt payloads from ground-truth masks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def normalize_prompt_mode(prompt_mode: str) -> str:
    pm = str(prompt_mode or "").strip().lower()
    mapping = {
        "bbox": "prompt_bbox", "box": "prompt_bbox", "prompt_bbox": "prompt_bbox",
        "point": "prompt_point", "prompt_point": "prompt_point",
        "point_box": "prompt_point_box", "pointbox": "prompt_point_box",
        "prompt_point_box": "prompt_point_box",
        "autogen": "autogen", "auto": "autogen", "automatic": "autogen",
    }
    if pm in mapping:
        return mapping[pm]
    raise ValueError(
        f"Unsupported prompt mode '{prompt_mode}'. "
        "Expected: prompt_point, prompt_bbox, prompt_point_box, autogen."
    )


def mask_to_bbox(
    mask: np.ndarray, margin_ratio: float = 0.02,
) -> Optional[Tuple[int, int, int, int]]:
    m = np.asarray(mask).astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = m.shape[:2]
    mx = int((x1 - x0 + 1) * margin_ratio)
    my = int((y1 - y0 + 1) * margin_ratio)
    return (max(0, x0 - mx), max(0, y0 - my), min(w - 1, x1 + mx), min(h - 1, y1 + my))


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    m = np.asarray(mask).astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return (int(np.round(xs.mean())), int(np.round(ys.mean())))


def build_prompt(
    gt_mask: Optional[np.ndarray],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Build a prompt payload dict with ``point`` and ``bbox`` keys."""
    h, w = image_shape
    if gt_mask is not None and np.asarray(gt_mask).astype(bool).any():
        return {"point": mask_centroid(gt_mask), "bbox": mask_to_bbox(gt_mask)}
    return {"point": (w // 2, h // 2), "bbox": (w // 4, h // 4, 3 * w // 4, 3 * h // 4)}
