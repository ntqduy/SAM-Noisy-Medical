"""
ModelRunner – abstract base class for all segmentation model wrappers.

Every model wrapper must expose:
  - ``load_model()``
  - ``predict(image, prompt)``

Also provides shared heuristic-fallback utilities so that individual
runners do not need to duplicate them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cv2
import numpy as np


# ── shared CV utilities ─────────────────────────────────────────────────

def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected component of a binary mask."""
    u8 = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def component_at_point(mask: np.ndarray, point: tuple) -> np.ndarray:
    """Return the connected component under *point*, or the largest."""
    u8 = (mask > 0).astype(np.uint8)
    h, w = u8.shape[:2]
    x = int(np.clip(point[0], 0, w - 1))
    y = int(np.clip(point[1], 0, h - 1))
    n, labels, _, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return u8
    lbl = int(labels[y, x])
    if lbl == 0:
        return largest_cc(u8)
    return (labels == lbl).astype(np.uint8)


def apply_bbox(mask: np.ndarray, bbox) -> np.ndarray:
    """Zero out everything outside *bbox* (x0, y0, x1, y1)."""
    if bbox is None:
        return mask.astype(np.uint8)
    out = np.zeros_like(mask, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    out[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]
    return out


def autogen_mask(image: np.ndarray) -> np.ndarray:
    """Otsu-based automatic foreground mask (heuristic)."""
    blur = cv2.GaussianBlur(image.astype(np.uint8), (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)
    cand = th if float((th > 0).mean()) <= float((th_inv > 0).mean()) else th_inv
    k = np.ones((3, 3), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k, iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=1)
    return largest_cc((cand > 0).astype(np.uint8))


# ── base class ───────────────────────────────────────────────────────────

class ModelRunner(ABC):
    """Base interface for segmentation model runners."""

    def __init__(
        self,
        model_name: str,
        prompt_mode: str,
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.device = device
        self.model_cfg = model_cfg or {}

    @abstractmethod
    def load_model(self) -> None:
        """Load / initialise the underlying model weights."""
        ...

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Run segmentation on *image* using the given *prompt*.

        Returns a binary mask ``np.ndarray`` of shape HxW with values {0, 1}.
        """
        ...

    # ── shared heuristic fallback ────────────────────────────────────────

    def _heuristic(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        """Prompt-aware heuristic fallback when the real model is unavailable."""
        base = autogen_mask(image)
        if self.prompt_mode == "autogen":
            return base
        if self.prompt_mode == "prompt_bbox":
            return apply_bbox(base, prompt.get("bbox"))
        if self.prompt_mode in ("prompt_point", "prompt_multi_point"):
            pt = prompt.get("point") or (image.shape[1] // 2, image.shape[0] // 2)
            return component_at_point(base, pt)
        if self.prompt_mode == "prompt_point_box":
            pt = prompt.get("point") or (image.shape[1] // 2, image.shape[0] // 2)
            roi = apply_bbox(base, prompt.get("bbox"))
            return component_at_point(roi, pt)
        raise ValueError(f"Unsupported prompt mode: {self.prompt_mode}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, prompt={self.prompt_mode!r})"
