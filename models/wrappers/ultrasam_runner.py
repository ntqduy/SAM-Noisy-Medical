"""
UltraSAM model wrapper.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import (
    build_sam_prompt_kwargs,
    normalize_prompt_mode,
    resolve_prompt,
    select_best_mask,
)


class UltraSAMRunner(ModelRunner):
    """
    Wrapper for UltraSAM.
    Falls back to heuristic inference when the model package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "ULTRASAM",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None

    def load_model(self) -> None:
        try:
            from ultrasam import UltraSamPredictor

            ckpt = self.model_cfg.get("checkpoint", "weights/UltraSam.pth")
            self._model = UltraSamPredictor(checkpoint=ckpt, device=self.device)
        except Exception as e:
            warnings.warn(
                f"UltraSAM load failed ({e}); using heuristic fallback.",
                RuntimeWarning,
            )
            self._model = None

    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        p = resolve_prompt(prompt, image.shape[:2], prompt_mode=self.prompt_mode)

        if self._model is not None:
            return self._run_inference(image, p)
        return self._heuristic(image, p)

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
        self._model.set_image(img_rgb)

        kwargs = build_sam_prompt_kwargs(self.prompt_mode, prompt, batched_box=False)

        masks, iou_predictions, _ = self._model.predict(**kwargs)
        return select_best_mask(masks, iou_predictions)
