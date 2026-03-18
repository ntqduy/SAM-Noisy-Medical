"""
SAM-Med2D model wrapper.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import build_prompt, normalize_prompt_mode


class SAMMed2DRunner(ModelRunner):
    """
    Wrapper for SAM-Med2D.
    Falls back to heuristic inference when the model package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "SAM-MED2D",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None

    def load_model(self) -> None:
        try:
            from sam_med2d import sam_model_registry, SamPredictor

            ckpt = self.model_cfg.get("checkpoint", "weights/sam_med2d.pt")
            model_type = self.model_cfg.get("model_type", "vit_b")
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(self.device)
            self._model = SamPredictor(sam)
        except Exception as e:
            warnings.warn(
                f"SAM-Med2D load failed ({e}); using heuristic fallback.",
                RuntimeWarning,
            )
            self._model = None

    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if prompt is None:
            prompt = {}
        gt_mask = prompt.get("gt_mask")
        p = build_prompt(gt_mask, image.shape[:2])

        if self._model is not None:
            return self._run_inference(image, p)
        return self._heuristic(image, p)

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
        self._model.set_image(img_rgb)

        kwargs: Dict[str, Any] = {"multimask_output": False}
        if self.prompt_mode in ("prompt_bbox", "prompt_point_box") and prompt.get("bbox"):
            kwargs["box"] = np.array(prompt["bbox"])
        if self.prompt_mode in ("prompt_point", "prompt_point_box") and prompt.get("point"):
            pt = prompt["point"]
            kwargs["point_coords"] = np.array([[pt[0], pt[1]]])
            kwargs["point_labels"] = np.array([1])

        masks, _, _ = self._model.predict(**kwargs)
        return (masks[0] > 0).astype(np.uint8)
