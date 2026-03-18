"""MedSAM 1 wrapper (original MedSAM - SAM1 backbone fine-tuned).

For MedSAM2 see medsam2_runner.py, for MedSAM3 see medsam3_runner.py.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from model.base_model import ModelRunner
from model.prompt_utils import build_prompt, normalize_prompt_mode
from model.model_wrappers.sam_runner import (
    _autogen_mask,
    _apply_bbox,
    _component_at_point,
)

_MEDSAM_DEFAULTS: Dict[str, Dict[str, str]] = {
    "MEDSAM": {"checkpoint": "weights/medsam_vit_b.pth", "model_type": "vit_b"},
    "MEDSAM1": {"checkpoint": "weights/medsam_vit_b.pth", "model_type": "vit_b"},
}


class MedSAMRunner(ModelRunner):
    """
    Wrapper for MedSAM 1 (original MedSAM on SAM1/ViT backbone).
    Falls back to heuristic inference when the model package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "MEDSAM",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._predictor = None

    def load_model(self) -> None:
        key = self.model_name.upper()
        defaults = _MEDSAM_DEFAULTS.get(key, _MEDSAM_DEFAULTS["MEDSAM"])
        ckpt = self.model_cfg.get("checkpoint", defaults["checkpoint"])
        model_type = self.model_cfg.get("model_type", defaults.get("model_type", "vit_b"))

        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(self.device)
            self._predictor = SamPredictor(sam)
        except Exception as e:
            warnings.warn(
                f"MedSAM ({self.model_name}) load failed ({e}); using heuristic.",
                RuntimeWarning,
            )
            self._predictor = None

    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if prompt is None:
            prompt = {}
        gt_mask = prompt.get("gt_mask")
        p = build_prompt(gt_mask, image.shape[:2])

        if self._predictor is not None:
            return self._run_inference(image, p)
        return self._heuristic(image, p)

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
        self._predictor.set_image(img_rgb)

        kwargs: Dict[str, Any] = {"multimask_output": False}
        if self.prompt_mode in ("prompt_bbox", "prompt_point_box") and prompt.get("bbox"):
            kwargs["box"] = np.array(prompt["bbox"])
        if self.prompt_mode in ("prompt_point", "prompt_point_box") and prompt.get("point"):
            pt = prompt["point"]
            kwargs["point_coords"] = np.array([[pt[0], pt[1]]])
            kwargs["point_labels"] = np.array([1])

        masks, _, _ = self._predictor.predict(**kwargs)
        return (masks[0] > 0).astype(np.uint8)

    def _heuristic(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        base = _autogen_mask(image)
        if self.prompt_mode == "autogen":
            return base
        if self.prompt_mode == "prompt_bbox":
            return _apply_bbox(base, prompt.get("bbox"))
        if self.prompt_mode == "prompt_point":
            pt = prompt.get("point") or (image.shape[1] // 2, image.shape[0] // 2)
            return _component_at_point(base, pt)
        if self.prompt_mode == "prompt_point_box":
            pt = prompt.get("point") or (image.shape[1] // 2, image.shape[0] // 2)
            roi = _apply_bbox(base, prompt.get("bbox"))
            return _component_at_point(roi, pt)
        raise ValueError(f"Unsupported prompt mode: {self.prompt_mode}")
