"""
MedSAM2 model wrapper – MedSAM fine-tuned on SAM2 backbone.

Uses the ``sam2`` package from ``models/external/MedSAM2``.
Default weight: ``weights/MedSAM2_latest.pt``
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Any, Dict, Optional

import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import build_prompt, normalize_prompt_mode


class MedSAM2Runner(ModelRunner):
    """
    Wrapper for MedSAM2 (SAM2-backbone fine-tuned for medical imaging).
    Falls back to heuristic inference when the model package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "MEDSAM2",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None

    def load_model(self) -> None:
        try:
            medsam2_root = os.path.join(
                os.path.dirname(__file__), os.pardir, "external", "MedSAM2"
            )
            medsam2_root = os.path.abspath(medsam2_root)
            if medsam2_root not in sys.path:
                sys.path.insert(0, medsam2_root)

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            ckpt = self.model_cfg.get("checkpoint", "weights/MedSAM2_latest.pt")
            cfg_path = self.model_cfg.get("config", "sam2.1_hiera_t512.yaml")
            sam2 = build_sam2(cfg_path, ckpt, device=self.device)
            self._model = SAM2ImagePredictor(sam2)
        except Exception as e:
            warnings.warn(
                f"MedSAM2 load failed ({e}); using heuristic fallback.",
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
