"""
SAM 3 model wrapper.
"""

from __future__ import annotations

import sys
import os
import warnings
from typing import Any, Dict, Optional

import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import build_prompt, normalize_prompt_mode


class SAM3Runner(ModelRunner):
    """
    Wrapper for SAM 3.
    Falls back to heuristic inference when the ``sam3`` package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "SAM3",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        try:
            sam3_root = os.path.join(
                os.path.dirname(__file__), os.pardir, "external", "sam3"
            )
            sam3_root = os.path.abspath(sam3_root)
            if sam3_root not in sys.path:
                sys.path.insert(0, sam3_root)

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            ckpt = self.model_cfg.get("checkpoint", "weights/sam3.pt")

            self._model = build_sam3_image_model(
                device=self.device,
                compile=False,
                checkpoint_path=ckpt,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
            self._processor = Sam3Processor(
                self._model, device=self.device
            )
        except Exception as e:
            warnings.warn(f"SAM3 load failed ({e}); using heuristic fallback.", RuntimeWarning)
            self._model = None
            self._processor = None

    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if prompt is None:
            prompt = {}
        gt_mask = prompt.get("gt_mask")
        p = build_prompt(gt_mask, image.shape[:2])

        if self._model is not None and self._processor is not None:
            return self._run_inference(image, p)
        return self._heuristic(image, p)

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        import torch
        from PIL import Image as PILImage

        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
        pil_img = PILImage.fromarray(img_rgb.astype(np.uint8))

        with torch.inference_mode():
            inference_state = self._processor.set_image(pil_img)

            kwargs: Dict[str, Any] = {"multimask_output": False}
            if self.prompt_mode in ("prompt_bbox", "prompt_point_box") and prompt.get("bbox"):
                kwargs["box"] = np.array(prompt["bbox"])[None, :]
            if self.prompt_mode in ("prompt_point", "prompt_point_box") and prompt.get("point"):
                pt = prompt["point"]
                kwargs["point_coords"] = np.array([[pt[0], pt[1]]])
                kwargs["point_labels"] = np.array([1])

            masks, _, _ = self._model.predict_inst(
                inference_state, **kwargs
            )

        if masks.ndim == 4:
            mask = masks[0, 0]
        elif masks.ndim == 3:
            mask = masks[0]
        else:
            mask = masks

        return (mask > 0).astype(np.uint8)
