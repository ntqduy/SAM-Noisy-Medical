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
from models.wrappers.prompt_utils import (
    build_sam_prompt_kwargs,
    normalize_prompt_mode,
    resolve_prompt,
    select_best_mask,
)


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

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            ckpt = self.model_cfg.get("checkpoint", "weights/sam3.pt")
            if not os.path.isabs(ckpt):
                ckpt_candidate = os.path.join(project_root, ckpt)
                if os.path.exists(ckpt_candidate):
                    ckpt = ckpt_candidate
            if not os.path.exists(ckpt):
                ckpt_in_weights = os.path.join(project_root, "weights", os.path.basename(ckpt))
                if os.path.exists(ckpt_in_weights):
                    ckpt = ckpt_in_weights

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
        p = resolve_prompt(prompt, image.shape[:2], prompt_mode=self.prompt_mode)

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

            kwargs = build_sam_prompt_kwargs(self.prompt_mode, prompt, batched_box=True)

            masks, iou_predictions, _ = self._model.predict_inst(
                inference_state, **kwargs
            )
        return select_best_mask(masks, iou_predictions)
