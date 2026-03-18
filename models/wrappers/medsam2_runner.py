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
from models.wrappers.prompt_utils import (
    build_sam_prompt_kwargs,
    normalize_prompt_mode,
    resolve_prompt,
    select_best_mask,
)


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

            # ``sam2`` module name is shared by multiple bundled forks.
            # Ensure this runner imports the MedSAM2 fork, not a previously cached one.
            sam2_mod = sys.modules.get("sam2")
            if sam2_mod is not None:
                mod_file = os.path.abspath(getattr(sam2_mod, "__file__", "") or "")
                if not mod_file.startswith(medsam2_root):
                    for k in list(sys.modules.keys()):
                        if k == "sam2" or k.startswith("sam2."):
                            sys.modules.pop(k, None)

            if medsam2_root not in sys.path:
                sys.path.insert(0, medsam2_root)

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            ckpt = self.model_cfg.get("checkpoint", "weights/MedSAM2_latest.pt")
            if not os.path.isabs(ckpt):
                ckpt_candidate = os.path.join(project_root, ckpt)
                if os.path.exists(ckpt_candidate):
                    ckpt = ckpt_candidate
            if not os.path.exists(ckpt):
                ckpt_in_weights = os.path.join(project_root, "weights", os.path.basename(ckpt))
                if os.path.exists(ckpt_in_weights):
                    ckpt = ckpt_in_weights

            cfg_name = str(self.model_cfg.get("config", "sam2.1_hiera_t512.yaml")).replace("\\", "/")
            # For this fork, Hydra resolves configs under pkg://sam2 with the "configs/" prefix.
            if not cfg_name.startswith("configs/"):
                cfg_name = f"configs/{cfg_name}"

            cfg_dir = os.path.join(medsam2_root, "sam2", "configs")
            if not os.path.exists(os.path.join(medsam2_root, "sam2", cfg_name)):
                base = os.path.basename(cfg_name)
                if os.path.exists(os.path.join(cfg_dir, base)):
                    cfg_name = f"configs/{base}"

            sam2 = build_sam2(cfg_name, ckpt, device=self.device)
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
