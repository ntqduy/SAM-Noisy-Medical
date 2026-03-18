"""MedSAM 1 wrapper (original MedSAM - SAM1 backbone fine-tuned).

For MedSAM2 see medsam2_runner.py, for MedSAM3 see medsam3_runner.py.
"""

from __future__ import annotations

import os
import sys
import warnings
import importlib
from typing import Any, Dict, Optional

import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import (
    build_sam_prompt_kwargs,
    normalize_prompt_mode,
    resolve_prompt,
    select_best_mask,
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
            sam_root = os.path.join(
                os.path.dirname(__file__), os.pardir, "external", "sam"
            )
            sam_root = os.path.abspath(sam_root)

            # ``segment_anything`` exists in multiple bundled forks.
            # Ensure MedSAMRunner uses the original SAM package.
            seg_mod = sys.modules.get("segment_anything")
            if seg_mod is not None:
                mod_file = os.path.abspath(getattr(seg_mod, "__file__", "") or "")
                if not mod_file.startswith(sam_root):
                    for k in list(sys.modules.keys()):
                        if k == "segment_anything" or k.startswith("segment_anything."):
                            sys.modules.pop(k, None)

            # Force SAM source root to the highest import priority.
            if sam_root in sys.path:
                sys.path.remove(sam_root)
            sys.path.insert(0, sam_root)
            importlib.invalidate_caches()

            from segment_anything import sam_model_registry, SamPredictor

            seg_mod = sys.modules.get("segment_anything")
            seg_file = os.path.abspath(getattr(seg_mod, "__file__", "") or "")
            if not seg_file.startswith(sam_root):
                raise RuntimeError(
                    "Imported wrong segment_anything package for MedSAM: "
                    f"{seg_file}"
                )

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            if not os.path.isabs(ckpt):
                ckpt_candidate = os.path.join(project_root, ckpt)
                if os.path.exists(ckpt_candidate):
                    ckpt = ckpt_candidate
            if not os.path.exists(ckpt):
                ckpt_in_weights = os.path.join(project_root, "weights", os.path.basename(ckpt))
                if os.path.exists(ckpt_in_weights):
                    ckpt = ckpt_in_weights

            load_device = self.device
            if str(self.device).startswith("cuda"):
                try:
                    import torch

                    if torch.cuda.is_available():
                        major, minor = torch.cuda.get_device_capability(0)
                        current_arch = f"sm_{major}{minor}"
                        supported_arches = set(torch.cuda.get_arch_list())
                        if current_arch not in supported_arches:
                            warnings.warn(
                                "MedSAM CUDA arch is unsupported by current torch build "
                                f"({current_arch} not in {sorted(supported_arches)}); "
                                "forcing CPU for MedSAM.",
                                RuntimeWarning,
                            )
                            load_device = "cpu"
                except Exception:
                    pass

            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(load_device)
            self.device = load_device
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
        p = resolve_prompt(prompt, image.shape[:2], prompt_mode=self.prompt_mode)

        if self._predictor is not None:
            return self._run_inference(image, p)
        return self._heuristic(image, p)

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
        self._predictor.set_image(img_rgb)

        kwargs = build_sam_prompt_kwargs(self.prompt_mode, prompt, batched_box=False)

        masks, iou_predictions, _ = self._predictor.predict(**kwargs)
        return select_best_mask(masks, iou_predictions)
