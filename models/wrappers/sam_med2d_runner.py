"""
SAM-Med2D model wrapper.
"""

from __future__ import annotations

import os
import sys
import warnings
import importlib
import zipfile
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import (
    build_sam_prompt_kwargs,
    normalize_prompt_mode,
    resolve_prompt,
    select_best_mask,
)


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
            import torch

            sam_med2d_root = os.path.join(
                os.path.dirname(__file__), os.pardir, "external", "SAM-Med2D"
            )
            sam_med2d_root = os.path.abspath(sam_med2d_root)

            # ``segment_anything`` is shared across multiple bundled forks.
            # Ensure this runner imports SAM-Med2D's local package.
            seg_mod = sys.modules.get("segment_anything")
            if seg_mod is not None:
                mod_file = os.path.abspath(getattr(seg_mod, "__file__", "") or "")
                if not mod_file.startswith(sam_med2d_root):
                    for k in list(sys.modules.keys()):
                        if k == "segment_anything" or k.startswith("segment_anything."):
                            sys.modules.pop(k, None)

            # Force SAM-Med2D source root to the highest import priority.
            if sam_med2d_root in sys.path:
                sys.path.remove(sam_med2d_root)
            sys.path.insert(0, sam_med2d_root)
            importlib.invalidate_caches()

            from segment_anything import sam_model_registry, SamPredictor

            seg_mod = sys.modules.get("segment_anything")
            seg_file = os.path.abspath(getattr(seg_mod, "__file__", "") or "")
            if not seg_file.startswith(sam_med2d_root):
                raise RuntimeError(
                    "Imported wrong segment_anything package for SAM-Med2D: "
                    f"{seg_file}"
                )

            ckpt = self.model_cfg.get("checkpoint", "weights/sam_med2d.pt")
            model_type = self.model_cfg.get("model_type", "vit_b")

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            if not os.path.isabs(ckpt):
                ckpt_candidate = os.path.join(project_root, ckpt)
                if os.path.exists(ckpt_candidate):
                    ckpt = ckpt_candidate
            if not os.path.exists(ckpt):
                weights_dir = os.path.join(project_root, "weights")
                basename = os.path.basename(ckpt)
                alt_candidates = [
                    basename,
                    "sam-med2d_b.pth",
                    "sam_med2d_b.pth",
                    "sam_med2d.pt",
                ]
                for c in alt_candidates:
                    cand = os.path.join(weights_dir, c)
                    if os.path.exists(cand):
                        ckpt = cand
                        break

            if not os.path.exists(ckpt):
                raise RuntimeError(f"SAM-Med2D checkpoint not found: {ckpt}")

            # Quick integrity check: many modern checkpoints are zip containers.
            # If a file starts with PK signature but zip central directory is broken,
            # torch.load will fail later with a cryptic PytorchStreamReader error.
            with open(ckpt, "rb") as f:
                sig = f.read(4)
            if sig.startswith(b"PK"):
                try:
                    with zipfile.ZipFile(ckpt, "r") as zf:
                        _ = zf.namelist()[:1]
                except zipfile.BadZipFile as e:
                    raise RuntimeError(
                        f"SAM-Med2D checkpoint appears corrupted/incomplete: {ckpt}. "
                        "Please re-download the file."
                    ) from e

            load_device = self.device
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                current_arch = f"sm_{major}{minor}"
                supported_arches = set(torch.cuda.get_arch_list())
                if current_arch not in supported_arches:
                    warnings.warn(
                        "SAM-Med2D CUDA arch is unsupported by current torch build "
                        f"({current_arch} not in {sorted(supported_arches)}); "
                        "forcing CPU for SAM-Med2D.",
                        RuntimeWarning,
                    )
                    load_device = "cpu"

            image_size = int(self.model_cfg.get("image_size", 256))
            encoder_adapter = bool(self.model_cfg.get("encoder_adapter", True))
            builder_args = SimpleNamespace(
                image_size=image_size,
                sam_checkpoint=ckpt,
                encoder_adapter=encoder_adapter,
            )

            sam = sam_model_registry[model_type](builder_args)
            sam.to(load_device)
            self.device = load_device
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
