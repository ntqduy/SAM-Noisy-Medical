"""
SAM 2 model wrapper.
"""

from __future__ import annotations

import os
import sys
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


class SAM2Runner(ModelRunner):
    """
    Wrapper for SAM 2.
    Falls back to heuristic inference when the ``sam2`` package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "SAM2",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None

    def load_model(self) -> None:
        try:
            # Prefer the bundled SAM2 source tree so users don't need a separate pip install.
            sam2_root = os.path.join(
                os.path.dirname(__file__), os.pardir, "external", "sam2"
            )
            sam2_root = os.path.abspath(sam2_root)

            # ``sam2`` module name is shared by multiple bundled forks.
            # Ensure this runner imports the standalone SAM2 fork.
            sam2_mod = sys.modules.get("sam2")
            if sam2_mod is not None:
                mod_file = os.path.abspath(getattr(sam2_mod, "__file__", "") or "")
                if not mod_file.startswith(sam2_root):
                    for k in list(sys.modules.keys()):
                        if k == "sam2" or k.startswith("sam2."):
                            sys.modules.pop(k, None)

            if sam2_root not in sys.path:
                sys.path.insert(0, sam2_root)

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            ckpt = self.model_cfg.get("checkpoint", "weights/sam2_b.pt")
            if not os.path.isabs(ckpt):
                ckpt_candidate = os.path.join(project_root, ckpt)
                if os.path.exists(ckpt_candidate):
                    ckpt = ckpt_candidate
            if not os.path.exists(ckpt):
                ckpt_in_weights = os.path.join(project_root, "weights", os.path.basename(ckpt))
                if os.path.exists(ckpt_in_weights):
                    ckpt = ckpt_in_weights

            cfg_path = self.model_cfg.get("config", "configs/sam2/sam2_hiera_b+.yaml")
            cfg_candidates = [cfg_path]
            if not os.path.isabs(cfg_path):
                cfg_candidates.append(os.path.join(project_root, cfg_path))
                cfg_candidates.append(os.path.join(sam2_root, cfg_path))
            cfg_base = os.path.basename(cfg_path)
            cfg_candidates.append(os.path.join(sam2_root, "sam2", "configs", "sam2", cfg_base))
            cfg_candidates.append(os.path.join(sam2_root, "sam2", "configs", "sam2.1", cfg_base))
            cfg_candidates.append(os.path.join(sam2_root, "sam2", cfg_base))
            cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), cfg_path)

            # SAM2's hydra compose expects a package-relative config name (e.g. configs/sam2/...).
            cfg_name = str(cfg_path).replace("\\", "/")
            if os.path.isabs(cfg_path):
                sam2_pkg_root = os.path.join(sam2_root, "sam2")
                try:
                    rel_cfg = os.path.relpath(cfg_path, sam2_pkg_root).replace("\\", "/")
                    if not rel_cfg.startswith("../") and rel_cfg != "..":
                        cfg_name = rel_cfg
                except Exception:
                    pass

            if cfg_name.endswith(".yaml") and "/" not in cfg_name:
                if "sam2.1" in cfg_name:
                    cfg_name = f"configs/sam2.1/{cfg_name}"
                else:
                    cfg_name = f"configs/sam2/{cfg_name}"

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
                                "SAM2 CUDA arch is unsupported by current torch build "
                                f"({current_arch} not in {sorted(supported_arches)}); "
                                "forcing CPU for SAM2.",
                                RuntimeWarning,
                            )
                            load_device = "cpu"
                except Exception:
                    # If detection fails, keep the requested device and let build_sam2 handle it.
                    pass

            try:
                sam2 = build_sam2(cfg_name, ckpt, device=load_device)
                self.device = load_device
            except Exception as e_cuda:
                # On some machines (e.g., newer GPUs with older torch builds),
                # CUDA init can fail even when a checkpoint is valid.
                if str(load_device).startswith("cuda"):
                    warnings.warn(
                        f"SAM2 CUDA load failed ({e_cuda}); retrying on CPU.",
                        RuntimeWarning,
                    )
                    sam2 = build_sam2(cfg_name, ckpt, device="cpu")
                    self.device = "cpu"
                else:
                    raise
            self._model = SAM2ImagePredictor(sam2)
        except Exception as e:
            warnings.warn(f"SAM2 load failed ({e}); using heuristic fallback.", RuntimeWarning)
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
