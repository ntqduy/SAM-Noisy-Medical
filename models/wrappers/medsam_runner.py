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
        if False:
            raise ValueError(
                "MedSAM (original) supports native box prompt only in this benchmark setup. "
                f"Received prompt_mode={self.prompt_mode}."
            )

        p = resolve_prompt(prompt, image.shape[:2], prompt_mode=self.prompt_mode)

        if self._predictor is not None:
            return self._run_inference(image, p)
        raise RuntimeError(
            "MedSAM predictor is not loaded. Real-weight inference is required; "
            "heuristic fallback is disabled."
        )

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        import cv2

        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image

        # MedSAM was trained with 1024 resizing; resize and scale prompts accordingly.
        src_h, src_w = img_rgb.shape[:2]
        target = 1024
        scale = float(target) / float(max(src_h, src_w))
        new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Scale bbox / points to resized space.
        bbox = prompt.get("bbox")
        if bbox is not None:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            bbox = (
                int(round(x0 * scale)),
                int(round(y0 * scale)),
                int(round(x1 * scale)),
                int(round(y1 * scale)),
            )

        pts = prompt.get("points")
        if pts is None and prompt.get("point") is not None:
            pt = prompt["point"]
            pts = np.asarray([[pt[0], pt[1]]], dtype=np.float32)
        else:
            pts = None if pts is None else np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if pts is not None:
            pts = (pts * scale).astype(np.float32)

        lbl = prompt.get("point_labels")
        if lbl is None and pts is not None:
            lbl = np.ones((pts.shape[0],), dtype=np.int32)
        elif lbl is not None:
            lbl = np.asarray(lbl, dtype=np.int32).reshape(-1)
            if pts is not None and lbl.shape[0] < pts.shape[0]:
                pad = np.ones((pts.shape[0] - lbl.shape[0],), dtype=np.int32)
                lbl = np.concatenate([lbl, pad], axis=0)
            if pts is not None:
                lbl = lbl[: pts.shape[0]]

        # If prompt_mode is bbox-only, allow box; otherwise use points when present.
        # Set image embedding on resized input.
        self._predictor.set_image(resized)

        kwargs = {
            "multimask_output": True,
        }
        if bbox is not None and self.prompt_mode in ("prompt_bbox", "prompt_point_box"):
            box_arr = np.asarray(bbox, dtype=np.float32)
            kwargs["box"] = box_arr
        if pts is not None and self.prompt_mode in ("prompt_point", "prompt_point_box", "prompt_multi_point"):
            kwargs["point_coords"] = pts
            kwargs["point_labels"] = lbl if lbl is not None else np.ones((pts.shape[0],), dtype=np.int32)

        if self.prompt_mode == "prompt_point" and pts is not None:
            # Point-only MedSAM can return tiny binary masks at threshold 0.
            # Use logits with a lower threshold and keep the component around the click.
            masks, scores, logits = self._predictor.predict(return_logits=True, **kwargs)
            logit_thresh = float(self.model_cfg.get("point_logit_threshold", -1.2))
            point_arr = prompt.get("points")
            if point_arr is None and prompt.get("point") is not None:
                point_arr = np.asarray([prompt["point"]], dtype=np.float32)
            elif point_arr is not None:
                point_arr = np.asarray(point_arr, dtype=np.float32).reshape(-1, 2)

            logits_arr = np.asarray(logits)
            if logits_arr.ndim == 2:
                logits_arr = logits_arr[None, ...]

            best = None
            best_area = -1
            for lo in logits_arr:
                bin_mask = (lo > logit_thresh).astype(np.uint8)

                if point_arr is not None and point_arr.shape[0] > 0:
                    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
                    if num_labels > 1:
                        keep_labels = set()
                        for point_xy in point_arr:
                            px = int(round(float(point_xy[0]) * scale))
                            py = int(round(float(point_xy[1]) * scale))
                            px = int(np.clip(px, 0, bin_mask.shape[1] - 1))
                            py = int(np.clip(py, 0, bin_mask.shape[0] - 1))
                            click_label = int(labels[py, px])
                            if click_label > 0:
                                keep_labels.add(click_label)
                        if keep_labels:
                            keep_mask = np.isin(labels, list(keep_labels))
                            bin_mask = keep_mask.astype(np.uint8)

                area = int(bin_mask.sum())
                if area > best_area:
                    best_area = area
                    best = bin_mask

            if best is None:
                best = select_best_mask(masks, scores)
        else:
            masks, scores, _ = self._predictor.predict(**kwargs)
            best = select_best_mask(masks, scores)

        # Resize mask back to source size.
        if best.shape[:2] != (src_h, src_w):
            best = cv2.resize(best.astype(np.uint8), (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        return (best > 0).astype(np.uint8)
