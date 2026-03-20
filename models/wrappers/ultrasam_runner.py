"""
UltraSAM model wrapper.
"""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from typing import Any, Dict, Optional

import cv2
import numpy as np

from models.wrappers.base_model import ModelRunner
from models.wrappers.prompt_utils import (
    build_sam_prompt_kwargs,
    normalize_prompt_mode,
    resolve_prompt,
    select_best_mask,
)


class UltraSAMRunner(ModelRunner):
    """
    Wrapper for UltraSAM.
    Falls back to heuristic inference when the model package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "ULTRASAM",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None
        self._backend = None
        self._mode_warning_emitted = False
        self._mha_monkey_patched = False

    def _ensure_ultrasam_mha_patch(self) -> None:
        """Apply UltraSAM's custom MHA functional patch for inference paths.

        UltraSAM upstream normally applies this via a runtime hook in MMEngine
        training/testing loops. Our wrapper calls `model.test_step` directly,
        so we patch here as well to keep behavior consistent.
        """
        if self._mha_monkey_patched:
            return
        try:
            from endosam.models.utils.custom_functional import (
                multi_head_attention_forward as ultrasam_mha_forward,
            )
            import torch.nn.functional as F

            F.multi_head_attention_forward = ultrasam_mha_forward
            self._mha_monkey_patched = True
        except Exception:
            # If patching fails, keep default behavior and let downstream raise
            # a clear runtime error from inference.
            self._mha_monkey_patched = False

    def _mode_suffix(self) -> Optional[str]:
        mapping = {
            "prompt_point": "point",
            "prompt_bbox": "bbox",
            "prompt_point_box": "point_box",
        }
        return mapping.get(str(self.prompt_mode))

    def _mode_cfg_value(self, base_key: str, default_value):
        """
        Resolve model_cfg value with optional prompt-mode-specific override.

        Priority:
        1) `<base_key>_<mode_suffix>` e.g. config_point / config_bbox / config_point_box
        2) `<base_key>`
        3) `default_value`
        """
        suffix = self._mode_suffix()
        if suffix:
            mode_key = f"{base_key}_{suffix}"
            if mode_key in self.model_cfg:
                return self.model_cfg.get(mode_key)
        return self.model_cfg.get(base_key, default_value)

    def load_model(self) -> None:
        attempted_backend = "unknown"
        try:
            UltraSamPredictor = None
            try:
                from ultrasam import UltraSamPredictor
            except Exception:
                UltraSamPredictor = None

            if UltraSamPredictor is None:
                attempted_backend = "mmdet"
                # Try bundled UltraSam source tree first.
                ultrasam_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), os.pardir, "external", "UltraSam")
                )
                if ultrasam_root not in sys.path:
                    sys.path.insert(0, ultrasam_root)
                from mmdet.apis import init_detector

                # Use the box-refine config by default; it matches bundled UltraSAM checkpoints.
                # If caller provides a custom config (e.g., point/no-refine weights), it overrides.
                cfg = self._mode_cfg_value(
                    "config",
                    os.path.join(
                        ultrasam_root,
                        "configs",
                        "UltraSAM",
                        "UltraSAM_full",
                        "UltraSAM_box_refine.py",
                    ),
                )
                ckpt = self._mode_cfg_value("checkpoint", "weights/UltraSam.pth")

                # Map relative paths from project root.
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
                )
                if not os.path.isabs(cfg):
                    cfg = os.path.join(project_root, cfg)
                if not os.path.isabs(ckpt):
                    ckpt = os.path.join(project_root, ckpt)

                # Keep UltraSAM attention behavior aligned with upstream hooks.
                self._ensure_ultrasam_mha_patch()

                infer_device = self.device
                if str(self.device).startswith("cuda"):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            major, minor = torch.cuda.get_device_capability(0)
                            current_arch = f"sm_{major}{minor}"
                            supported_arches = set(torch.cuda.get_arch_list())
                            if current_arch not in supported_arches:
                                warnings.warn(
                                    "UltraSAM CUDA arch is unsupported by current torch build "
                                    f"({current_arch} not in {sorted(supported_arches)}); "
                                    "forcing CPU for UltraSAM.",
                                    RuntimeWarning,
                                )
                                infer_device = "cpu"
                    except Exception:
                        infer_device = "cpu"

                self._model = init_detector(config=cfg, checkpoint=ckpt, device=infer_device)
                self._model.eval()
                self._backend = "mmdet"
                self.device = infer_device

                if self.prompt_mode == "prompt_point":
                    warnings.warn(
                        "UltraSAM MMDet backend is running in native POINT prompt mode.",
                        RuntimeWarning,
                    )
                elif self.prompt_mode == "prompt_bbox":
                    warnings.warn(
                        "UltraSAM MMDet backend is running in native BOX prompt mode.",
                        RuntimeWarning,
                    )
                elif self.prompt_mode == "prompt_point_box":
                    warnings.warn(
                        "UltraSAM MMDet backend will run point+box as two real passes "
                        "(point and box) and fuse masks.",
                        RuntimeWarning,
                    )
                return

            else:
                attempted_backend = "predictor"
                ckpt = self.model_cfg.get("checkpoint", "weights/UltraSam.pth")
                self._model = UltraSamPredictor(checkpoint=ckpt, device=self.device)
                self._backend = "predictor"
        except Exception as e:
            install_hint = (
                "Install UltraSAM deps in your active env, e.g.: "
                "pip install -U openmim && "
                "mim install mmengine && "
                "mim install 'mmcv==2.1.0' && "
                "mim install mmdet mmpretrain"
            )
            err_msg = f"{type(e).__name__}: {e!r}"
            tb_tail = traceback.format_exc(limit=6).strip()
            warnings.warn(
                "UltraSAM load failed while initializing backend="
                f"{attempted_backend}. If backend=mmdet, ensure OpenMMLab deps are installed "
                "(mmengine/mmcv/mmdet/mmpretrain) and the UltraSAM config is compatible. "
                "If backend=predictor, ensure a Python package `ultrasam` exposing "
                "`UltraSamPredictor` is available. "
                f"{install_hint}. Root error: {err_msg}. Trace tail: {tb_tail}",
                RuntimeWarning,
            )
            self._model = None
            self._backend = None

    def predict(
        self,
        image: np.ndarray,
        prompt: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        p = resolve_prompt(prompt, image.shape[:2], prompt_mode=self.prompt_mode)

        if self._model is not None:
            return self._run_inference(image, p)
        if bool(self.model_cfg.get("allow_fallback", False)):
            return self._heuristic(image, p)
        raise RuntimeError(
            "UltraSAM model is not loaded. Real-weight inference is required; "
            "heuristic fallback is disabled."
        )

    def _run_inference(self, image: np.ndarray, prompt: dict) -> np.ndarray:
        img_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image

        if self._backend == "predictor":
            self._model.set_image(img_rgb)
            kwargs = build_sam_prompt_kwargs(self.prompt_mode, prompt, batched_box=False)
            masks, iou_predictions, _ = self._model.predict(**kwargs)
            return select_best_mask(masks, iou_predictions)

        if self._backend == "mmdet":
            return self._run_mmdet_prompt_native(img_rgb, prompt)

        raise RuntimeError("UltraSAM backend is not initialized.")

    def _run_mmdet_prompt_native(self, img_rgb: np.ndarray, prompt: dict) -> np.ndarray:
        if self.prompt_mode == "prompt_point":
            # Keep behavior aligned with other SAM wrappers: try native point first.
            # If user's checkpoint/config cannot run point branch, optionally fall
            # back to box when force_box_for_point=true.
            try:
                return self._predict_mmdet_single_prompt(img_rgb, prompt, prompt_variant="point")
            except Exception as e:
                if bool(self.model_cfg.get("force_box_for_point", False)):
                    warnings.warn(
                        "UltraSAM point branch failed; falling back to box branch because "
                        "force_box_for_point=true. "
                        f"Root error: {type(e).__name__}: {e}",
                        RuntimeWarning,
                    )
                    return self._predict_mmdet_single_prompt(img_rgb, prompt, prompt_variant="box")
                raise
        if self.prompt_mode == "prompt_bbox":
            return self._predict_mmdet_single_prompt(img_rgb, prompt, prompt_variant="box")
        if self.prompt_mode == "prompt_point_box":
            box_mask, box_score = self._predict_mmdet_single_prompt(
                img_rgb, prompt, prompt_variant="box", return_score=True
            )
            try:
                point_mask, point_score = self._predict_mmdet_single_prompt(
                    img_rgb, prompt, prompt_variant="point", return_score=True
                )
            except Exception as e:
                if bool(self.model_cfg.get("force_box_for_point", False)):
                    warnings.warn(
                        "UltraSAM point+box point branch failed; falling back to box branch "
                        "because force_box_for_point=true. "
                        f"Root error: {type(e).__name__}: {e}",
                        RuntimeWarning,
                    )
                    return box_mask
                raise

            fusion_mode = str(self.model_cfg.get("point_box_fusion", "best_score")).strip().lower()
            if fusion_mode in {"best", "best_score", "score"}:
                return point_mask if float(point_score) >= float(box_score) else box_mask
            if fusion_mode in {"intersection", "inter"}:
                inter = ((point_mask > 0) & (box_mask > 0)).astype(np.uint8)
                if int(inter.sum()) > 0:
                    return inter
                return box_mask
            if fusion_mode == "union":
                return ((point_mask > 0) | (box_mask > 0)).astype(np.uint8)
            if fusion_mode in {"point", "point_only"}:
                return point_mask
            if fusion_mode in {"box", "box_only"}:
                return box_mask

            warnings.warn(
                f"Unknown UltraSAM point_box_fusion='{fusion_mode}', using best_score.",
                RuntimeWarning,
            )
            return point_mask if float(point_score) >= float(box_score) else box_mask

        raise ValueError(
            "Unsupported UltraSAM prompt mode for MMDet backend: "
            f"{self.prompt_mode}. Supported: prompt_point, prompt_bbox, prompt_point_box."
        )

    def _predict_mmdet_single_prompt(
        self,
        img_rgb: np.ndarray,
        prompt: dict,
        *,
        prompt_variant: str,
        return_score: bool = False,
    ):
        import torch
        from mmengine.structures import InstanceData
        from mmdet.structures import DetDataSample

        if prompt_variant not in {"point", "box"}:
            raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")

        in_size = int(self.model_cfg.get("input_size", 1024))
        src_h, src_w = img_rgb.shape[:2]

        # Match UltraSAM training/inference scale used in its configs.
        resized = cv2.resize(img_rgb, (in_size, in_size), interpolation=cv2.INTER_LINEAR)
        sx = float(in_size) / float(max(src_w, 1))
        sy = float(in_size) / float(max(src_h, 1))

        bbox = prompt.get("bbox")
        if bbox is None:
            bbox = (src_w // 4, src_h // 4, 3 * src_w // 4, 3 * src_h // 4)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        x0 = int(np.clip(x0, 0, src_w - 1))
        x1 = int(np.clip(x1, 0, src_w - 1))
        y0 = int(np.clip(y0, 0, src_h - 1))
        y1 = int(np.clip(y1, 0, src_h - 1))

        # Build point instances.
        point_instances = []
        if prompt_variant == "point":
            pts = prompt.get("points")
            if pts is not None:
                arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                for pxy in arr:
                    px = int(np.clip(int(pxy[0]), 0, src_w - 1))
                    py = int(np.clip(int(pxy[1]), 0, src_h - 1))
                    point_instances.append((px, py))

        if not point_instances:
            pt = prompt.get("point")
            if pt is None:
                pt = ((x0 + x1) // 2, (y0 + y1) // 2)
            px, py = int(pt[0]), int(pt[1])
            px = int(np.clip(px, 0, src_w - 1))
            py = int(np.clip(py, 0, src_h - 1))
            point_instances = [(px, py)]

        n_instances = len(point_instances)
        point_coords_np = np.asarray(
            [[[px * sx + 0.5, py * sy + 0.5]] for (px, py) in point_instances],
            dtype=np.float32,
        )
        box_pair = np.asarray(
            [[x0 * sx + 0.5, y0 * sy + 0.5], [x1 * sx + 0.5, y1 * sy + 0.5]],
            dtype=np.float32,
        )
        box_coords_np = np.repeat(box_pair[None, :, :], n_instances, axis=0)
        flat_bbox_np = np.repeat(
            np.asarray([[x0 * sx, y0 * sy, x1 * sx, y1 * sy]], dtype=np.float32),
            n_instances,
            axis=0,
        )

        # Prompt coordinates follow UltraSAM test transform convention (+0.5 offset).
        point_coords = torch.from_numpy(point_coords_np)
        box_coords = torch.from_numpy(box_coords_np)
        flat_bbox = torch.from_numpy(flat_bbox_np)

        prompt_type_value = 0 if prompt_variant == "point" else 1
        gt_instances = InstanceData(
            points=point_coords,
            boxes=box_coords,
            bboxes=flat_bbox,
            labels=torch.zeros((n_instances,), dtype=torch.long),
            prompt_types=torch.full((n_instances,), prompt_type_value, dtype=torch.long),
        )

        data_sample = DetDataSample()
        data_sample.gt_instances = gt_instances
        data_sample.set_metainfo(
            {
                "img_shape": (in_size, in_size),
                "ori_shape": (src_h, src_w),
                "pad_shape": (in_size, in_size),
                "batch_input_shape": (in_size, in_size),
                "scale_factor": (sx, sy),
            }
        )

        inputs = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
        batch = {"inputs": [inputs], "data_samples": [data_sample]}

        try:
            processed = self._model.data_preprocessor(batch, training=False)
            with torch.no_grad():
                pred_samples = self._model.test_step(processed)
        except Exception as e:
            raise RuntimeError(
                "UltraSAM MMDet prompt-native inference failed. "
                f"mode={self.prompt_mode}, prompt_variant={prompt_variant}. Root error: {e}"
            ) from e

        if not pred_samples:
            empty = np.zeros((src_h, src_w), dtype=np.uint8)
            return (empty, 0.0) if return_score else empty

        pred_instances = pred_samples[0].pred_instances
        if pred_instances is None or len(pred_instances) == 0 or not hasattr(pred_instances, "masks"):
            empty = np.zeros((src_h, src_w), dtype=np.uint8)
            return (empty, 0.0) if return_score else empty

        masks = pred_instances.masks
        if isinstance(masks, torch.Tensor):
            masks_np = masks.detach().cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        pred_score = 0.0
        if masks_np.ndim == 2:
            mask = (masks_np > 0).astype(np.uint8)
            if hasattr(pred_instances, "scores") and len(pred_instances.scores) > 0:
                pred_score = float(pred_instances.scores[0].detach().cpu().item())
        else:
            if prompt_variant == "point" and n_instances > 1:
                # Multi-point policy for disconnected objects: keep union across
                # point-instance predictions.
                mask = (masks_np > 0).any(axis=0).astype(np.uint8)
                if hasattr(pred_instances, "scores") and len(pred_instances.scores) == masks_np.shape[0]:
                    pred_score = float(torch.max(pred_instances.scores).detach().cpu().item())
            else:
                idx = 0
                if hasattr(pred_instances, "scores") and len(pred_instances.scores) == masks_np.shape[0]:
                    idx = int(torch.argmax(pred_instances.scores).item())
                    pred_score = float(pred_instances.scores[idx].detach().cpu().item())
                mask = (masks_np[idx] > 0).astype(np.uint8)

        if mask.shape[:2] != (src_h, src_w):
            mask = cv2.resize(mask.astype(np.uint8), (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        out = (mask > 0).astype(np.uint8)
        return (out, pred_score) if return_score else out
