"""
SAM 3 model wrapper.
"""

from __future__ import annotations

import gc
import sys
import os
import warnings
from contextlib import nullcontext
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
        self._processor_cls = None
        self._oom_cpu_fallback_used = False

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

            requested_device = str(self.device)
            load_device = "cpu"
            try:
                import torch

                if requested_device.startswith("cuda") and torch.cuda.is_available():
                    if ":" in requested_device:
                        idx = int(requested_device.split(":", 1)[1])
                        torch.cuda.set_device(idx)
                    load_device = "cuda"
                else:
                    load_device = "cpu"
            except Exception:
                load_device = "cpu"
            try:
                self._model = build_sam3_image_model(
                    device=load_device,
                    compile=False,
                    checkpoint_path=ckpt,
                    load_from_HF=False,
                    enable_inst_interactivity=True,
                )
                self.device = requested_device if load_device == "cuda" else "cpu"
            except Exception as e_cuda:
                if str(load_device).startswith("cuda"):
                    warnings.warn(
                        f"SAM3 CUDA load failed ({e_cuda}); retrying on CPU.",
                        RuntimeWarning,
                    )
                    self._model = build_sam3_image_model(
                        device="cpu",
                        compile=False,
                        checkpoint_path=ckpt,
                        load_from_HF=False,
                        enable_inst_interactivity=True,
                    )
                    self.device = "cpu"
                else:
                    raise
            self._processor_cls = Sam3Processor
            self._processor = Sam3Processor(
                self._model, device=self.device
            )
        except Exception as e:
            warnings.warn(f"SAM3 load failed ({e}); using heuristic fallback.", RuntimeWarning)
            self._model = None
            self._processor = None
            self._processor_cls = None

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

        amp_context = nullcontext()
        if self._should_use_amp():
            amp_context = torch.autocast(
                device_type="cuda",
                dtype=self._amp_dtype(torch),
            )

        try:
            with torch.inference_mode():
                with amp_context:
                    inference_state = self._processor.set_image(pil_img)

                    kwargs = build_sam_prompt_kwargs(self.prompt_mode, prompt, batched_box=True)

                    masks, iou_predictions, _ = self._model.predict_inst(
                        inference_state, **kwargs
                    )
            return select_best_mask(masks, iou_predictions)
        except torch.OutOfMemoryError as e:
            self._handle_cuda_oom(torch)
            if self._should_fallback_oom_to_cpu():
                warnings.warn(
                    "SAM3 CUDA OOM during inference; switching SAM3 runner to CPU and retrying.",
                    RuntimeWarning,
                )
                self._switch_to_cpu_processor()
                return self._run_inference(image, prompt)
            raise RuntimeError(
                "SAM3 ran out of CUDA memory. Free VRAM or set model_cfg.oom_fallback_to_cpu=true."
            ) from e

    def _handle_cuda_oom(self, torch_module) -> None:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        gc.collect()

    def _should_use_amp(self) -> bool:
        return bool(self.model_cfg.get("use_amp", True)) and str(self.device).startswith("cuda")

    def _amp_dtype(self, torch_module):
        raw = str(self.model_cfg.get("amp_dtype", "float16")).strip().lower()
        if raw in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        return torch_module.float16

    def _should_fallback_oom_to_cpu(self) -> bool:
        return (
            str(self.device).startswith("cuda")
            and bool(self.model_cfg.get("oom_fallback_to_cpu", True))
            and not self._oom_cpu_fallback_used
        )

    def _switch_to_cpu_processor(self) -> None:
        if self._model is None:
            return
        try:
            self._model.to("cpu")
        except Exception:
            pass
        self.device = "cpu"
        if self._processor_cls is None:
            from sam3.model.sam3_image_processor import Sam3Processor

            self._processor_cls = Sam3Processor
        self._processor = self._processor_cls(self._model, device="cpu")
        self._oom_cpu_fallback_used = True
