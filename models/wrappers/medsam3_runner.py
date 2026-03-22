"""
MedSAM3 model wrapper – SAM3 base + LoRA fine-tuned weights.

Uses ``sam3`` package from ``models/external/sam3`` and LoRA from
``models/external/MedSAM3``.
Default base weight:  ``weights/sam3.pt``
Default LoRA weight:  ``weights/MedSAM3/best_lora_weights.pt``
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


class MedSAM3Runner(ModelRunner):
    """
    Wrapper for MedSAM3 (SAM3 + LoRA fine-tuned for medical imaging).

    Loads the SAM3 base model with ``enable_inst_interactivity=True``
    (SAM1-compatible point/box prompt interface), then applies LoRA
    weights on top.

    Falls back to heuristic inference when the model package is unavailable.
    """

    def __init__(
        self,
        model_name: str = "MEDSAM3",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_name, normalize_prompt_mode(prompt_mode), device, model_cfg)
        self._model = None       # Sam3Image model
        self._processor = None   # Sam3Processor for set_image
        self._processor_cls = None
        self._oom_cpu_fallback_used = False

    @staticmethod
    def _resolve_existing_path(project_root: str, raw_path: str, *extra_candidates: str) -> str:
        candidates = []
        if raw_path:
            candidates.append(raw_path)
            if not os.path.isabs(raw_path):
                candidates.append(os.path.join(project_root, raw_path))
                candidates.append(os.path.join(project_root, "weights", os.path.basename(raw_path)))
        candidates.extend(extra_candidates)

        checked = []
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            normalized = os.path.abspath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            checked.append(normalized)
            if os.path.exists(normalized):
                return normalized

        raise FileNotFoundError(
            f"Required MedSAM3 file was not found. Checked: {', '.join(checked)}"
        )

    def load_model(self) -> None:
        try:
            external_root = os.path.join(
                os.path.dirname(__file__), os.pardir, "external"
            )
            external_root = os.path.abspath(external_root)
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )

            sam3_root = os.path.join(external_root, "sam3")
            medsam3_root = os.path.join(external_root, "MedSAM3")

            for p in (sam3_root, medsam3_root):
                if p not in sys.path:
                    sys.path.insert(0, p)

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            base_ckpt = self._resolve_existing_path(
                project_root,
                self.model_cfg.get("checkpoint", "weights/sam3.pt"),
            )
            lora_weights = self._resolve_existing_path(
                project_root,
                self.model_cfg.get("lora_weights", "weights/MedSAM3/best_lora_weights.pt"),
                os.path.join(project_root, "weights", "MedSAM3", "best_lora_weights.pt"),
            )
            bpe_path = self._resolve_existing_path(
                project_root,
                self.model_cfg.get("bpe_path", ""),
                os.path.join(medsam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
                os.path.join(medsam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
                os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
            )

            requested_device = str(self.device)
            load_device = "cpu"
            try:
                import torch

                if requested_device.startswith("cuda") and torch.cuda.is_available():
                    if ":" in requested_device:
                        idx = int(requested_device.split(":", 1)[1])
                        torch.cuda.set_device(idx)
                    min_free_gb = float(self.model_cfg.get("min_cuda_free_gb_for_load", 6.0))
                    try:
                        free_bytes, _ = torch.cuda.mem_get_info()
                        free_gb = float(free_bytes) / float(1024**3)
                    except Exception:
                        free_gb = min_free_gb
                    if free_gb >= min_free_gb:
                        load_device = "cuda"
                    else:
                        warnings.warn(
                            f"MedSAM3 detected low free CUDA memory ({free_gb:.2f} GiB). "
                            "Loading MedSAM3 directly on CPU to avoid CUDA OOM.",
                            RuntimeWarning,
                        )
                        load_device = "cpu"
                else:
                    load_device = "cpu"
            except Exception:
                load_device = "cpu"
            try:
                self._model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    device=load_device,
                    compile=False,
                    checkpoint_path=base_ckpt,
                    load_from_HF=False,
                    enable_inst_interactivity=True,
                    eval_mode=False,
                )
                self.device = requested_device if load_device == "cuda" else "cpu"
            except Exception as e_cuda:
                if str(load_device).startswith("cuda"):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    warnings.warn(
                        f"MedSAM3 CUDA load failed ({e_cuda}); retrying on CPU.",
                        RuntimeWarning,
                    )
                    self._model = build_sam3_image_model(
                        bpe_path=bpe_path,
                        device="cpu",
                        compile=False,
                        checkpoint_path=base_ckpt,
                        load_from_HF=False,
                        enable_inst_interactivity=True,
                        eval_mode=False,
                    )
                    self.device = "cpu"
                else:
                    raise

            self._apply_lora(lora_weights)

            self._model.to(self.device)
            self._model.eval()

            self._processor_cls = Sam3Processor
            self._processor = Sam3Processor(
                self._model, device=self.device
            )

        except Exception as e:
            warnings.warn(
                f"MedSAM3 load failed ({e}); using heuristic fallback.",
                RuntimeWarning,
            )
            self._model = None
            self._processor = None
            self._processor_cls = None

    def _apply_lora(self, lora_weights_path: str) -> None:
        """Apply LoRA adapter and load fine-tuned weights."""
        from lora_layers import (
            LoRAConfig,
            apply_lora_to_model,
            load_lora_weights,
        )

        lora_cfg = self.model_cfg.get("lora", {})
        lora_config = LoRAConfig(
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 32),
            dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", None),
            apply_to_vision_encoder=lora_cfg.get("apply_to_vision_encoder", True),
            apply_to_text_encoder=lora_cfg.get("apply_to_text_encoder", True),
            apply_to_geometry_encoder=lora_cfg.get("apply_to_geometry_encoder", True),
            apply_to_detr_encoder=lora_cfg.get("apply_to_detr_encoder", True),
            apply_to_detr_decoder=lora_cfg.get("apply_to_detr_decoder", True),
            apply_to_mask_decoder=lora_cfg.get("apply_to_mask_decoder", True),
        )
        self._model = apply_lora_to_model(self._model, lora_config)
        load_lora_weights(self._model, lora_weights_path)

    # ── inference ────────────────────────────────────────────────────────

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
                    "MedSAM3 CUDA OOM during inference; switching MedSAM3 runner to CPU and retrying.",
                    RuntimeWarning,
                )
                self._switch_to_cpu_processor()
                return self._run_inference(image, prompt)
            raise RuntimeError(
                "MedSAM3 ran out of CUDA memory. Free VRAM or set model_cfg.oom_fallback_to_cpu=true."
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
