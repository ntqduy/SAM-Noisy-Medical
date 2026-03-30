"""
MedicoSAM model wrapper.

MedicoSAM publishes an exported SAM-compatible ViT-B checkpoint, so we can
reuse the standard SAM inference path while keeping a distinct model identity
and default checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models.wrappers.sam_runner import SAMRunner


class MedicoSAMRunner(SAMRunner):
    """Wrapper for MedicoSAM using the SAM1 predictor interface."""

    def __init__(
        self,
        model_name: str = "MEDICOSAM",
        prompt_mode: str = "prompt_bbox",
        device: str = "cpu",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = dict(model_cfg or {})
        cfg.setdefault("checkpoint", "weights/vit_b_medicosam.pt")
        cfg.setdefault("model_type", "vit_b")
        super().__init__(model_name, prompt_mode, device, cfg)
