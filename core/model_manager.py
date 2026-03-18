"""
ModelManager – factory / registry for instantiating model wrappers.

Usage::

    mgr = ModelManager(device="cuda")
    runner = mgr.get_model("SAM2", prompt_mode="prompt_bbox", model_cfg={...})
    runner.load_model()
    mask = runner.predict(image, prompt)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Type

from models.wrappers.base_model import ModelRunner
from models.wrappers.sam_runner import SAMRunner
from models.wrappers.sam2_runner import SAM2Runner
from models.wrappers.sam3_runner import SAM3Runner
from models.wrappers.medsam_runner import MedSAMRunner
from models.wrappers.medsam2_runner import MedSAM2Runner
from models.wrappers.medsam3_runner import MedSAM3Runner
from models.wrappers.sam_med2d_runner import SAMMed2DRunner
from models.wrappers.ultrasam_runner import UltraSAMRunner


_DEFAULT_REGISTRY: Dict[str, Type[ModelRunner]] = {
    "SAM": SAMRunner,
    "SAM1": SAMRunner,
    "SAM2": SAM2Runner,
    "SAM3": SAM3Runner,
    "MEDSAM": MedSAMRunner,
    "MEDSAM1": MedSAMRunner,
    "MEDSAM2": MedSAM2Runner,
    "MEDSAM3": MedSAM3Runner,
    "SAM-MED2D": SAMMed2DRunner,
    "ULTRASAM": UltraSAMRunner,
}


class ModelManager:
    """Central factory for building model runners."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._registry: Dict[str, Type[ModelRunner]] = dict(_DEFAULT_REGISTRY)

    # ── public API ───────────────────────────────────────────────────────

    def get_model(
        self,
        model_name: str,
        prompt_mode: str = "prompt_bbox",
        model_cfg: Optional[Dict[str, Any]] = None,
    ) -> ModelRunner:
        """Instantiate a model runner by name and load its weights."""
        key = model_name.strip().upper()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )
        cls = self._registry[key]
        runner = cls(
            model_name=model_name,
            prompt_mode=prompt_mode,
            device=self.device,
            model_cfg=model_cfg or {},
        )
        runner.load_model()
        return runner

    def register(self, name: str, cls: Type[ModelRunner]) -> None:
        """Register a custom model wrapper at runtime."""
        self._registry[name.strip().upper()] = cls

    def list_models(self) -> list[str]:
        return sorted(self._registry)
