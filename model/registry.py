"""
Model registry for SAM, SAM2, SAM3, MedSAM runners.
Supports multi-model + multi-weights architecture.
"""
import warnings
from typing import Dict, Any

from model.base import BaseModelRunner


# Lazy imports to avoid import errors if dependencies not installed
def _get_sam1_runner():
    from model.sam1 import SAM1Runner
    return SAM1Runner


def _get_sam2_runner():
    from model.sam2 import SAM2Runner
    return SAM2Runner


def _get_medsam_runner():
    from model.medsam import MedSAMRunner
    return MedSAMRunner


def _get_sam3_runner():
    from model.sam3 import SAM3Runner
    return SAM3Runner


_RUNNER_FACTORY = {
    "SAM1": _get_sam1_runner,
    "SAM": _get_sam1_runner,       # alias
    "SAM2": _get_sam2_runner,
    "MedSAM": _get_medsam_runner,
    "SAM3": _get_sam3_runner,
}


def build_model_runner(runner_key: str, weight_cfg: Dict[str, Any], mode: str, device: str = "cpu") -> BaseModelRunner:
    """
    Build a model runner given:
      - runner_key: e.g. "SAM1", "SAM2", "MedSAM", "SAM3"
      - weight_cfg: dict with "checkpoint", "model_type", "id", etc.
      - mode: "prompt_bbox", "automatic", "prompt_points", etc.
      - device: "cpu" or "cuda"
    
    Returns a BaseModelRunner instance.
    """
    if runner_key not in _RUNNER_FACTORY:
        raise ValueError(f"Unknown runner: {runner_key}. Available: {list(_RUNNER_FACTORY.keys())}")
    
    try:
        runner_cls = _RUNNER_FACTORY[runner_key]()
        return runner_cls(weight_cfg=weight_cfg, mode=mode, device=device)
    except FileNotFoundError as e:
        warnings.warn(f"[WARN] Checkpoint not found for {runner_key}: {e}")
        raise
    except Exception as e:
        warnings.warn(f"[WARN] Failed to build {runner_key} runner: {e}")
        raise


def list_available_runners():
    """List all registered runner keys."""
    return list(_RUNNER_FACTORY.keys())
