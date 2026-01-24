"""Noise injection modules for medical imaging artifact simulation.

Extended for AIO25 NoisySAM project with:
  - NoiseResult dataclass with full metadata (PSNR, SSIM, intensity_scalar)
  - PARAM_RANGES for all noise types
  - Enhanced registry with metadata support
"""
from noises.base import NoiseBase, NoiseResult, CleanNoise, compute_psnr, compute_ssim
from noises.registry import build_noise, get_noise_class, list_available_noises
from noises.presets import (
    get_all_presets,
    get_preset_for_level,
    DEFAULT_COUPLED_PRESETS,
    PHASE2_OPTIONAL_PRESETS,
)

__all__ = [
    "NoiseBase",
    "NoiseResult",
    "CleanNoise",
    "compute_psnr",
    "compute_ssim",
    "build_noise",
    "get_noise_class",
    "list_available_noises",
    "get_all_presets",
    "get_preset_for_level",
    "DEFAULT_COUPLED_PRESETS",
    "PHASE2_OPTIONAL_PRESETS",
]
