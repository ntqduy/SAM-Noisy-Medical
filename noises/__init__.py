"""Noise injection modules for the segmentation robustness benchmark."""

from noises.base import NoiseBase, NoiseResult, CleanNoise, compute_psnr, compute_ssim
from noises.noise_registry import get_noise_class, list_available_noises, register_noise
from noises.noise_manager import NoiseManager
from noises.severity_mapping import (
    SEVERITY_MAPPING_DIRECTION,
    get_severity_direction,
    is_inverted_severity,
)

__all__ = [
    "NoiseBase",
    "NoiseResult",
    "CleanNoise",
    "compute_psnr",
    "compute_ssim",
    "get_noise_class",
    "list_available_noises",
    "register_noise",
    "NoiseManager",
    # Severity mapping utilities
    "SEVERITY_MAPPING_DIRECTION",
    "get_severity_direction",
    "is_inverted_severity",
]
