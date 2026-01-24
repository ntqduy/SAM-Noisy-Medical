"""
Noise registry for AIO25 NoisySAM benchmark.
Provides factory functions for building noise instances with full metadata tracking.
"""
from typing import Dict, Optional, Type, Callable

from noises.base import NoiseBase, NoiseResult, CleanNoise
from noises.gaussian import GaussianNoise
from noises.poisson import PoissonNoise
from noises.salt_pepper import SaltPepperNoise
from noises.motion_blur import MotionBlur
from noises.bias_field import BiasField
from noises.low_contrast import LowContrast

from noises.optional_extras import (
    SpeckleNoise, UniformNoise, JPEGArtifacts, QuantizationNoise,
    DefocusBlur, CoarseDropout, GridMask
)

# Registry mapping noise names to classes
_REG: Dict[str, Optional[Type[NoiseBase]]] = {
    "clean": CleanNoise,
    "gaussian": GaussianNoise,
    "poisson": PoissonNoise,
    "salt_pepper": SaltPepperNoise,
    "motion_blur": MotionBlur,
    "bias_field": BiasField,
    "low_contrast": LowContrast,

    # optional phase2
    "speckle": SpeckleNoise,
    "uniform": UniformNoise,
    "jpeg": JPEGArtifacts,
    "quantization": QuantizationNoise,
    "defocus_blur": DefocusBlur,
    "coarse_dropout": CoarseDropout,
    "gridmask": GridMask,
}


def get_noise_class(name: str) -> Optional[Type[NoiseBase]]:
    """Get noise class by name."""
    name = (name or "clean").lower()
    return _REG.get(name)


def list_available_noises() -> list:
    """List all registered noise types."""
    return list(_REG.keys())


def build_noise(
    name: str,
    p: float,
    params: Dict,
    seed: int = 42,
    level: str = "L0",
    protocol: str = "P1",
    compute_distortion: bool = True
) -> Optional[Callable]:
    """
    Build a noise callable that returns NoiseResult.
    
    Args:
        name: Noise type name (gaussian, poisson, etc.)
        p: Probability of noise application
        params: Noise-specific parameters
        seed: Random seed for reproducibility
        level: Noise level (L0..L4)
        protocol: Protocol type (P0/P1/P2a/P2b/P3)
        compute_distortion: Whether to compute PSNR/SSIM
        
    Returns:
        Callable that takes image and returns NoiseResult,
        or None for clean baseline
    """
    name = (name or "clean").lower()
    cls = _REG.get(name)
    
    if cls is None:
        return None
    
    noise_instance = cls(
        p=p,
        params=params,
        seed=seed,
        level=level,
        protocol=protocol,
        compute_distortion=compute_distortion
    )
    
    return noise_instance


def build_noise_legacy(name: str, p: float, params: Dict, seed: int = 42):
    """
    Legacy interface: returns callable that returns just the noisy image.
    For backward compatibility with existing code.
    """
    name = (name or "clean").lower()
    cls = _REG.get(name)
    if cls is None:
        return None
    
    def apply_noise(x):
        instance = cls(p=p, params=params, seed=seed)
        return instance.maybe_apply(x)
    
    return apply_noise


def get_noise_param_ranges(name: str) -> Dict[str, tuple]:
    """Get parameter ranges for a noise type."""
    cls = get_noise_class(name)
    if cls is None:
        return {}
    return getattr(cls, "PARAM_RANGES", {})
