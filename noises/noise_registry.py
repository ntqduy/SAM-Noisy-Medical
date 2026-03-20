"""
Noise class registry – maps noise names to concrete ``NoiseBase`` subclasses.
"""

from typing import Dict, List, Optional, Type

from noises.base import NoiseBase, CleanNoise
from noises.gaussian import GaussianNoise
from noises.poisson import PoissonNoise
from noises.salt_pepper import SaltPepperNoise
from noises.motion_blur import MotionBlur
from noises.bias_field import BiasField
from noises.low_contrast import LowContrast
from noises.optional_extras import (
    SpeckleNoise,
    UniformNoise,
    JPEGArtifacts,
    PixelationNoise,
    LowBrightnessNoise,
    HighBrightnessNoise,
    HighContrastNoise,
    QuantizationNoise,
    DefocusBlur,
    CoarseDropout,
    GridMask,
)

_REGISTRY: Dict[str, Type[NoiseBase]] = {
    "clean": CleanNoise,
    "gaussian": GaussianNoise,
    "poisson": PoissonNoise,
    "salt_pepper": SaltPepperNoise,
    "motion_blur": MotionBlur,
    "bias_field": BiasField,
    "low_contrast": LowContrast,
    # optional / phase-2
    "speckle": SpeckleNoise,
    "uniform": UniformNoise,
    "jpeg": JPEGArtifacts,
    "pixelation": PixelationNoise,
    "low_brightness": LowBrightnessNoise,
    "high_brightness": HighBrightnessNoise,
    "high_contrast": HighContrastNoise,
    "quantization": QuantizationNoise,
    "defocus_blur": DefocusBlur,
    "coarse_dropout": CoarseDropout,
    "gridmask": GridMask,
}


def register_noise(name: str, cls: Type[NoiseBase]) -> None:
    _REGISTRY[name.lower()] = cls


def get_noise_class(name: str) -> Optional[Type[NoiseBase]]:
    return _REGISTRY.get((name or "clean").lower())


def list_available_noises() -> List[str]:
    return list(_REGISTRY.keys())
