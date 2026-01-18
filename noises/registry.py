from typing import Dict, Optional

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

_REG = {
    "clean": None,
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

def build_noise(name: str, p: float, params: Dict, seed: int = 42):
    name = (name or "clean").lower()
    cls = _REG.get(name)
    if cls is None:
        return None
    return lambda x: cls(p=p, params=params, seed=seed).maybe_apply(x)
