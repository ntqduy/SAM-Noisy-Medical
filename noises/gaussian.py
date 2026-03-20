import numpy as np
from noises.base import NoiseBase


class GaussianNoise(NoiseBase):
    """
    Additive Gaussian noise.
    
    Parameters:
        sigma: Standard deviation of noise (supports stronger levels up to ~80+)
    """
    
    PARAM_RANGES = {
        "sigma": (0.0, 90.0),  # Keep headroom for aggressive L8-L9 setups
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "gaussian"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        sigma = float(self.params.get("sigma", 10.0))
        y = x.astype(np.float32) + self.rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        return np.clip(y, 0, 255).astype(np.uint8)
