import numpy as np
from noises.base import NoiseBase


class GaussianNoise(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        sigma = float(self.params.get("sigma", 10.0))
        y = x.astype(np.float32) + self.rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        return np.clip(y, 0, 255).astype(np.uint8)
