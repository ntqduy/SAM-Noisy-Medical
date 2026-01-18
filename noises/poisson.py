import numpy as np
from noises.base import NoiseBase


class PoissonNoise(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        lam = float(self.params.get("lam", 20.0))
        x_f = x.astype(np.float32) / 255.0
        # poisson on scaled intensity
        noisy = self.rng.poisson(lam * x_f).astype(np.float32) / max(lam, 1e-6)
        y = noisy * 255.0
        return np.clip(y, 0, 255).astype(np.uint8)
