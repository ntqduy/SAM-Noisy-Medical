import numpy as np
from noises.base import NoiseBase


class LowContrast(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        alpha = float(self.params.get("alpha", 0.75))  # <1 reduce contrast
        beta = float(self.params.get("beta", 0.0))
        y = x.astype(np.float32) * alpha + beta
        return np.clip(y, 0, 255).astype(np.uint8)
