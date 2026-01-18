import numpy as np
from noises.base import NoiseBase


class SaltPepperNoise(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        amount = float(self.params.get("amount", 0.01))
        y = x.copy().astype(np.uint8)
        n = int(amount * y.size)
        if n <= 0:
            return y
        coords = self.rng.integers(0, y.size, size=n)
        y.flat[coords] = 255
        coords = self.rng.integers(0, y.size, size=n)
        y.flat[coords] = 0
        return y
