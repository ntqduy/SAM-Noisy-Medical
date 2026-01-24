import numpy as np
from noises.base import NoiseBase


class SaltPepperNoise(NoiseBase):
    """
    Salt and pepper noise (impulse noise).
    
    Parameters:
        amount: Fraction of pixels affected (0.0 - 0.1 typical)
    """
    
    PARAM_RANGES = {
        "amount": (0.0, 0.1),  # Fraction of pixels
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "salt_pepper"
    
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
