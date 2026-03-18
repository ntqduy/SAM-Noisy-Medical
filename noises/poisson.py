import numpy as np
from noises.base import NoiseBase


class PoissonNoise(NoiseBase):
    """
    Poisson noise (shot noise).
    
    Parameters:
        lam: Lambda parameter (higher = more noise relative to signal)
    """
    
    PARAM_RANGES = {
        "lam": (1.0, 80.0),  # Lambda range (includes full_benchmark L1=80)
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "poisson"

    def get_severity_scalar(self) -> float:
        """Lower lambda yields stronger shot noise, so invert normalization."""
        lam = float(self.params.get("lam", 20.0))
        min_lam, max_lam = self.PARAM_RANGES["lam"]
        if max_lam <= min_lam:
            return 0.5
        normalized = 1.0 - (lam - min_lam) / (max_lam - min_lam)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        lam = float(self.params.get("lam", 20.0))
        x_f = x.astype(np.float32) / 255.0
        # poisson on scaled intensity
        noisy = self.rng.poisson(lam * x_f).astype(np.float32) / max(lam, 1e-6)
        y = noisy * 255.0
        return np.clip(y, 0, 255).astype(np.uint8)
