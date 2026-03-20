import numpy as np
from noises.base import NoiseBase


class LowContrast(NoiseBase):
    """
    Low contrast degradation.
    
    Parameters:
        alpha: Contrast multiplier (<1 reduces contrast)
        beta: Brightness offset
    """
    
    PARAM_RANGES = {
        "alpha": (0.1, 1.0),  # Wider range for aggressive low-contrast schedules
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "low_contrast"
    
    def get_severity_scalar(self) -> float:
        """For low_contrast, lower alpha = higher severity."""
        alpha = float(self.params.get("alpha", 0.75))
        min_alpha, max_alpha = self.PARAM_RANGES["alpha"]
        # Invert: lower alpha = higher severity
        normalized = 1.0 - (alpha - min_alpha) / (max_alpha - min_alpha)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        alpha = float(self.params.get("alpha", 0.75))  # <1 reduce contrast
        beta = float(self.params.get("beta", 0.0))
        # Contrast change around mid-gray keeps global brightness more stable.
        y = (x.astype(np.float32) - 128.0) * alpha + 128.0 + beta
        return np.clip(y, 0, 255).astype(np.uint8)
