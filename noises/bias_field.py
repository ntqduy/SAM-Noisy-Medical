import numpy as np
import cv2
from noises.base import NoiseBase


class BiasField(NoiseBase):
    """
    Bias field (intensity inhomogeneity) common in medical imaging.
    
    Parameters:
        strength: Intensity of the bias field (0.0 - 1.0)
        smooth: Smoothness of the field (kernel size)
    """
    
    PARAM_RANGES = {
        "strength": (0.0, 1.0),  # Bias field strength
        "smooth": (16, 256),  # Smoothness kernel
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "bias_field"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        strength = float(self.params.get("strength", 0.5))  # 0..1
        smooth = int(self.params.get("smooth", 64))

        arr = np.asarray(x)
        h, w = arr.shape[:2]

        # Generate smooth random bias field (2D)
        field = self.rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        k = max(3, smooth | 1)
        field = cv2.GaussianBlur(field, (k, k), 0)

        field = (field - field.min()) / (field.max() - field.min() + 1e-6)  # 0..1
        field = 1.0 + strength * (field - 0.5) * 2.0  # around 1.0

        # Apply field: for multi-channel, broadcast field over all channels
        if arr.ndim == 3:
            field = field[..., np.newaxis]  # (H, W, 1) for broadcasting

        y = arr.astype(np.float32) * field
        return np.clip(y, 0, 255).astype(np.uint8)
