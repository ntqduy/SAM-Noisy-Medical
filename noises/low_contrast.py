import numpy as np
from noises.base import NoiseBase


class LowContrastNoise(NoiseBase):
    """
    Reduce image contrast by shrinking intensities toward mid-gray.

    Parameters
    ----------
    alpha : float
        Contrast multiplier in [0.1, 1.0]. Lower values produce lower contrast.
    beta : float
        Optional brightness offset added after contrast adjustment.
    """

    PARAM_RANGES = {
        "alpha": (0.1, 1.0),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "low_contrast"

    def get_severity_scalar(self) -> float:
        """
        Map contrast reduction strength to normalized severity in [0, 1].

        Returns
        -------
        float
            0.0 means no contrast reduction (alpha=1.0),
            1.0 means strongest configured contrast reduction.
        """
        min_alpha, max_alpha = self.PARAM_RANGES["alpha"]
        alpha = float(np.clip(self.params.get("alpha", 0.75), min_alpha, max_alpha))
        normalized = 1.0 - (alpha - min_alpha) / (max_alpha - min_alpha)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply low-contrast degradation while preserving channel structure.

        Parameters
        ----------
        x : np.ndarray
            Input image, typically uint8.

        Returns
        -------
        np.ndarray
            Low-contrast image in uint8 format.
        """
        min_alpha, max_alpha = self.PARAM_RANGES["alpha"]
        alpha = float(np.clip(self.params.get("alpha", 0.75), min_alpha, max_alpha))
        beta = float(self.params.get("beta", 0.0))

        arr = np.asarray(x).astype(np.float32)
        out = (arr - 128.0) * alpha + 128.0 + beta
        return np.clip(out, 0, 255).astype(np.uint8)