"""
Rician noise implementation for MRI magnitude images.

Rician noise arises in magnitude MRI images due to Gaussian noise in the
complex (real + imaginary) k-space domain. The magnitude reconstruction
produces a non-Gaussian, signal-dependent noise distribution.

Mathematical model:
    Given clean signal x and noise standard deviation σ:
    - Real component:      R = x + n1,  where n1 ~ N(0, σ²)
    - Imaginary component: I = n2,      where n2 ~ N(0, σ²)
    - Noisy magnitude:     M = sqrt(R² + I²) = sqrt((x + n1)² + n2²)

Properties:
    - High SNR (x >> σ): approximately Gaussian
    - Low SNR (x << σ): approximately Rayleigh (positive bias)
    - Always non-negative (physical constraint)

References:
    - Gudbjartsson & Patz (1995) "The Rician Distribution of Noisy MRI Data"
    - Macovski (1996) "Noise in MRI"
"""

import numpy as np
from noises.base import NoiseBase


class RicianNoise(NoiseBase):
    """
    Rician noise typical of MRI magnitude images.

    This noise model simulates the effect of Gaussian noise in the complex
    (quadrature) domain on magnitude-reconstructed MR images. Unlike additive
    Gaussian noise, Rician noise exhibits signal-dependent behavior with a
    positive bias in low-signal regions.

    Parameters
    ----------
    sigma : float
        Standard deviation of the underlying Gaussian noise in the complex
        domain. The effective noise level in the magnitude image depends on
        both sigma and the local signal intensity.

        Typical ranges for 8-bit images [0, 255]:
        - 5-15: mild noise (high SNR)
        - 15-30: moderate noise
        - 30-50: strong noise (low SNR regime)
        - 50+: severe noise (Rayleigh-dominated in dark regions)

    Notes
    -----
    - Input images are expected as uint8 in [0, 255] range.
    - The output preserves the input shape and dtype.
    - For RGB/multi-channel images, noise is applied channel-wise with
      the same sigma but independent noise realizations per channel.
    - This implementation assumes phase = 0 (real-valued positive signal),
      which is standard for magnitude MRI modeling.

    Examples
    --------
    >>> noise = RicianNoise(p=1.0, params={"sigma": 20}, seed=42)
    >>> result = noise(image)
    >>> noisy_image = result.noisy_image
    """

    PARAM_RANGES = {
        "sigma": (0.0, 80.0),  # Consistent with GaussianNoise upper headroom
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._noise_type = "rician"

    def get_severity_scalar(self) -> float:
        """
        Compute normalized severity scalar in [0, 1].

        Higher sigma corresponds to higher severity (standard mapping).

        Returns
        -------
        float
            Normalized severity where 0.0 = no noise, 1.0 = maximum noise.
        """
        sigma = float(self.params.get("sigma", 20.0))
        min_val, max_val = self.PARAM_RANGES["sigma"]

        if max_val <= min_val:
            return 0.5

        normalized = (sigma - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Rician noise to an image.

        The noise is applied using the magnitude reconstruction model:
            M = sqrt((x + n1)² + n2²)
        where n1, n2 ~ N(0, σ²) are independent Gaussian noise realizations.

        Parameters
        ----------
        x : np.ndarray
            Input image, expected uint8 with values in [0, 255].
            Shape: (H, W) for grayscale or (H, W, C) for multi-channel.

        Returns
        -------
        np.ndarray
            Noisy image with Rician noise applied, same shape as input,
            dtype uint8, clipped to [0, 255].
        """
        sigma = float(max(0.0, self.params.get("sigma", 20.0)))
        arr = np.asarray(x)
        if sigma <= 0.0:
            return np.clip(arr, 0, 255).astype(np.uint8, copy=True)

        x_f = arr.astype(np.float32, copy=False)

        # Standard MRI magnitude-noise construction:
        # M = sqrt((x + n1)^2 + n2^2), with n1, n2 iid N(0, sigma^2).
        n1 = self.rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        n2 = self.rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        magnitude = np.hypot(x_f + n1, n2)

        return np.clip(magnitude, 0.0, 255.0).astype(np.uint8, copy=False)
