"""
Poisson noise (shot noise / photon noise) implementation.

Poisson noise models the statistical uncertainty in photon counting processes,
which is fundamental in imaging systems. It's signal-dependent: brighter regions
have more photons and thus higher absolute noise but lower relative noise.

Mathematical model:
    Given clean signal x (normalized to [0,1]) and peak photon count:
    - Photon counts: N ~ Poisson(x × peak)
    - Noisy signal: y = N / peak

    Properties:
    - E[y] = x (unbiased)
    - Var[y] = x / peak (signal-dependent variance)
    - SNR = √(x × peak) (improves with more photons)

Parameter semantics:
    - Higher peak → more photons → higher SNR → LESS noise (milder)
    - Lower peak → fewer photons → lower SNR → MORE noise (severe)

    This is an INVERTED severity mapping:
    - L1 (mild): peak = 80 (high SNR)
    - L9 (severe): peak = 4 (low SNR)

References:
    - Foi et al. (2008) "Practical Poissonian-Gaussian Noise Modeling"
    - Makitalo & Foi (2011) "Optimal Inversion of the Anscombe Transformation"
"""

import numpy as np
from noises.base import NoiseBase


class PoissonNoise(NoiseBase):
    """
    Poisson noise (shot noise / photon noise).

    This noise models photon counting statistics in imaging systems.
    It's signal-dependent: the noise variance scales with the signal intensity.

    Parameters
    ----------
    peak : float
        Peak photon count (equivalent to the maximum number of photons
        at full intensity). Controls the signal-to-noise ratio.

        - Higher peak → more photons → less relative noise (high SNR)
        - Lower peak → fewer photons → more relative noise (low SNR)

        Typical ranges:
        - 60-100: mild noise (high SNR, like good imaging conditions)
        - 20-60: moderate noise
        - 5-20: strong noise (low SNR, like low-light conditions)
        - 1-5: severe noise (very few photons)

    Notes
    -----
    - This uses INVERTED severity mapping: lower peak = higher severity
    - Input images are expected as uint8 in [0, 255] range
    - For multi-channel images, noise is applied independently per channel
    - Zero-valued pixels remain zero (no photons → no noise)

    Examples
    --------
    >>> noise = PoissonNoise(p=1.0, params={"peak": 20}, seed=42)
    >>> result = noise(image)
    >>> noisy_image = result.noisy_image
    """

    PARAM_RANGES = {
        "peak": (1.0, 100.0),  # Peak photon count range
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "poisson"

    def get_severity_scalar(self) -> float:
        """
        Compute normalized severity scalar in [0, 1].

        INVERTED mapping: lower peak = higher severity (more noise).

        Returns
        -------
        float
            Normalized severity where 0.0 = no noise (high peak),
            1.0 = maximum noise (low peak).
        """
        # Support both 'peak' (new) and 'lam' (legacy) parameter names
        peak = float(self.params.get("peak", self.params.get("lam", 20.0)))
        min_peak, max_peak = self.PARAM_RANGES["peak"]

        if max_peak <= min_peak:
            return 0.5

        # INVERTED: lower peak = higher severity
        normalized = 1.0 - (peak - min_peak) / (max_peak - min_peak)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Poisson noise using standard photon counting model.

        The noise is applied using:
            y = Poisson(x_normalized × peak) / peak × 255

        This correctly models:
        - Signal-dependent noise (brighter regions have more absolute noise)
        - SNR scaling with √peak (more photons = cleaner image)

        Parameters
        ----------
        x : np.ndarray
            Input image, expected uint8 with values in [0, 255].
            Shape: (H, W) for grayscale or (H, W, C) for multi-channel.

        Returns
        -------
        np.ndarray
            Noisy image with Poisson noise, same shape as input,
            dtype uint8, clipped to [0, 255].
        """
        # Support both 'peak' (new) and 'lam' (legacy) parameter names
        peak = float(self.params.get("peak", self.params.get("lam", 20.0)))

        # Ensure peak is positive
        peak = max(peak, 1e-6)

        # Ensure proper array handling
        arr = np.asarray(x)

        # Normalize to [0, 1]
        x_normalized = arr.astype(np.float64) / 255.0

        # Apply Poisson noise model:
        # - Scale signal by peak (simulating photon counts)
        # - Sample from Poisson distribution
        # - Normalize back by dividing by peak
        #
        # This gives: E[y] = x, Var[y] = x/peak
        # SNR = E[y]/√Var[y] = √(x × peak)

        # Generate Poisson samples
        # For each pixel, sample from Poisson(x * peak)
        photon_counts = self.rng.poisson(x_normalized * peak)

        # Normalize back to [0, 1] range
        noisy_normalized = photon_counts.astype(np.float64) / peak

        # Scale back to [0, 255]
        noisy = noisy_normalized * 255.0

        # Clip and convert to uint8
        return np.clip(noisy, 0.0, 255.0).astype(np.uint8)
