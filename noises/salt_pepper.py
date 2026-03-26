import numpy as np
from noises.base import NoiseBase


class SaltPepperNoise(NoiseBase):
    """
    Salt and pepper noise (impulse noise).

    Parameters
    ----------
    amount : float
        Fraction of pixels affected.
        Half of affected pixels become pepper (0),
        half become salt (255).
    """

    PARAM_RANGES = {
        "amount": (0.0, 0.5),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._noise_type = "salt_pepper"

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply salt-and-pepper noise to an image.

        Notes
        -----
        - For grayscale images of shape (H, W), noise is applied per pixel.
        - For multi-channel images of shape (H, W, C), noise is applied per pixel
          and all channels of the selected pixel are set together to 0 or 255.
        """
        amount = float(self.params.get("amount", 0.01))

        y = np.asarray(x).copy()

        if amount <= 0.0:
            return y.astype(np.uint8)

        if y.ndim not in (2, 3):
            raise ValueError(
                f"SaltPepperNoise expects a 2D or 3D image, got shape={y.shape}"
            )

        # Ensure uint8 image domain
        y = np.clip(y, 0, 255).astype(np.uint8)

        h, w = y.shape[:2]
        total_pixels = h * w
        n = int(amount * total_pixels)

        if n <= 0:
            return y

        # Unique pixel positions to avoid excessive overwrite collisions
        indices = self.rng.choice(total_pixels, size=n, replace=False)
        rows, cols = np.unravel_index(indices, (h, w))

        # Split affected pixels into pepper and salt
        half = n // 2
        pepper_rows, pepper_cols = rows[:half], cols[:half]
        salt_rows, salt_cols = rows[half:], cols[half:]

        if y.ndim == 2:
            y[pepper_rows, pepper_cols] = 0
            y[salt_rows, salt_cols] = 255
        else:
            y[pepper_rows, pepper_cols, :] = 0
            y[salt_rows, salt_cols, :] = 255

        return y