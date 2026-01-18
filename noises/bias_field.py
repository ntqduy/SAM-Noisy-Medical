import numpy as np
import cv2
from noises.base import NoiseBase


class BiasField(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        strength = float(self.params.get("strength", 0.5))  # 0..1
        smooth = int(self.params.get("smooth", 64))

        h, w = x.shape
        field = self.rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        k = max(3, smooth | 1)
        field = cv2.GaussianBlur(field, (k, k), 0)

        field = (field - field.min()) / (field.max() - field.min() + 1e-6)  # 0..1
        field = 1.0 + strength * (field - 0.5) * 2.0  # around 1.0

        y = x.astype(np.float32) * field
        return np.clip(y, 0, 255).astype(np.uint8)
