import numpy as np
import cv2
from noises.base import NoiseBase


def motion_kernel(k: int, angle: float) -> np.ndarray:
    k = max(3, int(k) | 1)  # odd
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k // 2, :] = 1.0
    # rotate
    M = cv2.getRotationMatrix2D((k/2, k/2), angle, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    s = kern.sum()
    return kern / s if s > 0 else kern


class MotionBlur(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        k = int(self.params.get("k", 9))
        angle = float(self.params.get("angle", 15.0))
        kern = motion_kernel(k, angle)
        y = cv2.filter2D(x, -1, kern)
        return y.astype(np.uint8)
