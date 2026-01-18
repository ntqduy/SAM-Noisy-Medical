import numpy as np
import cv2
from noises.base import NoiseBase


class SpeckleNoise(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        sigma = float(self.params.get("sigma", 0.08))  # relative
        n = self.rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        y = x.astype(np.float32) + x.astype(np.float32) * n
        return np.clip(y, 0, 255).astype(np.uint8)


class UniformNoise(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        a = float(self.params.get("a", -10.0))
        b = float(self.params.get("b", 10.0))
        y = x.astype(np.float32) + self.rng.uniform(a, b, size=x.shape).astype(np.float32)
        return np.clip(y, 0, 255).astype(np.uint8)


class JPEGArtifacts(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        quality = int(self.params.get("quality", 40))
        rgb = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(95, quality))]
        ok, enc = cv2.imencode(".jpg", rgb, encode_param)
        if not ok:
            return x
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.uint8)


class QuantizationNoise(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        step = int(self.params.get("step", 8))
        step = max(1, step)
        y = (x // step) * step
        return y.astype(np.uint8)


class DefocusBlur(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        k = int(self.params.get("k", 7))
        k = max(3, k | 1)
        y = cv2.GaussianBlur(x, (k, k), 0)
        return y.astype(np.uint8)


class CoarseDropout(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        holes = int(self.params.get("holes", 8))
        size = int(self.params.get("size", 24))
        y = x.copy()
        h, w = y.shape
        for _ in range(holes):
            cx = int(self.rng.integers(0, w))
            cy = int(self.rng.integers(0, h))
            x0 = max(0, cx - size // 2); x1 = min(w, cx + size // 2)
            y0 = max(0, cy - size // 2); y1 = min(h, cy + size // 2)
            y[y0:y1, x0:x1] = 0
        return y.astype(np.uint8)


class GridMask(NoiseBase):
    def apply(self, x: np.ndarray) -> np.ndarray:
        d = int(self.params.get("d", 48))
        r = int(self.params.get("r", 24))
        y = x.copy()
        h, w = y.shape
        for yy in range(0, h, d):
            y[yy:yy+r, :] = 0
        for xx in range(0, w, d):
            y[:, xx:xx+r] = 0
        return y.astype(np.uint8)
