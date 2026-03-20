"""
Optional/Phase 2 noise types for AIO25 NoisySAM benchmark.
Each noise includes PARAM_RANGES for severity normalization.
"""
import numpy as np
import cv2
from noises.base import NoiseBase


def _to_gray_uint8(x: np.ndarray) -> np.ndarray:
    """Convert input image to 2D uint8 grayscale."""
    arr = np.asarray(x)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = np.squeeze(arr)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _restore_like_input(gray: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match output channel layout to input (HxW or HxWxC)."""
    if np.asarray(ref).ndim == 3:
        channels = int(np.asarray(ref).shape[2])
        return np.repeat(gray[..., None], channels, axis=2).astype(np.uint8)
    return gray.astype(np.uint8)


class SpeckleNoise(NoiseBase):
    """Multiplicative speckle noise (common in ultrasound/radar)."""
    
    PARAM_RANGES = {"sigma": (0.0, 0.9)}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "speckle"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        sigma = float(self.params.get("sigma", 0.08))  # relative
        n = self.rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        y = x.astype(np.float32) + x.astype(np.float32) * n
        return np.clip(y, 0, 255).astype(np.uint8)


class UniformNoise(NoiseBase):
    """Additive uniform noise."""
    
    PARAM_RANGES = {"b": (0.0, 50.0)}  # Use upper bound as primary
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "uniform"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        a = float(self.params.get("a", -10.0))
        b = float(self.params.get("b", 10.0))
        y = x.astype(np.float32) + self.rng.uniform(a, b, size=x.shape).astype(np.float32)
        return np.clip(y, 0, 255).astype(np.uint8)


class JPEGArtifacts(NoiseBase):
    """JPEG compression artifacts."""
    
    PARAM_RANGES = {"quality": (5, 95)}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "jpeg"
    
    def get_severity_scalar(self) -> float:
        """Lower quality = higher severity."""
        quality = float(self.params.get("quality", 40))
        min_q, max_q = self.PARAM_RANGES["quality"]
        normalized = 1.0 - (quality - min_q) / (max_q - min_q)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        quality = int(self.params.get("quality", 40))
        gray = _to_gray_uint8(x)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(95, quality))]
        ok, enc = cv2.imencode(".jpg", rgb, encode_param)
        if not ok:
            return x
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        out_gray = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        return _restore_like_input(out_gray, x)


class PixelationNoise(NoiseBase):
    """Pixelation / blocky downsample-upsample artifacts."""

    PARAM_RANGES = {"block_size": (1, 50)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "pixelation"

    def apply(self, x: np.ndarray) -> np.ndarray:
        block_size = int(self.params.get("block_size", 8))
        block_size = max(1, block_size)
        gray = _to_gray_uint8(x)
        h, w = gray.shape[:2]
        small_w = max(1, w // block_size)
        small_h = max(1, h // block_size)
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return _restore_like_input(restored.astype(np.uint8), x)


class LowBrightnessNoise(NoiseBase):
    """Darken image globally by multiplicative factor (<1)."""

    PARAM_RANGES = {"factor": (0.05, 1.0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "low_brightness"

    def get_severity_scalar(self) -> float:
        factor = float(self.params.get("factor", 0.75))
        min_f, max_f = self.PARAM_RANGES["factor"]
        normalized = 1.0 - (factor - min_f) / (max_f - min_f)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        factor = float(self.params.get("factor", 0.75))
        gray = _to_gray_uint8(x).astype(np.float32)
        out = np.clip(gray * factor, 0, 255).astype(np.uint8)
        return _restore_like_input(out, x)


class HighBrightnessNoise(NoiseBase):
    """Brighten image globally by multiplicative factor (>1)."""

    PARAM_RANGES = {"factor": (1.0, 5.0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "high_brightness"

    def apply(self, x: np.ndarray) -> np.ndarray:
        factor = float(self.params.get("factor", 1.25))
        gray = _to_gray_uint8(x).astype(np.float32)
        out = np.clip(gray * factor, 0, 255).astype(np.uint8)
        return _restore_like_input(out, x)


class HighContrastNoise(NoiseBase):
    """Increase contrast around mid-gray pivot 128."""

    PARAM_RANGES = {"factor": (1.0, 8.0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "high_contrast"

    def apply(self, x: np.ndarray) -> np.ndarray:
        factor = float(self.params.get("factor", 1.4))
        gray = _to_gray_uint8(x).astype(np.float32)
        out = (gray - 128.0) * factor + 128.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        return _restore_like_input(out, x)


class QuantizationNoise(NoiseBase):
    """Quantization/posterization noise."""
    
    PARAM_RANGES = {"step": (1, 64)}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "quantization"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        step = int(self.params.get("step", 8))
        step = max(1, step)
        y = (x // step) * step
        return y.astype(np.uint8)


class DefocusBlur(NoiseBase):
    """Defocus/Gaussian blur."""
    
    PARAM_RANGES = {"k": (3, 31)}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "defocus_blur"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        k = int(self.params.get("k", 7))
        k = max(3, k | 1)
        y = cv2.GaussianBlur(x, (k, k), 0)
        return y.astype(np.uint8)


class CoarseDropout(NoiseBase):
    """Random rectangular region dropout."""
    
    PARAM_RANGES = {"holes": (1, 20), "size": (8, 64)}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "coarse_dropout"
    
    def get_severity_scalar(self) -> float:
        """Combine holes and size for severity."""
        holes = int(self.params.get("holes", 8))
        size = int(self.params.get("size", 24))
        h_norm = (holes - 1) / (20 - 1)
        s_norm = (size - 8) / (64 - 8)
        return float(np.clip((h_norm + s_norm) / 2, 0.0, 1.0))
    
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
    """Grid-based masking/dropout."""
    
    PARAM_RANGES = {"d": (16, 96), "r": (8, 48)}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "gridmask"
    
    def get_severity_scalar(self) -> float:
        """Ratio of masked area."""
        d = int(self.params.get("d", 48))
        r = int(self.params.get("r", 24))
        # Approximate mask ratio
        mask_ratio = min(1.0, (2 * r) / d)
        return float(np.clip(mask_ratio, 0.0, 1.0))
    
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
