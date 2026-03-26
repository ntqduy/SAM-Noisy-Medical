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
    """JPEG compression artifacts.

    - Nếu input là grayscale 2D: nén JPEG trên ảnh xám.
    - Nếu input là RGB/BGR 3D: nén JPEG trực tiếp trên ảnh màu.
    - Giữ output cùng số chiều / số kênh với input.
    """

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
        """Apply JPEG compression artifacts while preserving input domain.

        Parameters
        ----------
        x : np.ndarray
            Input image. Supported:
            - HxW
            - HxWx1
            - HxWx3

        Returns
        -------
        np.ndarray
            Image with JPEG artifacts, same shape as input.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input x must be a numpy array.")

        quality = int(self.params.get("quality", 40))
        quality = max(1, min(95, quality))

        x_uint8 = self._to_uint8_image(x)

        # Case 1: grayscale 2D
        if x_uint8.ndim == 2:
            ok, enc = cv2.imencode(
                ".jpg",
                x_uint8,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                return x.copy()

            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            return self._restore_dtype_and_range(dec, x)

        # Case 2: HxWx1 grayscale
        if x_uint8.ndim == 3 and x_uint8.shape[2] == 1:
            gray = x_uint8[:, :, 0]
            ok, enc = cv2.imencode(
                ".jpg",
                gray,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                return x.copy()

            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            dec = dec[:, :, None]
            return self._restore_dtype_and_range(dec, x)

        # Case 3: HxWx3 color
        if x_uint8.ndim == 3 and x_uint8.shape[2] == 3:
            ok, enc = cv2.imencode(
                ".jpg",
                x_uint8,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if not ok:
                return x.copy()

            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            return self._restore_dtype_and_range(dec, x)

        raise ValueError(
            f"Unsupported input shape {x.shape}. Expected HxW, HxWx1, or HxWx3."
        )

    @staticmethod
    def _to_uint8_image(x: np.ndarray) -> np.ndarray:
        """Convert image to uint8 safely.

        Supports:
        - uint8 input: keep as is
        - float input in [0, 1]: scale to [0, 255]
        - float input in [0, 255]: clip and cast
        """
        if x.dtype == np.uint8:
            return x.copy()

        x_float = x.astype(np.float32)

        if x_float.max() <= 1.0:
            x_float = x_float * 255.0

        x_float = np.clip(x_float, 0.0, 255.0)
        return x_float.astype(np.uint8)

    @staticmethod
    def _restore_dtype_and_range(out_uint8: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Restore output to match dtype/range style of reference input."""
        if ref.dtype == np.uint8:
            return out_uint8

        out = out_uint8.astype(np.float32)

        if np.issubdtype(ref.dtype, np.floating):
            if ref.max() <= 1.0:
                out = out / 255.0
            return out.astype(ref.dtype)

        return out.astype(ref.dtype)
class PixelationNoise(NoiseBase):
    """Pixelation / blocky downsample-upsample artifacts."""

    PARAM_RANGES = {"block_size": (1, 50)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "pixelation"

    def get_severity_scalar(self) -> float:
        block_size = float(self.params.get("block_size", 8))
        min_b, max_b = self.PARAM_RANGES["block_size"]
        normalized = (block_size - min_b) / (max_b - min_b)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        block_size = int(self.params.get("block_size", 8))
        block_size = max(1, block_size)

        h, w = x.shape[:2]
        small_w = max(1, w // block_size)
        small_h = max(1, h // block_size)

        small = cv2.resize(x, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return restored.astype(x.dtype, copy=False)

class LowBrightnessNoise(NoiseBase):
    """Darken image globally by multiplicative factor (<1) while preserving channels."""

    PARAM_RANGES = {"factor": (0.05, 1.0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "low_brightness"

    def get_severity_scalar(self) -> float:
        min_f, max_f = self.PARAM_RANGES["factor"]
        factor = float(np.clip(self.params.get("factor", 0.75), min_f, max_f))
        normalized = 1.0 - (factor - min_f) / (max_f - min_f)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        min_f, max_f = self.PARAM_RANGES["factor"]
        factor = float(np.clip(self.params.get("factor", 0.75), min_f, max_f))

        arr = np.asarray(x).astype(np.float32)
        out = np.clip(arr * factor, 0, 255).astype(np.uint8)
        return out

class HighBrightnessNoise(NoiseBase):
    """Brighten image globally by a multiplicative factor (>1) while preserving channels."""

    PARAM_RANGES = {"factor": (1.0, 5.0)}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "high_brightness"

    def get_severity_scalar(self) -> float:
        """
        Map brightness factor to a normalized severity in [0, 1].

        Returns
        -------
        float
            0.0 means no brightness increase (factor=1.0),
            1.0 means maximum configured brightness increase.
        """
        min_f, max_f = self.PARAM_RANGES["factor"]
        factor = float(np.clip(self.params.get("factor", 1.25), min_f, max_f))
        normalized = (factor - min_f) / (max_f - min_f)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply global brightness increase while preserving the original channel structure.

        Parameters
        ----------
        x : np.ndarray
            Input image. Expected dtype is typically uint8, but other numeric
            dtypes are also accepted.

        Returns
        -------
        np.ndarray
            Brightened image with the same shape and dtype as the input when possible.
        """
        min_f, max_f = self.PARAM_RANGES["factor"]
        factor = float(np.clip(self.params.get("factor", 1.25), min_f, max_f))

        arr = np.asarray(x)
        orig_dtype = arr.dtype

        arr_f = arr.astype(np.float32)
        out = np.clip(arr_f * factor, 0, 255)

        if np.issubdtype(orig_dtype, np.integer):
            return out.astype(orig_dtype)

        return out.astype(np.float32)

class HighContrastNoise(NoiseBase):
    """Increase image contrast around the mid-gray pivot 128 while preserving channels."""

    PARAM_RANGES = {"factor": (1.0, 10.0)}  # Extended to cover config L6-L9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_type = "high_contrast"

    def get_severity_scalar(self) -> float:
        """
        Map contrast factor to a normalized severity in [0, 1].

        Returns
        -------
        float
            0.0 means no contrast increase (factor=1.0),
            1.0 means maximum configured contrast increase.
        """
        min_f, max_f = self.PARAM_RANGES["factor"]
        factor = float(np.clip(self.params.get("factor", 1.4), min_f, max_f))
        normalized = (factor - min_f) / (max_f - min_f)
        return float(np.clip(normalized, 0.0, 1.0))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply high-contrast degradation while preserving the original channel structure.

        Parameters
        ----------
        x : np.ndarray
            Input image, typically uint8.

        Returns
        -------
        np.ndarray
            Contrast-enhanced image in uint8 format.
        """
        min_f, max_f = self.PARAM_RANGES["factor"]
        factor = float(np.clip(self.params.get("factor", 1.4), min_f, max_f))

        arr = np.asarray(x).astype(np.float32)
        out = (arr - 128.0) * factor + 128.0
        return np.clip(out, 0, 255).astype(np.uint8)


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
        h, w = y.shape[:2]
        for _ in range(holes):
            cx = int(self.rng.integers(0, w))
            cy = int(self.rng.integers(0, h))
            x0 = max(0, cx - size // 2); x1 = min(w, cx + size // 2)
            y0 = max(0, cy - size // 2); y1 = min(h, cy + size // 2)
            if y.ndim == 2:
                y[y0:y1, x0:x1] = 0
            else:
                y[y0:y1, x0:x1, :] = 0
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
        h, w = y.shape[:2]
        for yy in range(0, h, d):
            if y.ndim == 2:
                y[yy:yy+r, :] = 0
            else:
                y[yy:yy+r, :, :] = 0
        for xx in range(0, w, d):
            if y.ndim == 2:
                y[:, xx:xx+r] = 0
            else:
                y[:, xx:xx+r, :] = 0
        return y.astype(np.uint8)
