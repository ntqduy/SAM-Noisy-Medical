"""
Base noise classes and NoiseResult dataclass for AIO25 NoisySAM.
Provides unified noise application with full metadata tracking.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np


@dataclass
class NoiseResult:
    """
    Unified result from noise application with full metadata tracking.
    
    Attributes:
        noisy_image: The noise-corrupted image (uint8 HxW or HxWxC)
        meta: Dictionary containing noise metadata:
            - noise_type: str - Name of noise (gaussian, poisson, etc.)
            - protocol: str - P0/P1/P2a/P2b/P3
            - level: str - L0..L4 or custom
            - p: float - Probability of application
            - severity_scalar: float - Normalized severity in [0,1]
            - intensity_scalar: float - Overall intensity in [0,1] (combines p and severity)
            - severity_params: dict - Noise-specific parameters
            - noise_seed: int - Random seed used
            - applied: bool - Whether noise was actually applied (probabilistic)
            - psnr: float (optional) - PSNR vs clean image
            - ssim: float (optional) - SSIM vs clean image
    """
    noisy_image: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def noise_type(self) -> str:
        return self.meta.get("noise_type", "unknown")
    
    @property
    def level(self) -> str:
        return self.meta.get("level", "L0")
    
    @property
    def p(self) -> float:
        return self.meta.get("p", 1.0)
    
    @property
    def severity_scalar(self) -> float:
        return self.meta.get("severity_scalar", 0.0)
    
    @property
    def intensity_scalar(self) -> float:
        return self.meta.get("intensity_scalar", 0.0)
    
    @property
    def noise_seed(self) -> int:
        return self.meta.get("noise_seed", 0)


def compute_psnr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    clean = clean.astype(np.float64)
    noisy = noisy.astype(np.float64)
    mse = np.mean((clean - noisy) ** 2)
    if mse < 1e-10:
        return 100.0  # Perfect match
    max_val = 255.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def compute_ssim(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Compute Structural Similarity Index (simplified version)."""
    try:
        from skimage.metrics import structural_similarity as ssim
        kwargs = {"data_range": 255}
        if np.asarray(clean).ndim == 3:
            kwargs["channel_axis"] = -1
        return float(ssim(clean, noisy, **kwargs))
    except ImportError:
        # Fallback: simplified SSIM approximation
        clean = clean.astype(np.float64)
        noisy = noisy.astype(np.float64)
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        mu_x = np.mean(clean)
        mu_y = np.mean(noisy)
        sigma_x = np.std(clean)
        sigma_y = np.std(noisy)
        sigma_xy = np.mean((clean - mu_x) * (noisy - mu_y))
        
        ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        return float(ssim_val)
    except Exception:
        clean = clean.astype(np.float64)
        noisy = noisy.astype(np.float64)

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_x = np.mean(clean)
        mu_y = np.mean(noisy)
        sigma_x = np.std(clean)
        sigma_y = np.std(noisy)
        sigma_xy = np.mean((clean - mu_x) * (noisy - mu_y))

        ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        return float(ssim_val)


class NoiseBase(ABC):
    """
    Base class for all noise types.
    
    Each noise subclass must implement:
        - apply(x) -> noisy image
        - get_severity_scalar() -> float in [0,1]
        - PARAM_RANGES: class attribute mapping param names to (min, max) tuples
    """
    
    # Subclasses should override with their parameter ranges for normalization
    PARAM_RANGES: Dict[str, tuple] = {}
    
    def __init__(self, p: float, params: Dict, seed: int = 42, 
                 level: str = "L0", protocol: str = "P1",
                 compute_distortion: bool = True):
        self.p = float(p)
        self.params = dict(params or {})
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.level = level
        self.protocol = protocol
        self.compute_distortion = compute_distortion
        self._noise_type = self.__class__.__name__.lower().replace("noise", "").replace("blur", "_blur")

    def maybe_apply(self, x: np.ndarray) -> np.ndarray:
        """Legacy interface: apply noise probabilistically, return image only."""
        if self.p <= 0:
            return x
        if self.p >= 1.0:
            return self.apply(x)
        if float(self.rng.random()) < self.p:
            return self.apply(x)
        return x
    
    def __call__(self, x: np.ndarray, return_meta: bool = False) -> NoiseResult:
        """
        Apply noise and return ``NoiseResult`` with full metadata.

        Parameters
        ----------
        x : np.ndarray
            Clean input image, typically uint8 in [0, 255].
        return_meta : bool, optional
            Deprecated compatibility flag. The method always returns
            ``NoiseResult`` in the current framework.

        Returns
        -------
        NoiseResult
            Structured result containing ``noisy_image`` and metadata.
        """
        clean = x.copy()
        applied = False
        
        if self.p <= 0:
            noisy = x.copy()
        elif self.p >= 1.0:
            noisy = self.apply(x)
            applied = True
        elif float(self.rng.random()) < self.p:
            noisy = self.apply(x)
            applied = True
        else:
            noisy = x.copy()
        
        severity_scalar = self.get_severity_scalar()
        intensity_scalar = self.p * severity_scalar  # Combined intensity
        
        meta = {
            "noise_type": self._noise_type,
            "protocol": self.protocol,
            "level": self.level,
            "p": self.p,
            "severity_scalar": severity_scalar,
            "intensity_scalar": intensity_scalar,
            "severity_params": dict(self.params),
            "noise_seed": self.seed,
            "applied": applied,
        }
        
        # Compute distortion metrics if enabled and noise was applied
        if self.compute_distortion and applied:
            meta["psnr"] = compute_psnr(clean, noisy)
            meta["ssim"] = compute_ssim(clean, noisy)
        
        if not return_meta:
            # Legacy behavior: return NoiseResult but caller may just use .noisy_image
            pass
        
        return NoiseResult(noisy_image=noisy, meta=meta)
    
    def get_severity_scalar(self) -> float:
        """
        Compute normalized severity scalar in [0, 1].
        
        Uses PARAM_RANGES to normalize the primary parameter.
        Subclasses can override for custom normalization.
        """
        if not self.PARAM_RANGES:
            return 0.5  # Default if no ranges defined
        
        # Use first parameter as primary
        primary_param = list(self.PARAM_RANGES.keys())[0]
        if primary_param not in self.params:
            return 0.5
        
        val = float(self.params[primary_param])
        min_val, max_val = self.PARAM_RANGES[primary_param]
        
        if max_val <= min_val:
            return 0.5
        
        normalized = (val - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the noise transformation. Must be implemented by subclasses."""
        pass


class CleanNoise(NoiseBase):
    """No-op noise for clean baseline (P0/L0)."""
    
    PARAM_RANGES = {}
    
    def __init__(self, **kwargs):
        kwargs.setdefault("p", 0.0)
        kwargs.setdefault("params", {})
        super().__init__(**kwargs)
        self._noise_type = "clean"
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.copy()
    
    def get_severity_scalar(self) -> float:
        return 0.0
