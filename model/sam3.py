"""
SAM3 (Segment Anything 3) model runner - placeholder for future model.
This is a stub implementation that can be extended when SAM3 is released.
"""
from typing import Any, Dict, Tuple
import warnings
import numpy as np

from model.base import BaseModelRunner


class SAM3Runner(BaseModelRunner):
    """
    SAM3 runner placeholder.
    
    This is a stub implementation. When SAM3 becomes available,
    implement the actual model loading and inference logic here.
    """
    
    def __init__(self, weight_cfg: dict, mode: str, device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.ckpt = weight_cfg.get("checkpoint", "")
        self.model_type = weight_cfg.get("model_type", "sam3_default")
        self.weight_id = weight_cfg.get("id", "sam3")
        
        warnings.warn(
            f"[WARN] SAM3 runner is a placeholder. "
            f"SAM3 is not yet released. Returning empty predictions."
        )

    def predict(self, image_gray: np.ndarray, gt_mask: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        SAM3 prediction placeholder.
        
        Returns empty mask until SAM3 implementation is available.
        """
        warnings.warn("[WARN] SAM3 not implemented, returning empty mask")
        return np.zeros(image_gray.shape, dtype=np.uint8), {}
