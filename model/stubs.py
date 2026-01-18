import warnings
import numpy as np
from typing import Any, Dict, Tuple
from model.base import BaseModelRunner

# (SAM2/SAM3/MedSAM placeholder)
class StubRunner(BaseModelRunner):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        warnings.warn(f"[WARN] {name} runner is a stub. Implement integration in models/{name.lower()}.py")

    def predict(self, image_gray: np.ndarray, gt_mask: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return np.zeros(image_gray.shape, dtype=np.uint8), {}
