from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseModelRunner(ABC):
    @abstractmethod
    def predict(self, image_gray: np.ndarray, gt_mask: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Return:
          pred_mask: uint8 HxW {0,1}
          extra: dict (optional confidence proxies)
        """
        ...
