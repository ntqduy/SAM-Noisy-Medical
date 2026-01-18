from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class NoiseBase(ABC):
    def __init__(self, p: float, params: Dict, seed: int = 42):
        self.p = float(p)
        self.params = dict(params or {})
        self.rng = np.random.default_rng(seed)

    def maybe_apply(self, x: np.ndarray) -> np.ndarray:
        if self.p <= 0:
            return x
        if self.p >= 1.0:
            return self.apply(x)
        if float(self.rng.random()) < self.p:
            return self.apply(x)
        return x

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        pass