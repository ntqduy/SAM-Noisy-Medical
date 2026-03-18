"""Base dataset adapter interface for the segmentation robustness benchmark."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class DatasetAdapter(ABC):
    """
    Abstract base class for all dataset adapters.

    Every dataset adapter must expose:
      - ``name``        – human-readable dataset name
      - ``__len__``     – number of samples
      - ``__getitem__`` – returns a standardised sample dict
    """

    name: str = ""

    # ------------------------------------------------------------------
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a sample dictionary with **at least** these keys:

        * ``image_id``  : ``str``  – unique identifier for the image
        * ``image``     : ``np.ndarray`` (uint8, HxW grayscale)
        * ``mask``      : ``np.ndarray`` (uint8, HxW, values {0, 1})

        Optional:
        * ``meta``      : ``dict`` – any extra metadata
        """
        ...

    # convenience ---------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, len={len(self)})"
