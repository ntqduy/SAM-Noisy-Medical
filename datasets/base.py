from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDatasetAdapter(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Must return:
          { id: str,
            image: uint8 HxW (grayscale),
            gt_mask: uint8 HxW {0,1},
            meta: dict }
        """
        ...
