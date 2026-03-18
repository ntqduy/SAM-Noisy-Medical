"""Dataset adapter registry – maps adapter names to concrete classes."""

from typing import Dict, Type

from datasets.base_dataset import DatasetAdapter
from datasets.adapters.image_mask_folder_adapter import ImageMaskFolderAdapter
from datasets.adapters.busi_adapter import BUSIAdapter
from datasets.adapters.camus_adapter import CAMUSAdapter


_REGISTRY: Dict[str, Type[DatasetAdapter]] = {
    "ImageMaskFolderAdapter": ImageMaskFolderAdapter,
    "BUSIAdapter": BUSIAdapter,
    "CAMUSAdapter": CAMUSAdapter,
}


def register_adapter(name: str, cls: Type[DatasetAdapter]) -> None:
    _REGISTRY[name] = cls


def build_dataset(cfg: dict) -> DatasetAdapter:
    adapter_name = cfg["adapter"]
    if adapter_name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset adapter: {adapter_name}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[adapter_name](cfg)
