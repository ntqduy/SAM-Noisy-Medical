"""Dataset adapters for the segmentation robustness benchmark."""

from datasets.base_dataset import DatasetAdapter
from datasets.adapters.image_mask_folder_adapter import ImageMaskFolderAdapter
from datasets.adapters.busi_adapter import BUSIAdapter
from datasets.adapters.camus_adapter import CAMUSAdapter
from datasets.dataset_registry import build_dataset, register_adapter

__all__ = [
    "DatasetAdapter",
    "ImageMaskFolderAdapter",
    "BUSIAdapter",
    "CAMUSAdapter",
    "build_dataset",
    "register_adapter",
]
