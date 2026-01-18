"""Dataset adapters for SAM benchmark."""
from datasets.base import BaseDatasetAdapter
from datasets.image_mask_folder import ImageMaskFolderAdapter
from datasets.registry import build_dataset

__all__ = ["BaseDatasetAdapter", "ImageMaskFolderAdapter", "build_dataset"]
