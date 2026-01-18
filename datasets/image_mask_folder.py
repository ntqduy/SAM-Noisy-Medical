from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from datasets.base import BaseDatasetAdapter


def read_gray_uint8(p: Path) -> np.ndarray:
    arr = np.asarray(Image.open(p).convert("L"))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def read_mask(p: Path, mask_type: str, class_id: int = 1) -> np.ndarray:
    m = read_gray_uint8(p)
    if mask_type == "binary_0_255":
        m = (m > 0).astype(np.uint8)
    elif mask_type == "multiclass":
        m = (m == int(class_id)).astype(np.uint8)
    else:
        raise ValueError(f"Unknown mask.type: {mask_type}")
    return m


class ImageMaskFolderAdapter(BaseDatasetAdapter):
    """
    Generic adapter:
      root/
        image_dir/
        mask_dir/
    mask matches by stem.
    """
    def __init__(self, cfg: dict):
        self.name = cfg["name"]
        self.root = Path(cfg["root"])
        self.image_dir = self.root / cfg["image_dir"]
        self.mask_dir = self.root / cfg["mask_dir"]
        self.image_exts = set([e.lower() for e in cfg.get("image_exts", [".png"])])
        self.mask_exts = set([e.lower() for e in cfg.get("mask_exts", [".png"])])
        self.mask_type = cfg.get("mask", {}).get("type", "binary_0_255")
        self.class_id = int(cfg.get("mask", {}).get("class_id", 1))

        if not self.image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"mask_dir not found: {self.mask_dir}")

        imgs = [p for p in sorted(self.image_dir.iterdir()) if p.is_file() and p.suffix.lower() in self.image_exts]

        items: List[Tuple[str, Path, Path]] = []
        for img_path in imgs:
            stem = img_path.stem
            # try same stem with any mask ext
            mask_path = None
            for ext in self.mask_exts:
                cand = self.mask_dir / f"{stem}{ext}"
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                cands = list(self.mask_dir.glob(stem + ".*"))
                cands = [c for c in cands if c.is_file()]
                if len(cands) == 0:
                    continue
                mask_path = cands[0]
            items.append((stem, img_path, mask_path))

        if len(items) == 0:
            raise RuntimeError(f"No (image,mask) pairs found in {self.root}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, img_path, mask_path = self.items[idx]
        img = read_gray_uint8(img_path)
        gt = read_mask(mask_path, self.mask_type, self.class_id)
        return {
            "id": sid,
            "image": img,
            "gt_mask": gt,
            "meta": {"img_path": str(img_path), "mask_path": str(mask_path)},
        }
