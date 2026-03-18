"""
BUSIAdapter – Dataset adapter for the BUSI (Breast Ultrasound Images) dataset.

Layout::

    root/
      benign/
        benign (1).png
        benign (1)_mask.png
        ...
      malignant/
        malignant (1).png
        malignant (1)_mask.png
        ...
      normal/
        normal (1).png
        normal (1)_mask.png
        ...

Images and masks live in the **same** folder.  Masks are identified by
the ``_mask`` suffix in the stem.  Files with ``_mask_1`` etc. (secondary
masks) are ignored — only the primary ``_mask`` is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image

from datasets.base_dataset import DatasetAdapter


def _read_gray_uint8(p: Path) -> np.ndarray:
    arr = np.asarray(Image.open(p).convert("L"))
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr


class BUSIAdapter(DatasetAdapter):
    """
    Adapter for the BUSI breast-ultrasound dataset.

    Config keys (YAML)::

        name: "BUSI"
        adapter: "BUSIAdapter"
        root: "data/BUSI/Dataset_BUSI_with_GT"
        categories: ["benign", "malignant"]   # optional, default all 3
        image_exts: [".png"]                  # optional
    """

    def __init__(self, cfg: dict) -> None:
        self.name: str = cfg.get("name", "BUSI")
        self.root = Path(cfg["root"])
        self.categories: List[str] = list(
            cfg.get("categories", ["benign", "malignant", "normal"])
        )
        self.image_exts: Set[str] = {
            e.lower() for e in cfg.get("image_exts", [".png"])
        }

        self.items: List[Tuple[str, Path, Path]] = []
        for cat in self.categories:
            cat_dir = self.root / cat
            if not cat_dir.is_dir():
                continue
            self._collect_category(cat, cat_dir)

        if not self.items:
            raise RuntimeError(
                f"No (image, mask) pairs found in {self.root} "
                f"for categories {self.categories}"
            )

    # ── collection ───────────────────────────────────────────────────────

    def _collect_category(self, cat: str, cat_dir: Path) -> None:
        """Pair each image with its primary ``_mask`` file."""
        images = sorted(
            p
            for p in cat_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.image_exts
            and "_mask" not in p.stem
        )
        for img_path in images:
            mask_path = self._find_mask(img_path)
            if mask_path is None:
                continue
            sid = f"{cat}/{img_path.stem}"
            self.items.append((sid, img_path, mask_path))

    @staticmethod
    def _find_mask(img_path: Path) -> Path | None:
        """Return the primary _mask file for *img_path* (ignore _mask_1 etc.)."""
        stem = img_path.stem
        mask_cand = img_path.with_name(f"{stem}_mask{img_path.suffix}")
        return mask_cand if mask_cand.exists() else None

    # ── interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, img_path, mask_path = self.items[idx]
        mask = _read_gray_uint8(mask_path)
        return {
            "image_id": sid,
            "image": _read_gray_uint8(img_path),
            "mask": (mask > 0).astype(np.uint8),
            "meta": {
                "img_path": str(img_path),
                "mask_path": str(mask_path),
            },
        }
