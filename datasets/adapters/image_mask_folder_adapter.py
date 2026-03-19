"""
ImageMaskFolderAdapter – generic directory-based dataset adapter.

Expected layout (one of):
  root/image_dir/           root/split/image_dir/
  root/mask_dir/            root/split/mask_dir/

Or explicit ``sources`` list in the YAML config.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from datasets.base_dataset import DatasetAdapter


# ── helpers ──────────────────────────────────────────────────────────────

def _read_gray_uint8(p: Path) -> np.ndarray:
    arr = np.asarray(Image.open(p).convert("L"))
    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr


def _read_mask(
    p: Path,
    mask_type: str,
    class_id: int = 1,
    binary_threshold: int = 0,
) -> np.ndarray:
    m = _read_gray_uint8(p)
    if mask_type == "binary_0_255":
        return (m > int(binary_threshold)).astype(np.uint8)
    if mask_type == "multiclass":
        return (m == int(class_id)).astype(np.uint8)
    raise ValueError(f"Unknown mask.type: {mask_type}")


# ── adapter ──────────────────────────────────────────────────────────────

class ImageMaskFolderAdapter(DatasetAdapter):
    """
    Generic adapter: ``root/image_dir`` + ``root/mask_dir``.

    Mask is matched by stem (+ common medical suffixes).
    """

    def __init__(self, cfg: dict) -> None:
        self.name: str = cfg["name"]
        self.root = Path(cfg["root"])
        self.image_dir_name: str = cfg["image_dir"]
        self.mask_dir_name: str = cfg["mask_dir"]
        self.image_exts: Set[str] = {e.lower() for e in cfg.get("image_exts", [".png"])}
        self.mask_exts: Set[str] = {e.lower() for e in cfg.get("mask_exts", [".png"])}
        mask_cfg = cfg.get("mask", {})
        self.mask_type: str = mask_cfg.get("type", "binary_0_255")
        self.class_id: int = int(mask_cfg.get("class_id", 1))
        default_threshold = 127 if any(e in {".jpg", ".jpeg"} for e in self.mask_exts) else 0
        self.mask_threshold: int = int(mask_cfg.get("threshold", default_threshold))
        self.split_dirs: List[str] = list(cfg.get("split_dirs", ["train", "val", "test"]))
        self.sources_cfg: List[dict] = list(cfg.get("sources", []))

        pair_roots = self._discover_pair_roots()
        if not pair_roots:
            raise FileNotFoundError(
                f"No valid (image_dir, mask_dir) found under {self.root}. "
                f"Checked {self.image_dir_name}/{self.mask_dir_name}."
            )

        items: List[Tuple[str, Path, Path]] = []
        for split_name, image_dir, mask_dir in pair_roots:
            for img_path in sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.image_exts
            ):
                mask_path = self._match_mask(img_path, mask_dir)
                if mask_path is None:
                    continue
                rel = img_path.relative_to(self.root)
                sid = str(rel.with_suffix("")).replace("\\", "/")
                if split_name:
                    sid = f"{split_name}:{sid}"
                items.append((sid, img_path, mask_path))

        if not items:
            raise RuntimeError(f"No (image, mask) pairs found in {self.root}")
        self.items = items

    # ── discovery ────────────────────────────────────────────────────────
    def _discover_pair_roots(self) -> List[Tuple[str, Path, Path]]:
        if self.sources_cfg:
            manual: List[Tuple[str, Path, Path]] = []
            for i, src in enumerate(self.sources_cfg):
                if not isinstance(src, dict):
                    continue
                img_d = self.root / str(src.get("image_dir", self.image_dir_name))
                msk_d = self.root / str(src.get("mask_dir", self.mask_dir_name))
                if img_d.exists() and msk_d.exists():
                    manual.append((str(src.get("name", f"src{i}")), img_d, msk_d))
            if manual:
                return manual

        candidates: List[Tuple[str, Path, Path]] = []
        d_img = self.root / self.image_dir_name
        d_msk = self.root / self.mask_dir_name
        if d_img.exists() and d_msk.exists():
            candidates.append(("", d_img, d_msk))
        for split in self.split_dirs:
            s_img = self.root / split / self.image_dir_name
            s_msk = self.root / split / self.mask_dir_name
            if s_img.exists() and s_msk.exists():
                candidates.append((split, s_img, s_msk))

        seen: set = set()
        deduped: List[Tuple[str, Path, Path]] = []
        for name, img, msk in candidates:
            key = (str(img.resolve()), str(msk.resolve()))
            if key not in seen:
                seen.add(key)
                deduped.append((name, img, msk))
        return deduped

    # ── mask matching ────────────────────────────────────────────────────
    def _match_mask(self, img_path: Path, mask_dir: Path) -> Optional[Path]:
        stem = img_path.stem
        for ext in self.mask_exts:
            cand = mask_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        for suffix in ("_mask", "_gt", "_label", "_seg", "_segmentation"):
            for ext in self.mask_exts:
                cand = mask_dir / f"{stem}{suffix}{ext}"
                if cand.exists():
                    return cand
        cands = sorted(
            (c for c in mask_dir.glob(f"{stem}*") if c.is_file() and c.suffix.lower() in self.mask_exts),
            key=lambda p: len(p.stem),
        )
        return cands[0] if cands else None

    # ── interface ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, img_path, mask_path = self.items[idx]
        return {
            "image_id": sid,
            "image": _read_gray_uint8(img_path),
            "mask": _read_mask(
                mask_path,
                self.mask_type,
                self.class_id,
                self.mask_threshold,
            ),
            "meta": {"img_path": str(img_path), "mask_path": str(mask_path)},
        }
