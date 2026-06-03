"""
CAMUSAdapter – Dataset adapter for the CAMUS cardiac ultrasound dataset.

Layout::

    root/              # e.g. data/CAMUS/CAMUS_public/database_nifti
      patient0001/
        patient0001_2CH_ED.nii.gz
        patient0001_2CH_ED_gt.nii.gz
        patient0001_2CH_ES.nii.gz
        patient0001_2CH_ES_gt.nii.gz
        patient0001_4CH_ED.nii.gz
        patient0001_4CH_ED_gt.nii.gz
        ...
      patient0002/
        ...

Each NIfTI file is a 2-D grayscale image (H×W).  Ground-truth labels:
0 = background, 1 = LV endocardium, 2 = myocardium, 3 = LV epicardium.

By default ``class_id: 1`` is used (LV endocardium) → binary mask.
Set ``class_id: 0`` to combine **all** foreground classes (mask > 0).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from datasets.base_dataset import DatasetAdapter


class CAMUSAdapter(DatasetAdapter):
    """
    Adapter for the CAMUS cardiac-ultrasound dataset (NIfTI).

    Config keys (YAML)::

        name: "CAMUS"
        adapter: "CAMUSAdapter"
        root: "data/CAMUS/CAMUS_public/database_nifti"
        views: ["2CH", "4CH"]          # optional, default both
        phases: ["ED", "ES"]           # optional, default both
        class_id: 1                    # 0 = all foreground, 1-3 for specific
        max_patients: null             # optional limit
    """

    def __init__(self, cfg: dict) -> None:
        self.name: str = cfg.get("name", "CAMUS")
        self.root = Path(cfg["root"])
        self.views: List[str] = list(cfg.get("views", ["2CH", "4CH"]))
        self.phases: List[str] = list(cfg.get("phases", ["ED", "ES"]))
        self.class_id: int = int(cfg.get("class_id", 1))
        max_patients: int | None = cfg.get("max_patients")

        self.items: List[Tuple[str, Path, Path]] = []
        self._collect(max_patients)

        if not self.items:
            raise RuntimeError(
                f"No CAMUS image/gt pairs found in {self.root}"
            )

    # ── collection ───────────────────────────────────────────────────────

    def _collect(self, max_patients: int | None) -> None:
        patient_dirs = sorted(
            d
            for d in self.root.iterdir()
            if d.is_dir() and re.match(r"patient\d+", d.name)
        )
        if max_patients:
            patient_dirs = patient_dirs[:max_patients]

        for pdir in patient_dirs:
            pid = pdir.name  # e.g. "patient0001"
            for view in self.views:
                for phase in self.phases:
                    stem = f"{pid}_{view}_{phase}"
                    img_path = pdir / f"{stem}.nii.gz"
                    gt_path = pdir / f"{stem}_gt.nii.gz"
                    if img_path.exists() and gt_path.exists():
                        sid = f"{pid}/{view}_{phase}"
                        self.items.append((sid, img_path, gt_path))

    # ── loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_nifti_2d(p: Path) -> np.ndarray:
        """Load a NIfTI file and return a 2-D float64 array."""
        import nibabel as nib

        data = nib.load(str(p)).get_fdata()
        # Squeeze any trailing singleton dims (e.g. (H, W, 1))
        return np.squeeze(data)

    @staticmethod
    def _spacing_yx(p: Path) -> Tuple[float, float] | None:
        try:
            import nibabel as nib

            zooms = tuple(float(v) for v in nib.load(str(p)).header.get_zooms())
        except Exception:
            return None
        if len(zooms) >= 2 and zooms[0] > 0 and zooms[1] > 0:
            return (zooms[1], zooms[0])
        return None

    def _make_image(self, p: Path) -> np.ndarray:
        arr = self._load_nifti_2d(p)
        # Normalise to uint8
        arr = arr.astype(np.float64)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        return arr.astype(np.uint8)

    def _make_mask(self, p: Path) -> np.ndarray:
        gt = self._load_nifti_2d(p)
        if self.class_id == 0:
            # any foreground
            return (gt > 0).astype(np.uint8)
        return (gt == self.class_id).astype(np.uint8)

    # ── interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, img_path, gt_path = self.items[idx]
        return {
            "image_id": sid,
            "image": self._make_image(img_path),
            "mask": self._make_mask(gt_path),
            "meta": {
                "img_path": str(img_path),
                "gt_path": str(gt_path),
                "spacing": self._spacing_yx(img_path),
            },
        }
