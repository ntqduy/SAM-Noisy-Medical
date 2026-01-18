from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from model.base import BaseModelRunner
from model.prompts import mask_to_bbox


def to_rgb_repeat(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


class SAM1Runner(BaseModelRunner):
    def __init__(self, weight_cfg: dict, mode: str, device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.ckpt = weight_cfg["checkpoint"]
        self.model_type = weight_cfg.get("model_type", "vit_b")
        self._init()

    def _init(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except Exception as e:
            raise RuntimeError(
                "segment-anything not installed. Install:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                f"Error: {e}"
            )
        if not Path(self.ckpt).exists():
            raise FileNotFoundError(self.ckpt)

        sam = sam_model_registry[self.model_type](checkpoint=self.ckpt)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def predict(self, image_gray: np.ndarray, gt_mask: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        rgb = to_rgb_repeat(image_gray)
        self.predictor.set_image(rgb)

        extra = {}

        if self.mode == "prompt_bbox":
            box = mask_to_bbox(gt_mask)
            if box is None:
                return np.zeros(image_gray.shape, dtype=np.uint8), extra
            masks, scores, logits = self.predictor.predict(
                box=box[None, :],
                multimask_output=False
            )
            pred = (masks[0] > 0).astype(np.uint8)
            if scores is not None and len(scores) > 0:
                extra["pred_iou_score"] = float(scores[0])
            return pred, extra

        if self.mode == "automatic":
            # “auto mask” is not native in SAM1 Predictor; simplest fair baseline:
            # use bbox prompt derived from GT OFF (not allowed), so we do:
            # -> return empty and warn. (Bạn có thể plug-in SamAutomaticMaskGenerator nếu muốn.)
            # Để đúng yêu cầu “automatic”, ta dùng AutomaticMaskGenerator (fixed settings).
            from segment_anything import SamAutomaticMaskGenerator

            gen = SamAutomaticMaskGenerator(
                model=self.predictor.model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                min_mask_region_area=0,
            )
            anns = gen.generate(rgb)
            if len(anns) == 0:
                return np.zeros(image_gray.shape, dtype=np.uint8), extra

            # choose mask with largest area (fixed rule)
            best = max(anns, key=lambda a: a.get("area", 0))
            pred = best["segmentation"].astype(np.uint8)
            if "predicted_iou" in best:
                extra["pred_iou_score"] = float(best["predicted_iou"])
            return pred, extra

        raise ValueError(f"Unknown mode: {self.mode}")
