"""
SAM2 (Segment Anything 2) model runner.
Supports both prompt-based and automatic modes.
"""
from pathlib import Path
from typing import Any, Dict, Tuple
import warnings
import numpy as np

from model.base import BaseModelRunner
from model.prompts import mask_to_bbox


def to_rgb_repeat(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale to RGB by repeating channels."""
    return np.stack([gray, gray, gray], axis=-1)


class SAM2Runner(BaseModelRunner):
    """
    SAM2 runner with support for:
      - prompt_bbox: bounding box from GT
      - prompt_points: points from GT (optional)
      - automatic: automatic mask generation
    """
    
    def __init__(self, weight_cfg: dict, mode: str, device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.ckpt = weight_cfg.get("checkpoint", "")
        self.model_type = weight_cfg.get("model_type", "sam2_t")  # sam2_t, sam2_s, sam2_b, sam2_l
        self.weight_id = weight_cfg.get("id", "sam2")
        self._sam2 = None
        self._predictor = None
        self._init()

    def _init(self):
        """Initialize SAM2 model and predictor."""
        try:
            # Try sam2 package (official)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            if not self.ckpt or not Path(self.ckpt).exists():
                raise FileNotFoundError(f"SAM2 checkpoint not found: {self.ckpt}")
            
            # SAM2 model config mapping
            config_map = {
                "sam2_t": "sam2_hiera_t.yaml",
                "sam2_s": "sam2_hiera_s.yaml",
                "sam2_b": "sam2_hiera_b+.yaml",
                "sam2_l": "sam2_hiera_l.yaml",
            }
            config = config_map.get(self.model_type, "sam2_hiera_t.yaml")
            
            self._sam2 = build_sam2(config, self.ckpt, device=self.device)
            self._predictor = SAM2ImagePredictor(self._sam2)
            
        except ImportError:
            # Fallback: try alternative import path
            try:
                from segment_anything_2.build_sam import build_sam2
                from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
                
                if not self.ckpt or not Path(self.ckpt).exists():
                    raise FileNotFoundError(f"SAM2 checkpoint not found: {self.ckpt}")
                
                self._sam2 = build_sam2(self.model_type, self.ckpt, device=self.device)
                self._predictor = SAM2ImagePredictor(self._sam2)
                
            except ImportError as e:
                warnings.warn(
                    f"[WARN] SAM2 not installed. Install via:\n"
                    f"  pip install git+https://github.com/facebookresearch/segment-anything-2.git\n"
                    f"Error: {e}"
                )
                # Use stub mode
                self._predictor = None

    def predict(self, image_gray: np.ndarray, gt_mask: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run SAM2 prediction.
        
        Args:
            image_gray: uint8 HxW grayscale image
            gt_mask: uint8 HxW ground truth mask {0,1}
            meta: additional metadata
            
        Returns:
            pred_mask: uint8 HxW predicted mask {0,1}
            extra: dict with optional confidence proxies
        """
        extra = {}
        
        if self._predictor is None:
            warnings.warn("[WARN] SAM2 predictor not initialized, returning empty mask")
            return np.zeros(image_gray.shape, dtype=np.uint8), extra
        
        rgb = to_rgb_repeat(image_gray)
        self._predictor.set_image(rgb)
        
        if self.mode == "prompt_bbox":
            return self._predict_bbox(gt_mask, extra)
        elif self.mode == "prompt_points":
            return self._predict_points(gt_mask, extra)
        elif self.mode == "automatic":
            return self._predict_automatic(rgb, extra)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _predict_bbox(self, gt_mask: np.ndarray, extra: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict using bounding box prompt from GT."""
        box = mask_to_bbox(gt_mask)
        if box is None:
            return np.zeros(gt_mask.shape, dtype=np.uint8), extra
        
        masks, scores, logits = self._predictor.predict(
            box=box[None, :],
            multimask_output=False
        )
        
        pred = (masks[0] > 0).astype(np.uint8)
        
        if scores is not None and len(scores) > 0:
            extra["pred_iou_score"] = float(scores[0])
        
        return pred, extra

    def _predict_points(self, gt_mask: np.ndarray, extra: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict using point prompts from GT centroid."""
        from model.prompts import mask_to_center_point
        
        point = mask_to_center_point(gt_mask)
        if point is None:
            return np.zeros(gt_mask.shape, dtype=np.uint8), extra
        
        point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # 1 = foreground
        
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        
        pred = (masks[0] > 0).astype(np.uint8)
        
        if scores is not None and len(scores) > 0:
            extra["pred_iou_score"] = float(scores[0])
        
        return pred, extra

    def _predict_automatic(self, rgb: np.ndarray, extra: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Automatic mask generation."""
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            try:
                from segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            except ImportError:
                warnings.warn("[WARN] SAM2AutomaticMaskGenerator not available")
                return np.zeros(rgb.shape[:2], dtype=np.uint8), extra
        
        generator = SAM2AutomaticMaskGenerator(
            model=self._sam2,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=0,
        )
        
        anns = generator.generate(rgb)
        
        if len(anns) == 0:
            return np.zeros(rgb.shape[:2], dtype=np.uint8), extra
        
        # Choose mask with largest area (fixed rule for fair comparison)
        best = max(anns, key=lambda a: a.get("area", 0))
        pred = best["segmentation"].astype(np.uint8)
        
        if "predicted_iou" in best:
            extra["pred_iou_score"] = float(best["predicted_iou"])
        if "stability_score" in best:
            extra["stability_score"] = float(best["stability_score"])
        
        return pred, extra
