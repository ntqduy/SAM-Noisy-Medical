"""
MedSAM (Medical Segment Anything Model) runner.
Specialized SAM fine-tuned for medical imaging.
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


class MedSAMRunner(BaseModelRunner):
    """
    MedSAM runner with support for:
      - prompt_bbox: bounding box from GT (primary mode)
      - prompt_points: points from GT
      - automatic: automatic mask generation
    
    MedSAM is optimized for medical images with bbox prompts.
    """
    
    def __init__(self, weight_cfg: dict, mode: str, device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.ckpt = weight_cfg.get("checkpoint", "")
        self.model_type = weight_cfg.get("model_type", "vit_b")
        self.weight_id = weight_cfg.get("id", "medsam")
        self._predictor = None
        self._model = None
        self._init()

    def _init(self):
        """Initialize MedSAM model."""
        try:
            # Try MedSAM-specific import
            from segment_anything import sam_model_registry, SamPredictor
            
            if not self.ckpt or not Path(self.ckpt).exists():
                raise FileNotFoundError(f"MedSAM checkpoint not found: {self.ckpt}")
            
            # MedSAM uses same architecture as SAM but different weights
            self._model = sam_model_registry[self.model_type](checkpoint=self.ckpt)
            self._model.to(self.device)
            self._predictor = SamPredictor(self._model)
            
        except ImportError as e:
            warnings.warn(
                f"[WARN] segment-anything not installed. Install:\n"
                f"  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
                f"Error: {e}"
            )
            self._predictor = None
        except Exception as e:
            warnings.warn(f"[WARN] Failed to load MedSAM: {e}")
            self._predictor = None

    def predict(self, image_gray: np.ndarray, gt_mask: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run MedSAM prediction.
        
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
            warnings.warn("[WARN] MedSAM predictor not initialized, returning empty mask")
            return np.zeros(image_gray.shape, dtype=np.uint8), extra
        
        rgb = to_rgb_repeat(image_gray)
        
        # MedSAM specific preprocessing: resize to 1024x1024 for optimal performance
        rgb_resized, scale_factors = self._preprocess_image(rgb)
        self._predictor.set_image(rgb_resized)
        
        if self.mode == "prompt_bbox":
            pred, extra = self._predict_bbox(gt_mask, scale_factors, extra)
        elif self.mode == "prompt_points":
            pred, extra = self._predict_points(gt_mask, scale_factors, extra)
        elif self.mode == "automatic":
            pred, extra = self._predict_automatic(rgb_resized, extra)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Resize prediction back to original size
        if pred.shape != image_gray.shape:
            import cv2
            pred = cv2.resize(pred.astype(np.uint8), (image_gray.shape[1], image_gray.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        return pred, extra

    def _preprocess_image(self, rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Preprocess image for MedSAM (resize to 1024x1024)."""
        import cv2
        
        target_size = 1024
        h, w = rgb.shape[:2]
        
        # Scale to fit in target_size while preserving aspect ratio
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target_size x target_size
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        scale_factors = (scale, scale)
        return padded, scale_factors

    def _predict_bbox(self, gt_mask: np.ndarray, scale_factors: Tuple[float, float], extra: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict using bounding box prompt."""
        box = mask_to_bbox(gt_mask)
        if box is None:
            return np.zeros(gt_mask.shape, dtype=np.uint8), extra
        
        # Scale box coordinates
        sx, sy = scale_factors
        scaled_box = np.array([box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy], dtype=np.float32)
        
        masks, scores, logits = self._predictor.predict(
            box=scaled_box[None, :],
            multimask_output=False
        )
        
        pred = (masks[0] > 0).astype(np.uint8)
        
        if scores is not None and len(scores) > 0:
            extra["pred_iou_score"] = float(scores[0])
        
        return pred, extra

    def _predict_points(self, gt_mask: np.ndarray, scale_factors: Tuple[float, float], extra: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict using point prompts."""
        from model.prompts import mask_to_center_point
        
        point = mask_to_center_point(gt_mask)
        if point is None:
            return np.zeros(gt_mask.shape, dtype=np.uint8), extra
        
        sx, sy = scale_factors
        point_coords = np.array([[point[0]*sx, point[1]*sy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        
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
        """Automatic mask generation using SamAutomaticMaskGenerator."""
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            warnings.warn("[WARN] SamAutomaticMaskGenerator not available")
            return np.zeros(rgb.shape[:2], dtype=np.uint8), extra
        
        generator = SamAutomaticMaskGenerator(
            model=self._model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=0,
        )
        
        anns = generator.generate(rgb)
        
        if len(anns) == 0:
            return np.zeros(rgb.shape[:2], dtype=np.uint8), extra
        
        best = max(anns, key=lambda a: a.get("area", 0))
        pred = best["segmentation"].astype(np.uint8)
        
        if "predicted_iou" in best:
            extra["pred_iou_score"] = float(best["predicted_iou"])
        
        return pred, extra
