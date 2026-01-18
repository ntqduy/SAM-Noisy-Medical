import numpy as np
from typing import Optional, Tuple, List


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract bounding box from binary mask.
    
    Args:
        mask: Binary mask HxW
        
    Returns:
        bbox as [x0, y0, x1, y1] or None if mask is empty
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def mask_to_center_point(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Get centroid point from binary mask.
    
    Args:
        mask: Binary mask HxW
        
    Returns:
        (x, y) centroid or None if mask is empty
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy)


def mask_to_random_points(mask: np.ndarray, n_points: int = 5, seed: int = 42) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample random points from mask for point prompts.
    
    Args:
        mask: Binary mask HxW
        n_points: Number of points to sample
        seed: Random seed
        
    Returns:
        (point_coords, point_labels) or None if mask is empty
        point_coords: Nx2 array of (x, y) coordinates
        point_labels: N array of labels (1=foreground)
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    
    rng = np.random.default_rng(seed)
    n = min(n_points, len(xs))
    indices = rng.choice(len(xs), size=n, replace=False)
    
    point_coords = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
    point_labels = np.ones(n, dtype=np.int32)
    
    return point_coords, point_labels


def bbox_with_margin(bbox: np.ndarray, margin: float, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Add margin to bounding box.
    
    Args:
        bbox: [x0, y0, x1, y1]
        margin: Fraction to expand (e.g., 0.1 = 10%)
        img_shape: (H, W) to clip bounds
        
    Returns:
        Expanded bbox clipped to image bounds
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    dx = w * margin
    dy = h * margin
    
    H, W = img_shape
    x0 = max(0, x0 - dx)
    y0 = max(0, y0 - dy)
    x1 = min(W - 1, x1 + dx)
    y1 = min(H - 1, y1 + dy)
    
    return np.array([x0, y0, x1, y1], dtype=np.float32)
