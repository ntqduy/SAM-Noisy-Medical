import numpy as np


def overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.35, color=(255, 0, 0)) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    rgb = np.stack([g, g, g], axis=-1)
    m = mask.astype(bool)
    c = np.array(color, dtype=np.float32) / 255.0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * c
    return (rgb * 255.0).astype(np.uint8)
