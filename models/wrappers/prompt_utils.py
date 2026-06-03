"""
Prompt helper utilities – build standard prompt payloads from ground-truth masks.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def normalize_prompt_mode(prompt_mode: str) -> str:
    pm = str(prompt_mode or "").strip().lower()
    mapping = {
        "bbox": "prompt_bbox", "box": "prompt_bbox", "prompt_bbox": "prompt_bbox",
        "point": "prompt_point", "prompt_point": "prompt_point",
        "multi_point": "prompt_multi_point", "prompt_multi_point": "prompt_multi_point",
        "point_box": "prompt_point_box", "pointbox": "prompt_point_box",
        "point+bbox": "prompt_point_box",  # Visualization display alias for point+box
        "prompt_point_box": "prompt_point_box",
        "autogen": "autogen", "auto": "autogen", "automatic": "autogen",
    }
    if pm in mapping:
        return mapping[pm]
    raise ValueError(
        f"Unsupported prompt mode '{prompt_mode}'. "
        "Expected: prompt_point, prompt_multi_point, prompt_bbox, prompt_point_box, autogen."
    )


def _foreground_coords(mask: np.ndarray) -> np.ndarray:
    """Return foreground coordinates as Nx2 array in (x, y) order."""
    m = np.asarray(mask).astype(bool)
    ys, xs = np.where(m)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def mask_to_bbox(
    mask: np.ndarray,
    margin_ratio: float = 0.05,
    min_margin_px: int = 2,
    max_margin_ratio: float = 0.12,
    fg_coords: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Build an XYXY bounding box from foreground pixels with adaptive margin.

    Margin is proportional to object size and clipped by image-size-dependent caps.
    """
    m = np.asarray(mask).astype(bool)
    h, w = m.shape[:2]
    coords = _foreground_coords(m) if fg_coords is None else fg_coords
    if coords.shape[0] == 0:
        return None

    xs = coords[:, 0]
    ys = coords[:, 1]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    obj_w = x1 - x0 + 1
    obj_h = y1 - y0 + 1
    max_obj_dim = max(obj_w, obj_h)

    mx = max(min_margin_px, int(round(max_obj_dim * margin_ratio)))
    my = max(min_margin_px, int(round(max_obj_dim * margin_ratio)))
    mx = min(mx, int(round(w * max_margin_ratio)))
    my = min(my, int(round(h * max_margin_ratio)))

    return (max(0, x0 - mx), max(0, y0 - my), min(w - 1, x1 + mx), min(h - 1, y1 + my))


def mask_centroid(mask: np.ndarray, fg_coords: Optional[np.ndarray] = None) -> Optional[Tuple[int, int]]:
    m = np.asarray(mask).astype(bool)
    coords = _foreground_coords(m) if fg_coords is None else fg_coords
    if coords.shape[0] == 0:
        return None
    xs = coords[:, 0]
    ys = coords[:, 1]
    return (int(np.round(xs.mean())), int(np.round(ys.mean())))


def _farthest_point_sampling(coords: np.ndarray, k: int) -> np.ndarray:
    """Deterministic farthest-point sampling on Nx2 integer coordinates."""
    n = coords.shape[0]
    if n == 0 or k <= 0:
        return np.empty((0, 2), dtype=np.int32)
    if n <= k:
        return coords.astype(np.int32)

    selected = np.empty((k, 2), dtype=np.int32)
    # Start from point nearest centroid for stable single-point behavior.
    center = coords.astype(np.float32).mean(axis=0, keepdims=True)
    seed_idx = int(np.argmin(np.sum((coords.astype(np.float32) - center) ** 2, axis=1)))
    selected[0] = coords[seed_idx]

    d2 = np.sum((coords - selected[0]) ** 2, axis=1)
    for i in range(1, k):
        idx = int(np.argmax(d2))
        selected[i] = coords[idx]
        d2 = np.minimum(d2, np.sum((coords - selected[i]) ** 2, axis=1))
    return selected


def mask_to_points(
    mask: np.ndarray,
    k: int = 3,
    fg_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample K valid foreground points from a binary mask.

    Returns
    -------
    np.ndarray
        Shape (N, 2), where N <= K and points are in (x, y) pixel coordinates.
    """
    m = np.asarray(mask).astype(bool)
    coords = _foreground_coords(m) if fg_coords is None else fg_coords
    if coords.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    k = int(max(1, k))
    if k == 1:
        center = coords.astype(np.float32).mean(axis=0, keepdims=True)
        idx = int(np.argmin(np.sum((coords.astype(np.float32) - center) ** 2, axis=1)))
        return coords[idx : idx + 1].astype(np.int32)
    return _farthest_point_sampling(coords, k)


def _component_representative_points(
    mask: np.ndarray,
    max_points: Optional[int] = None,
) -> np.ndarray:
    """
    Return one deterministic foreground point per connected component.

    The point for each component is chosen as the foreground pixel nearest to the
    component centroid, ensuring:
    - 1 object -> 1 point
    - 2 disconnected objects -> 2 points
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim > 2:
        m = np.squeeze(m)
    if not m.any():
        return np.empty((0, 2), dtype=np.int32)

    try:
        import cv2

        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    except Exception:
        # Fallback: keep previous behavior if cv2 is unavailable.
        fg = _foreground_coords(m)
        return mask_to_points(m, k=1, fg_coords=fg)

    pts = []
    for lab in range(1, int(n_labels)):
        ys, xs = np.where(labels == lab)
        if xs.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        d2 = (xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2
        idx = int(np.argmin(d2))
        pts.append((int(xs[idx]), int(ys[idx])))

    if not pts:
        return np.empty((0, 2), dtype=np.int32)

    out = np.asarray(pts, dtype=np.int32)
    if max_points is not None:
        k = int(max(1, max_points))
        out = out[:k]
    return out


def build_prompt(
    gt_mask: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    k_points: int = 3,
    single_point: bool = True,
) -> Dict[str, Any]:
    """
    Build a prompt payload dict.

    Backward-compatible keys:
    - point: tuple[int, int]
    - bbox: tuple[int, int, int, int]

    New keys:
    - points: np.ndarray (N, 2)
    - point_labels: np.ndarray (N,)
    """
    h, w = image_shape
    if gt_mask is not None and np.asarray(gt_mask).astype(bool).any():
        coords = _foreground_coords(gt_mask)
        n_points = 1 if single_point else int(max(1, k_points))
        pts = mask_to_points(gt_mask, k=n_points, fg_coords=coords)
        pt = tuple(int(v) for v in pts[0])
        return {
            "point": pt,
            "points": pts,
            "point_labels": np.ones((pts.shape[0],), dtype=np.int32),
            "bbox": mask_to_bbox(gt_mask, fg_coords=coords),
            "gt_mask": gt_mask,
        }

    center = (w // 2, h // 2)
    return {
        "point": center,
        "points": np.asarray([center], dtype=np.int32),
        "point_labels": np.ones((1,), dtype=np.int32),
        "bbox": (w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        "gt_mask": gt_mask,
    }


def _clip_point(point: Optional[Tuple[int, int]], image_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if point is None:
        return None
    h, w = image_shape
    x, y = int(point[0]), int(point[1])
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    return (x, y)


def _clip_bbox(
    bbox: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    if bbox is None:
        return None
    h, w = image_shape
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    return (x0, y0, x1, y1)


def _tight_bbox(
    mask: np.ndarray,
    fg_coords: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int, int, int]]:
    coords = _foreground_coords(mask) if fg_coords is None else fg_coords
    if coords.shape[0] == 0:
        return None
    xs = coords[:, 0]
    ys = coords[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _bbox_center(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
    if bbox is None:
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return (int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0)))


def _as_range_value(value: Any, rng: np.random.Generator) -> float:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        lo = float(value[0])
        hi = float(value[1])
        return float(rng.uniform(lo, hi))
    return float(value)


def deterministic_variant_seed(parts: Sequence[Any]) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _bbox_from_variant(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    variant: Optional[Dict[str, Any]],
    *,
    deterministic_parts: Sequence[Any] = (),
) -> Optional[Tuple[int, int, int, int]]:
    if variant is None:
        return None

    m = np.asarray(mask).astype(bool)
    coords = _foreground_coords(m)
    if coords.shape[0] == 0:
        return None

    vtype = str(variant.get("type", "baseline")).strip().lower()
    if vtype == "baseline":
        expand = float(max(0.0, variant.get("expand", 0.05)))
        min_margin_px = int(max(0, variant.get("min_margin_px", 2)))
        max_margin_ratio = float(max(expand, variant.get("max_margin_ratio", 0.12)))
        return _clip_bbox(
            mask_to_bbox(
                m,
                margin_ratio=expand,
                min_margin_px=min_margin_px,
                max_margin_ratio=max_margin_ratio,
                fg_coords=coords,
            ),
            image_shape,
        )

    tight = _tight_bbox(m, fg_coords=coords)
    if tight is None:
        return None

    h, w = image_shape
    x0, y0, x1, y1 = [int(v) for v in tight]
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)

    if vtype == "expand":
        expand = float(max(0.0, variant.get("expand", 0.0)))
        dx = int(round(bw * expand))
        dy = int(round(bh * expand))
        return _clip_bbox((x0 - dx, y0 - dy, x1 + dx, y1 + dy), image_shape)

    base_expand = float(max(0.0, variant.get("base_expand", 0.05)))
    base_bbox = mask_to_bbox(m, margin_ratio=base_expand, fg_coords=coords)
    if base_bbox is None:
        base_bbox = tight
    x0, y0, x1, y1 = [int(v) for v in base_bbox]
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)

    if vtype == "translate":
        seed = deterministic_variant_seed(deterministic_parts)
        rng = np.random.default_rng(seed)
        sx = _as_range_value(variant.get("shift_x", 0.0), rng)
        sy = _as_range_value(variant.get("shift_y", 0.0), rng)
        if str(variant.get("relative_to", "bbox_size")).lower() == "bbox_size":
            dx = int(round(sx * bw))
            dy = int(round(sy * bh))
        else:
            dx = int(round(sx))
            dy = int(round(sy))
        return _clip_bbox((x0 + dx, y0 + dy, x1 + dx, y1 + dy), image_shape)

    if vtype == "shrink":
        shrink = float(max(0.0, variant.get("shrink", 0.0)))
        min_size = int(max(1, variant.get("min_size", 1)))
        dx = int(round(bw * shrink))
        dy = int(round(bh * shrink))
        nx0, ny0, nx1, ny1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
        if nx1 - nx0 + 1 < min_size:
            cx = (x0 + x1) / 2.0
            half = (min_size - 1) / 2.0
            nx0 = int(round(cx - half))
            nx1 = int(round(cx + half))
        if ny1 - ny0 + 1 < min_size:
            cy = (y0 + y1) / 2.0
            half = (min_size - 1) / 2.0
            ny0 = int(round(cy - half))
            ny1 = int(round(cy + half))
        if bool(variant.get("clip_to_image", True)):
            return _clip_bbox((nx0, ny0, nx1, ny1), image_shape)
        return (max(0, nx0), max(0, ny0), min(w - 1, nx1), min(h - 1, ny1))

    return _clip_bbox(base_bbox, image_shape)


def _point_from_variant(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    variant: Optional[Dict[str, Any]],
    *,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    deterministic_parts: Sequence[Any] = (),
) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
    if variant is None:
        return None, None

    m = np.asarray(mask).astype(bool)
    vtype = str(variant.get("type", "mask_centroid")).strip().lower()

    if vtype == "bbox_center":
        base_bbox = bbox
        if base_bbox is None:
            base_bbox = mask_to_bbox(m, margin_ratio=0.05)
        pt = _clip_point(_bbox_center(base_bbox), image_shape)
        if pt is None:
            return None, None
        return pt, np.asarray([pt], dtype=np.int32)

    if vtype == "mask_centroid":
        pt = _clip_point(mask_centroid(m), image_shape)
        if pt is None:
            return None, None
        return pt, np.asarray([pt], dtype=np.int32)

    if vtype == "random_inside_mask":
        coords = _foreground_coords(m)
        if coords.shape[0] == 0:
            return None, None
        n_random = int(max(1, variant.get("n_random", 1)))
        seed = deterministic_variant_seed(deterministic_parts)
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], size=min(n_random, coords.shape[0]), replace=False)
        pts = coords[np.asarray(idx)].astype(np.int32).reshape(-1, 2)
        pt = (int(pts[0, 0]), int(pts[0, 1]))
        return pt, pts

    return None, None


def bbox_center_inside_mask(
    mask: Optional[np.ndarray],
    bbox: Optional[Tuple[int, int, int, int]],
) -> float:
    if mask is None or bbox is None:
        return float("nan")
    m = np.asarray(mask).astype(bool)
    if m.ndim > 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        return float("nan")
    center = _bbox_center(bbox)
    if center is None:
        return float("nan")
    x, y = center
    if y < 0 or y >= m.shape[0] or x < 0 or x >= m.shape[1]:
        return float("nan")
    return 1.0 if bool(m[y, x]) else 0.0


def resolve_prompt(
    prompt: Optional[Dict[str, Any]],
    image_shape: Tuple[int, int],
    prompt_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve a complete prompt payload with normalized ``point`` and ``bbox``.

    Priority:
    1) Explicit ``prompt['point']`` / ``prompt['bbox']`` if provided.
    2) Values derived from ``prompt['gt_mask']``.
    3) Deterministic center fallback.
    """
    if prompt is None:
        prompt = {}

    h, w = image_shape
    gt_mask = prompt.get("gt_mask")
    mode = normalize_prompt_mode(prompt_mode) if prompt_mode is not None else None

    # Unified benchmark setup:
    # - prompt_point: default one positive click (can be overridden via n_fg_points)
    # - prompt_bbox: GT box with adaptive margin
    # - prompt_point_box: exactly one positive click + one GT box
    single_point_default = mode in ("prompt_point", "prompt_point_box")
    single_point = bool(prompt.get("single_point", single_point_default))
    requested_points = int(max(1, prompt.get("n_fg_points", 1 if single_point else 3)))
    auto_points_by_components = bool(prompt.get("auto_points_by_components", True))
    max_auto_fg_points = int(max(1, prompt.get("max_auto_fg_points", 2)))
    bbox_margin_ratio = float(max(0.0, prompt.get("bbox_margin_ratio", 0.05)))
    bbox_min_margin_px = int(max(0, prompt.get("bbox_min_margin_px", 2)))
    bbox_max_margin_ratio = float(
        max(bbox_margin_ratio, prompt.get("bbox_max_margin_ratio", 0.12))
    )

    user_point = prompt.get("point")
    user_bbox = prompt.get("bbox")
    user_points = prompt.get("points", prompt.get("point_coords"))
    user_labels = prompt.get("point_labels")

    point = None
    if user_point is not None:
        point = _clip_point((user_point[0], user_point[1]), image_shape)

    bbox = None
    if user_bbox is not None:
        bbox = _clip_bbox((user_bbox[0], user_bbox[1], user_bbox[2], user_bbox[3]), image_shape)

    points_arr: Optional[np.ndarray] = None
    labels_arr: Optional[np.ndarray] = None

    if user_points is not None:
        raw_points = np.asarray(user_points, dtype=np.int32).reshape(-1, 2)
        clipped = [_clip_point((int(x), int(y)), image_shape) for x, y in raw_points]
        valid = [p for p in clipped if p is not None]
        if valid:
            points_arr = np.asarray(valid, dtype=np.int32)
            if user_labels is not None:
                lbl = np.asarray(user_labels, dtype=np.int32).reshape(-1)
                if lbl.shape[0] >= points_arr.shape[0]:
                    labels_arr = lbl[: points_arr.shape[0]]
            if labels_arr is None:
                labels_arr = np.ones((points_arr.shape[0],), dtype=np.int32)

    if points_arr is None and point is not None:
        points_arr = np.asarray([point], dtype=np.int32)
        labels_arr = np.ones((1,), dtype=np.int32)

    if gt_mask is not None and np.asarray(gt_mask).astype(bool).any():
        fg_coords = _foreground_coords(gt_mask)
        if fg_coords.shape[0] > 0 and points_arr is None:
            if mode == "prompt_point" and auto_points_by_components:
                # Fair point policy for benchmark: 1 connected object -> 1 point,
                # 2 connected objects -> 2 points (capped by max_auto_fg_points).
                fg_points = _component_representative_points(
                    gt_mask, max_points=max_auto_fg_points
                )
                if fg_points.shape[0] > 0:
                    requested_points = int(fg_points.shape[0])
                    single_point = requested_points <= 1
            else:
                # For disconnected objects (e.g., left/right lungs), arithmetic centroid
                # can lie on background. Use guaranteed foreground points instead.
                n_points = 1 if single_point else requested_points
                fg_points = mask_to_points(gt_mask, k=n_points, fg_coords=fg_coords)
            if fg_points.shape[0] > 0:
                points_arr = fg_points.astype(np.int32)
                labels_arr = np.ones((points_arr.shape[0],), dtype=np.int32)
                point = (int(points_arr[0, 0]), int(points_arr[0, 1]))
        if bbox is None:
            bbox = _clip_bbox(
                mask_to_bbox(
                    gt_mask,
                    margin_ratio=bbox_margin_ratio,
                    min_margin_px=bbox_min_margin_px,
                    max_margin_ratio=bbox_max_margin_ratio,
                    fg_coords=fg_coords,
                ),
                image_shape,
            )

    if point is None:
        point = (w // 2, h // 2)
    if points_arr is None:
        points_arr = np.asarray([point], dtype=np.int32)
        labels_arr = np.ones((1,), dtype=np.int32)
    if labels_arr is None:
        labels_arr = np.ones((points_arr.shape[0],), dtype=np.int32)
    if bbox is None:
        bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    # Enforce strict prompt-mode separation to avoid leakage across modes.
    if mode == "prompt_point":
        # Point-only: default one click, unless caller explicitly requests multi-point.
        bbox = None
        if requested_points <= 1 and points_arr is not None and points_arr.shape[0] > 0:
            points_arr = points_arr[:1]
            labels_arr = np.ones((1,), dtype=np.int32)
            single_point = True
        else:
            single_point = False
    elif mode == "prompt_multi_point":
        bbox = None
        single_point = False
    elif mode == "prompt_bbox":
        points_arr = None
        labels_arr = None
        point = None
        single_point = False
    elif mode == "prompt_point_box":
        # Keep exactly one centroid point + one GT box.
        if points_arr is not None and points_arr.shape[0] > 0:
            points_arr = points_arr[:1]
            labels_arr = np.ones((1,), dtype=np.int32)
        single_point = True

    # Keep backward compatibility: "point" mirrors first sampled foreground point when present.
    point_for_logging = None
    if points_arr is not None and points_arr.shape[0] > 0:
        point_for_logging = (int(points_arr[0, 0]), int(points_arr[0, 1]))

    return {
        "point": point_for_logging,
        "points": points_arr,
        "point_labels": labels_arr,
        "single_point": single_point,
        "bbox": bbox,
        "gt_mask": gt_mask,
    }


def resolve_prompt_variant(
    prompt: Optional[Dict[str, Any]],
    image_shape: Tuple[int, int],
    prompt_mode: str,
    *,
    prompt_variant: Optional[Dict[str, Any]] = None,
    bbox_variants: Optional[Dict[str, Dict[str, Any]]] = None,
    point_variants: Optional[Dict[str, Dict[str, Any]]] = None,
    deterministic_parts: Sequence[Any] = (),
) -> Dict[str, Any]:
    """
    Resolve a prompt using an optional benchmark prompt variant.

    When ``prompt_variant`` is ``None``, this is equivalent to
    ``resolve_prompt`` plus metadata fields that label the row as ``default``.
    """
    payload = dict(prompt or {})
    gt_mask = payload.get("gt_mask")
    mode = normalize_prompt_mode(prompt_mode)
    variant_name = "default"
    bbox_variant_name = "default"
    point_variant_name = "default"
    prompt_source = "legacy_default"

    if prompt_variant:
        variant_name = str(prompt_variant.get("name", "variant"))
        prompt_source = "prompt_variants_config"

        bbox_cfg: Optional[Dict[str, Any]] = None
        point_cfg: Optional[Dict[str, Any]] = None

        if mode == "prompt_bbox":
            bbox_cfg = prompt_variant
            bbox_variant_name = variant_name
            point_variant_name = "none"
        elif mode == "prompt_point":
            point_cfg = prompt_variant
            point_variant_name = variant_name
            bbox_variant_name = "none"
        elif mode == "prompt_point_box":
            bbox_name = str(prompt_variant.get("bbox_variant", ""))
            point_name = str(prompt_variant.get("point_variant", ""))
            bbox_variant_name = bbox_name or "none"
            point_variant_name = point_name or "none"
            if bbox_variants and bbox_name in bbox_variants:
                bbox_cfg = bbox_variants[bbox_name]
            if point_variants and point_name in point_variants:
                point_cfg = point_variants[point_name]

        bbox = None
        if gt_mask is not None and bbox_cfg is not None:
            bbox = _bbox_from_variant(
                gt_mask,
                image_shape,
                bbox_cfg,
                deterministic_parts=[*deterministic_parts, variant_name, bbox_variant_name],
            )
            if bbox is not None:
                payload["bbox"] = bbox

        if gt_mask is not None and point_cfg is not None:
            point, points = _point_from_variant(
                gt_mask,
                image_shape,
                point_cfg,
                bbox=bbox,
                deterministic_parts=[*deterministic_parts, variant_name, point_variant_name],
            )
            if point is not None:
                payload["point"] = point
            if points is not None and points.shape[0] > 0:
                payload["points"] = points
                payload["point_labels"] = np.ones((points.shape[0],), dtype=np.int32)
                payload["single_point"] = points.shape[0] <= 1

    resolved = resolve_prompt(payload, image_shape, prompt_mode=mode)
    has_bbox = resolved.get("bbox") is not None
    points = resolved.get("points")
    has_point = resolved.get("point") is not None or (
        points is not None and np.asarray(points).reshape(-1, 2).shape[0] > 0
    )
    resolved.update({
        "prompt_variant": variant_name,
        "bbox_variant": bbox_variant_name,
        "point_variant": point_variant_name,
        "prompt_source": prompt_source,
        "prompt_has_bbox": 1 if has_bbox else 0,
        "prompt_has_point": 1 if has_point else 0,
        "is_bbox_center_inside_mask": bbox_center_inside_mask(
            gt_mask,
            resolved.get("bbox"),
        ),
    })
    return resolved


def prompt_bbox_stats(prompt: Dict[str, Any]) -> Dict[str, int]:
    """Return bbox dimensions for logging/debugging."""
    bbox = prompt.get("bbox")
    if bbox is None:
        return {"bbox_w": 0, "bbox_h": 0, "bbox_area": 0}
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(0, x1 - x0 + 1)
    bh = max(0, y1 - y0 + 1)
    return {"bbox_w": bw, "bbox_h": bh, "bbox_area": bw * bh}


def build_sam_prompt_kwargs(
    prompt_mode: str,
    prompt: Dict[str, Any],
    *,
    batched_box: bool = False,
) -> Dict[str, Any]:
    """Build prompt kwargs for SAM/SAM2-compatible predictors."""
    # Always request multiple masks; caller will pick the best.
    kwargs: Dict[str, Any] = {"multimask_output": True}

    bbox = prompt.get("bbox")
    if prompt_mode in ("prompt_bbox", "prompt_point_box") and bbox is not None:
        box_arr = np.asarray(bbox, dtype=np.float32)
        kwargs["box"] = box_arr[None, :] if batched_box else box_arr

    points = prompt.get("points")
    point_labels = prompt.get("point_labels")
    if points is None and prompt.get("point") is not None:
        pt = prompt["point"]
        points = np.asarray([[pt[0], pt[1]]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int32)

    if prompt_mode in ("prompt_point", "prompt_multi_point", "prompt_point_box") and points is not None:
        point_coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if point_labels is None:
            labels = np.ones((point_coords.shape[0],), dtype=np.int32)
        else:
            labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
            if labels.shape[0] < point_coords.shape[0]:
                pad = np.ones((point_coords.shape[0] - labels.shape[0],), dtype=np.int32)
                labels = np.concatenate([labels, pad], axis=0)
            labels = labels[: point_coords.shape[0]]

        if prompt_mode in ("prompt_point", "prompt_point_box") and bool(prompt.get("single_point", False)):
            point_coords = point_coords[:1]
            labels = labels[:1]

        kwargs["point_coords"] = point_coords
        kwargs["point_labels"] = labels

    return kwargs


def select_best_mask(
    masks: Any,
    iou_predictions: Optional[Any] = None,
) -> np.ndarray:
    """Return a binary mask, selecting the best candidate by IoU score when available."""
    arr = np.asarray(masks)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return (arr > 0).astype(np.uint8)

    if arr.ndim != 3:
        flat = np.asarray(arr).reshape(-1)
        return (flat > 0).astype(np.uint8)

    idx = 0
    if iou_predictions is not None:
        scores = np.asarray(iou_predictions)
        scores = np.squeeze(scores)
        if scores.ndim == 1 and scores.shape[0] == arr.shape[0]:
            if np.isfinite(scores).any():
                idx = int(np.nanargmax(scores))

    return (arr[idx] > 0).astype(np.uint8)
