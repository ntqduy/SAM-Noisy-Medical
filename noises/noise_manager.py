"""
NoiseManager – central controller for on-the-fly noise application.

Reads noise configuration from YAML, builds noise instances, and exposes a
single ``apply_noise(image, noise_type, level, seed)`` interface.

Noise is **never** pre-generated into the dataset; it is applied dynamically
per-image during inference.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

from noises.base import NoiseBase, NoiseResult
from noises.noise_registry import get_noise_class


class NoiseManager:
    """
    Central manager for deterministic, on-the-fly noise injection.

    Parameters
    ----------
    protocols : dict
        ``protocols.coupled_presets`` mapping noise_type -> level -> params.
    noise_config : dict
        Contains ``base_seed`` and ``n_noise_seeds``.
    """

    def __init__(
        self,
        protocols: Dict[str, Dict[str, Dict[str, Any]]],
        noise_config: Dict[str, Any],
    ) -> None:
        self.presets: Dict[str, Dict[str, Dict[str, Any]]] = protocols
        self.base_seed: int = int(noise_config.get("base_seed", 42))
        self.n_noise_seeds: int = max(1, int(noise_config.get("n_noise_seeds", 1)))

    # ── public interface ─────────────────────────────────────────────────

    def apply_noise(
        self,
        image: np.ndarray,
        noise_type: str,
        level: str,
        seed: int,
        *,
        dataset_name: str = "",
        image_id: str = "",
    ) -> np.ndarray:
        """
        Apply noise to *image* on-the-fly and return the corrupted image.

        For ``L0`` (clean baseline) the original image is returned unchanged.
        The seed is combined with context identifiers to ensure determinism.
        """
        if level == "L0":
            return image.copy()

        params = self._get_params(noise_type, level)
        if params is None:
            return image.copy()

        p = float(params.pop("p", 1.0))
        if p <= 0:
            return image.copy()

        effective_seed = self._compute_seed(
            seed, dataset_name, image_id, noise_type, level,
        )
        noise_cls = get_noise_class(noise_type)
        if noise_cls is None:
            return image.copy()

        noise_instance: NoiseBase = noise_cls(
            p=p,
            params=params,
            seed=effective_seed,
            level=level,
            protocol="coupled_presets",
            compute_distortion=False,
        )
        result: NoiseResult = noise_instance(image)
        return result.noisy_image

    def get_params(self, noise_type: str, level: str) -> Optional[Dict[str, Any]]:
        """Return a **copy** of the preset parameters (including ``p``)."""
        return self._get_params(noise_type, level)

    # ── internals ────────────────────────────────────────────────────────

    def _get_params(self, noise_type: str, level: str) -> Optional[Dict[str, Any]]:
        preset_levels = self.presets.get(noise_type)
        if not preset_levels or level not in preset_levels:
            return None
        return dict(preset_levels[level])

    def _compute_seed(
        self,
        noise_seed: int,
        dataset: str,
        image_id: str,
        noise_type: str,
        level: str,
    ) -> int:
        key = f"{self.base_seed}|{noise_seed}|{dataset}|{image_id}|{noise_type}|{level}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)
