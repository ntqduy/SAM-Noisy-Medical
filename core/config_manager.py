"""ConfigManager – load, validate, and normalise YAML experiment configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


_REQUIRED_SECTIONS = ("exp", "datasets", "models")


class ConfigManager:
    """Load and validate a YAML experiment config.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.

    Attributes
    ----------
    cfg : dict
        The parsed (and lightly normalised) configuration.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.cfg = self._load(self.path)
        self._validate(self.cfg)
        self._normalise(self.cfg)

    # ── public helpers ───────────────────────────────────────────────────

    @property
    def exp_name(self) -> str:
        return str(self.cfg["exp"].get("name", "experiment"))

    @property
    def exp_dir(self) -> Path:
        out_root = Path(str(self.cfg["exp"].get("out_root", "outputs")))
        d = out_root / self.exp_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def device(self) -> str:
        """Primary device (first in the list when multi-GPU)."""
        return self.devices[0]

    @property
    def devices(self) -> List[str]:
        """All available devices resolved from config."""
        return list(self.cfg.get("_resolved_devices", ["cpu"]))

    @property
    def datasets_cfg(self) -> List[Dict[str, Any]]:
        return self.cfg["datasets"]

    @property
    def models_cfg(self) -> List[Dict[str, Any]]:
        return self.cfg["models"]

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data)}")
        return data

    @staticmethod
    def _validate(cfg: Dict[str, Any]) -> None:
        missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
        if missing:
            raise ValueError(f"Config missing required sections: {missing}")

        if not isinstance(cfg["datasets"], list) or not cfg["datasets"]:
            raise ValueError("'datasets' must be a non-empty list.")
        if not isinstance(cfg["models"], list) or not cfg["models"]:
            raise ValueError("'models' must be a non-empty list.")

        for i, ds in enumerate(cfg["datasets"]):
            if "name" not in ds or "adapter" not in ds:
                raise ValueError(
                    f"Dataset entry [{i}] must have 'name' and 'adapter' keys."
                )

        for i, mdl in enumerate(cfg["models"]):
            if "name" not in mdl:
                raise ValueError(f"Model entry [{i}] must have a 'name' key.")

    @staticmethod
    def _normalise(cfg: Dict[str, Any]) -> None:
        # Ensure 'experiment' alias is merged into 'exp'
        if "experiment" in cfg and "exp" not in cfg:
            cfg["exp"] = cfg.pop("experiment")
        elif "experiment" in cfg:
            cfg["exp"].update(cfg.pop("experiment"))

        # Ensure out_root default
        cfg["exp"].setdefault("out_root", "outputs")
        cfg["exp"].setdefault("name", "experiment")

        # Resolve devices
        cfg["_resolved_devices"] = ConfigManager._resolve_devices(cfg)
        # Keep single `device` key pointing to the primary device
        cfg["device"] = cfg["_resolved_devices"][0]

        # Default noise_config
        cfg.setdefault("noise_config", {})
        cfg["noise_config"].setdefault("base_seed", 42)
        cfg["noise_config"].setdefault("n_noise_seeds", 3)

    # ── device resolution ────────────────────────────────────────────────

    @staticmethod
    def _resolve_devices(cfg: Dict[str, Any]) -> List[str]:
        """Build a list of device strings from the config.

        Supported config keys (checked in order):
          ``devices``  – explicit list, e.g. ["cuda:0", "cuda:1"]
          ``num_gpus``  – int; auto-generates cuda:0 … cuda:N-1
          ``device``   – single string, e.g. "cuda", "cuda:2", "cpu"

        Falls back to ``"cpu"`` when nothing is specified.
        """
        # 1) Explicit device list
        explicit = cfg.get("devices")
        if isinstance(explicit, list) and explicit:
            return [str(d) for d in explicit]

        # 2) num_gpus → auto-generate
        num_gpus = cfg.get("num_gpus")
        if num_gpus is not None:
            n = int(num_gpus)
            if n >= 1:
                avail = ConfigManager._available_gpu_count()
                n = min(n, avail) if avail > 0 else 0
                if n >= 1:
                    return [f"cuda:{i}" for i in range(n)]
            # fall through to single device

        # 3) Single device string
        dev = str(cfg.get("device", "cpu"))
        return [dev]

    @staticmethod
    def _available_gpu_count() -> int:
        """Number of CUDA GPUs visible to this process."""
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            pass
        # Fallback: check CUDA_VISIBLE_DEVICES env var
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if vis.strip():
            return len([x for x in vis.split(",") if x.strip()])
        return 0

    def override_devices(self, devices: List[str]) -> None:
        """Override the resolved device list (e.g. from CLI flags)."""
        self.cfg["_resolved_devices"] = list(devices)
        self.cfg["device"] = devices[0]
