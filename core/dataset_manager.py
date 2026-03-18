"""DatasetManager – build and cache dataset adapters from config."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets.dataset_registry import build_dataset
from datasets.base_dataset import DatasetAdapter


class DatasetManager:
    """Centralised dataset construction and caching.

    Builds :class:`DatasetAdapter` instances from the ``datasets`` section of
    the experiment config.  Adapters are cached so repeated calls for the same
    dataset name return the same object.

    Parameters
    ----------
    datasets_cfg : list[dict]
        The ``cfg["datasets"]`` list from the YAML config.
    """

    def __init__(self, datasets_cfg: List[Dict[str, Any]]) -> None:
        self._cfg_map: Dict[str, Dict[str, Any]] = {
            str(d["name"]): d for d in datasets_cfg
        }
        self._cache: Dict[str, DatasetAdapter] = {}

    # ── public API ───────────────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        """Return all dataset names defined in the config."""
        return list(self._cfg_map.keys())

    def get(self, name: str) -> DatasetAdapter:
        """Return a dataset adapter by name, building it on first access."""
        if name not in self._cfg_map:
            raise KeyError(
                f"Unknown dataset '{name}'. Available: {self.names}"
            )
        if name not in self._cache:
            self._cache[name] = build_dataset(self._cfg_map[name])
        return self._cache[name]

    def iter(
        self, *, filter_names: Optional[List[str]] = None,
    ):
        """Yield ``(name, adapter)`` pairs, optionally filtered."""
        for name in self.names:
            if filter_names and name not in filter_names:
                continue
            yield name, self.get(name)
