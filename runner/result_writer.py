"""
ExperimentResultWriter – handles CSV output for per-image experiment results.

This is an alternative to having the engine write CSV directly.
It can be used to replay / transform raw results.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


RAW_COLUMNS = [
    "dataset",
    "model",
    "prompt_mode",
    "noise_type",
    "noise_level",
    "noise_seed",
    "image_id",
    "IoU",
    "Dice",
    "Recall",
    "Precision",
    "F1",
    "HD",
]


def _slugify(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower())
    return out.strip("_") or "item"


def _prompt_suffix(prompt_mode: str) -> str:
    mapping = {
        "prompt_point": "point",
        "prompt_bbox": "bbox",
        "prompt_point_box": "pointbox",
        "autogen": "autogen",
    }
    return mapping.get(prompt_mode, prompt_mode)


class ExperimentResultWriter:
    """
    Writes per-image metric rows into CSV files following the naming convention::

        outputs/{experiment}/{dataset}/{model}/{runner}_{dataset}_{prompt}_raw.csv

    Parameters
    ----------
    exp_dir : Path
        Root experiment directory (e.g. ``outputs/phase1_tn3k_sam``).
    """

    def __init__(self, exp_dir: Path) -> None:
        self.exp_dir = Path(exp_dir)
        self._handles: Dict[str, Any] = {}  # key → (file_handle, csv_writer)

    # ── public API ───────────────────────────────────────────────────────

    def open(
        self,
        dataset_name: str,
        model_name: str,
        runner_name: str,
        prompt_mode: str,
    ) -> Path:
        """
        Open (and create) the raw CSV for a given combination.
        Returns the path to the CSV file.
        """
        key = self._key(dataset_name, model_name, runner_name, prompt_mode)
        if key in self._handles:
            _, _, path = self._handles[key]
            return path

        csv_dir = self.exp_dir / _slugify(dataset_name) / _slugify(model_name)
        csv_dir.mkdir(parents=True, exist_ok=True)
        raw_name = (
            f"{_slugify(runner_name)}_{_slugify(dataset_name)}_"
            f"{_prompt_suffix(prompt_mode)}_raw.csv"
        )
        path = csv_dir / raw_name
        fh = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        self._handles[key] = (fh, writer, path)
        return path

    def write_row(
        self,
        dataset_name: str,
        model_name: str,
        runner_name: str,
        prompt_mode: str,
        row: Dict[str, Any],
    ) -> None:
        """Append a single metric row to the appropriate CSV."""
        key = self._key(dataset_name, model_name, runner_name, prompt_mode)
        if key not in self._handles:
            self.open(dataset_name, model_name, runner_name, prompt_mode)
        _, writer, _ = self._handles[key]
        writer.writerow(row)

    def close_all(self) -> None:
        """Flush and close all open file handles."""
        for fh, _, _ in self._handles.values():
            fh.close()
        self._handles.clear()

    def get_csv_path(
        self,
        dataset_name: str,
        model_name: str,
        runner_name: str,
        prompt_mode: str,
    ) -> Path:
        key = self._key(dataset_name, model_name, runner_name, prompt_mode)
        if key in self._handles:
            return self._handles[key][2]
        csv_dir = self.exp_dir / _slugify(dataset_name) / _slugify(model_name)
        raw_name = (
            f"{_slugify(runner_name)}_{_slugify(dataset_name)}_"
            f"{_prompt_suffix(prompt_mode)}_raw.csv"
        )
        return csv_dir / raw_name

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close_all()

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _key(dataset: str, model: str, runner: str, prompt: str) -> str:
        return f"{dataset}|{model}|{runner}|{prompt}"
