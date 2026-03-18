"""
ExperimentEngine – orchestrates the full STEP 1 inference loop.

Loop structure::

    for dataset in datasets:
        for model in models:
            for prompt_mode in prompt_modes:
                load model once
                open CSV writer
                for noise_type in noise_types:
                    for level in levels:
                        for noise_seed in range(n_noise_seeds):
                            for image in dataset:
                                apply noise on-the-fly
                                predict
                                compute metrics
                                write row
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.model_manager import ModelManager
from datasets.dataset_registry import build_dataset
from metrics.metric_manager import MetricManager
from noises.noise_manager import NoiseManager


# ── helpers ──────────────────────────────────────────────────────────────

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


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim > 2:
        m = np.squeeze(m)
    return (m > 0).astype(np.uint8)


def _sanitize_id(image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id))


def _parse_level_idx(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


# ── noise case dataclass ────────────────────────────────────────────────

@dataclass(frozen=True)
class NoiseCase:
    noise_type: str
    noise_level: str
    noise_seed: int


# ── ExperimentEngine ────────────────────────────────────────────────────

class ExperimentEngine:
    """
    Orchestrates STEP 1: run experiments over
    ``datasets × models × prompt_modes × noise_types × levels × seeds × images``.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    """

    def __init__(self, cfg: Dict[str, Any], *, device: Optional[str] = None) -> None:
        self.cfg = cfg
        exp = cfg.get("exp", cfg.get("experiment", {}))
        self.exp_name: str = exp.get("name", "experiment")
        out_root = Path(exp.get("out_root", "outputs"))
        self.exp_dir: Path = out_root / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.device: str = device or str(cfg.get("device", "cpu"))
        self.model_manager = ModelManager(device=self.device)
        self.noise_manager = NoiseManager(
            protocols=cfg.get("protocols", {}).get("coupled_presets", {}),
            noise_config=cfg.get("noise_config", {}),
        )
        self.metric_manager = MetricManager()

        # Stage-1 options
        s1 = cfg.get("stage1", {})
        self.save_artifacts: bool = bool(s1.get("save_artifacts", True))
        self.artifact_samples_per_case: int = int(s1.get("artifact_samples_per_case", 5))

    # ── public API ───────────────────────────────────────────────────────

    def run(
        self,
        *,
        max_samples: Optional[int] = None,
        dataset_filter: Optional[List[str]] = None,
        model_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the full STEP 1 experiment loop and return a summary."""
        datasets_cfg = self.cfg.get("datasets", [])
        models_cfg = self.cfg.get("models", [])
        if not datasets_cfg or not models_cfg:
            raise RuntimeError("Config must include non-empty 'datasets' and 'models'.")

        noise_cases = self._build_noise_cases()
        if not noise_cases:
            raise RuntimeError(
                "No noise cases built from config. "
                "Check 'noises', 'levels', and 'protocols.coupled_presets'."
            )

        ds_filter = set(dataset_filter or [])
        mdl_filter = set(model_filter or [])
        manifest_rows: List[Dict[str, str]] = []
        artifact_counts: Dict[str, int] = {}

        for ds_cfg in datasets_cfg:
            ds_name = str(ds_cfg.get("name"))
            if ds_filter and ds_name not in ds_filter:
                continue
            dataset = build_dataset(ds_cfg)
            n_samples = len(dataset)
            if max_samples and max_samples > 0:
                n_samples = min(n_samples, max_samples)

            for mdl_cfg in models_cfg:
                mdl_name = str(mdl_cfg.get("name"))
                runner_name = str(mdl_cfg.get("runner", mdl_name))
                if mdl_filter and mdl_name not in mdl_filter and runner_name not in mdl_filter:
                    continue

                prompt_modes = self._get_prompt_modes(mdl_cfg)
                for pm in prompt_modes:
                    runner = self.model_manager.get_model(
                        runner_name, prompt_mode=pm, model_cfg=mdl_cfg,
                    )

                    csv_dir = self.exp_dir / _slugify(ds_name) / _slugify(mdl_name)
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    raw_name = (
                        f"{_slugify(runner_name)}_{_slugify(ds_name)}_"
                        f"{_prompt_suffix(pm)}_raw.csv"
                    )
                    raw_csv = csv_dir / raw_name

                    self._run_inner_loop(
                        runner=runner,
                        dataset=dataset,
                        n_samples=n_samples,
                        noise_cases=noise_cases,
                        ds_name=ds_name,
                        mdl_name=mdl_name,
                        prompt_mode=pm,
                        raw_csv=raw_csv,
                        artifact_counts=artifact_counts,
                    )

                    manifest_rows.append({
                        "dataset": ds_name,
                        "model": mdl_name,
                        "runner": runner_name,
                        "prompt_mode": pm,
                        "raw_csv": str(raw_csv),
                    })

        self._write_manifest(manifest_rows)
        return {
            "experiment": self.exp_name,
            "exp_dir": str(self.exp_dir),
            "n_csv_files": len(manifest_rows),
        }

    # ── inner loop ───────────────────────────────────────────────────────

    def _run_inner_loop(
        self,
        *,
        runner,
        dataset,
        n_samples: int,
        noise_cases: List[NoiseCase],
        ds_name: str,
        mdl_name: str,
        prompt_mode: str,
        raw_csv: Path,
        artifact_counts: Dict[str, int],
    ) -> None:
        with open(raw_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
            writer.writeheader()

            total = n_samples * len(noise_cases)
            desc = f"{ds_name}/{mdl_name}/{prompt_mode}"
            pbar = tqdm(total=total, desc=desc, leave=False)

            for case in noise_cases:
                for idx in range(n_samples):
                    sample = dataset[idx]
                    image_id = str(
                        sample.get("image_id", sample.get("id", idx))
                    )
                    image = np.asarray(sample["image"], dtype=np.uint8)
                    gt_mask = _to_uint8_mask(
                        sample.get("mask", sample.get("gt_mask"))
                    )

                    noisy_image = self.noise_manager.apply_noise(
                        image,
                        noise_type=case.noise_type,
                        level=case.noise_level,
                        seed=case.noise_seed,
                        dataset_name=ds_name,
                        image_id=image_id,
                    )

                    pred_mask = _to_uint8_mask(
                        runner.predict(noisy_image, prompt={"gt_mask": gt_mask})
                    )

                    m = self.metric_manager.compute(pred_mask, gt_mask)

                    writer.writerow({
                        "dataset": ds_name,
                        "model": mdl_name,
                        "prompt_mode": prompt_mode,
                        "noise_type": case.noise_type,
                        "noise_level": case.noise_level,
                        "noise_seed": case.noise_seed,
                        "image_id": image_id,
                        **m,
                    })

                    self._maybe_save_artifacts(
                        artifact_counts=artifact_counts,
                        ds_name=ds_name,
                        mdl_name=mdl_name,
                        prompt_mode=prompt_mode,
                        case=case,
                        image_id=image_id,
                        image=image,
                        noisy_image=noisy_image,
                        gt_mask=gt_mask,
                        pred_mask=pred_mask,
                    )
                    pbar.update(1)
            pbar.close()

    # ── noise case builder ───────────────────────────────────────────────

    def _build_noise_cases(self) -> List[NoiseCase]:
        levels = self._get_levels()
        noise_names = self._get_noise_names()
        presets = self.cfg.get("protocols", {}).get("coupled_presets", {})
        n_seeds = self.noise_manager.n_noise_seeds

        cases: List[NoiseCase] = []
        for noise_name in noise_names:
            preset_levels = presets.get(noise_name, {})
            if not isinstance(preset_levels, dict):
                continue
            for level in levels:
                if level == "L0":
                    for s in range(n_seeds):
                        cases.append(NoiseCase(noise_name, level, s))
                    continue
                if level not in preset_levels:
                    continue
                for s in range(n_seeds):
                    cases.append(NoiseCase(noise_name, level, s))
        return cases

    def _get_levels(self) -> List[str]:
        raw = self.cfg.get("levels")
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return [f"L{i}" for i in range(10)]

    def _get_noise_names(self) -> List[str]:
        raw = self.cfg.get("noises")
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, dict):
            enabled = raw.get("enabled")
            if isinstance(enabled, list):
                return [str(x) for x in enabled]
        presets = self.cfg.get("protocols", {}).get("coupled_presets", {})
        return list(presets.keys())

    def _get_prompt_modes(self, mdl_cfg: Dict[str, Any]) -> List[str]:
        from models.wrappers.prompt_utils import normalize_prompt_mode

        raw = mdl_cfg.get("prompt_modes", mdl_cfg.get("mode", self.cfg.get("prompt_modes", [])))
        if isinstance(raw, str):
            raw = [raw]
        if not raw:
            raw = ["prompt_bbox"]
        return list(dict.fromkeys(normalize_prompt_mode(x) for x in raw))

    # ── artifact saving ──────────────────────────────────────────────────

    def _maybe_save_artifacts(
        self,
        *,
        artifact_counts: Dict[str, int],
        ds_name: str,
        mdl_name: str,
        prompt_mode: str,
        case: NoiseCase,
        image_id: str,
        image: np.ndarray,
        noisy_image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
    ) -> None:
        if not self.save_artifacts:
            return
        case_key = (
            f"{ds_name}|{mdl_name}|{prompt_mode}|"
            f"{case.noise_type}|{case.noise_level}|{case.noise_seed}"
        )
        if self.artifact_samples_per_case > 0 and artifact_counts.get(case_key, 0) >= self.artifact_samples_per_case:
            return
        artifact_counts[case_key] = artifact_counts.get(case_key, 0) + 1

        sid = _sanitize_id(image_id)
        out_dir = (
            self.exp_dir / "artifacts"
            / _slugify(ds_name) / _slugify(mdl_name)
            / _prompt_suffix(prompt_mode)
            / _slugify(case.noise_type) / case.noise_level
            / f"seed{case.noise_seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.astype(np.uint8)).save(out_dir / f"{sid}_original.png")
        Image.fromarray(noisy_image.astype(np.uint8)).save(out_dir / f"{sid}_noisy.png")
        Image.fromarray((_to_uint8_mask(gt_mask) * 255)).save(out_dir / f"{sid}_gt.png")
        Image.fromarray((_to_uint8_mask(pred_mask) * 255)).save(out_dir / f"{sid}_pred.png")

    # ── manifest ─────────────────────────────────────────────────────────

    def _write_manifest(self, rows: List[Dict[str, str]]) -> None:
        path = self.exp_dir / "raw_files_manifest.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["dataset", "model", "runner", "prompt_mode", "raw_csv"],
            )
            writer.writeheader()
            writer.writerows(rows)
