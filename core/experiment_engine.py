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
import gc
import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Sequence, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.model_manager import ModelManager
from datasets.dataset_registry import build_dataset
from metrics.metric_manager import MetricManager
from models.wrappers.prompt_utils import (
    prompt_bbox_stats,
    resolve_prompt,
    resolve_prompt_variant,
)
from noises.noise_manager import NoiseManager


# ── helpers ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_RAW_COLUMNS = [
    "dataset",
    "model",
    "prompt_mode",
    "noise_type",
    "noise_level",
    "noise_seed",
    "image_id",
    "prompt_x",
    "prompt_y",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "gt_fg_pixels",
    "pred_fg_pixels",
    "is_gt_empty",
    "is_pred_empty",
    "IoU",
    "Dice",
    "Recall",
    "Precision",
    "F1",
    "HD",
]

RAW_COLUMNS = BASE_RAW_COLUMNS

PROMPT_METADATA_COLUMNS = [
    "experiment_type",
    "prompt_variant",
    "bbox_variant",
    "point_variant",
    "prompt_source",
    "prompt_has_bbox",
    "prompt_has_point",
    "is_bbox_center_inside_mask",
]

OUTPUT_PATH_COLUMNS = [
    "pred_mask_path",
    "gt_mask_source_path",
    "image_source_path",
    "noisy_image_path",
    "noisy_image_source",
]

HD95_COLUMNS = ["HD_px", "HD95_px"]
PHYSICAL_DISTANCE_COLUMNS = ["HD_mm", "HD95_mm"]
PERFORMANCE_COLUMNS = ["inference_time_ms", "FPS"]

RESUME_KEY_COLUMNS = {
    "dataset",
    "model",
    "prompt_mode",
    "noise_type",
    "noise_level",
    "noise_seed",
    "image_id",
}


def _slugify(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower())
    return out.strip("_") or "item"


def _prompt_suffix(prompt_mode: str) -> str:
    mapping = {
        "prompt_point": "point",
        "prompt_multi_point": "multipoint",
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


def _fg_stats(mask: np.ndarray) -> Dict[str, int]:
    fg = int(np.asarray(mask).astype(bool).sum())
    return {
        "fg_pixels": fg,
        "is_empty": 1 if fg == 0 else 0,
    }


def _sanitize_id(image_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_id))


def _parse_level_idx(level: str) -> int:
    m = re.search(r"(\d+)", str(level))
    return int(m.group(1)) if m else 10**9


def _append_columns(columns: List[str], extra: Sequence[str]) -> None:
    for col in extra:
        if col not in columns:
            columns.append(col)


def _prompt_row_fields(prompt: Dict[str, Any]) -> Dict[str, int]:
    point = prompt.get("point")
    bbox = prompt.get("bbox")
    row = {
        "prompt_x": int(point[0]) if point is not None else -1,
        "prompt_y": int(point[1]) if point is not None else -1,
        "bbox_x0": int(bbox[0]) if bbox is not None else -1,
        "bbox_y0": int(bbox[1]) if bbox is not None else -1,
        "bbox_x1": int(bbox[2]) if bbox is not None else -1,
        "bbox_y1": int(bbox[3]) if bbox is not None else -1,
    }
    row.update(prompt_bbox_stats(prompt))
    return row


def _prompt_metadata_row_fields(
    prompt: Dict[str, Any],
    *,
    experiment_type: str,
) -> Dict[str, Any]:
    return {
        "experiment_type": experiment_type,
        "prompt_variant": prompt.get("prompt_variant", "default"),
        "bbox_variant": prompt.get("bbox_variant", "default"),
        "point_variant": prompt.get("point_variant", "default"),
        "prompt_source": prompt.get("prompt_source", "legacy_default"),
        "prompt_has_bbox": int(prompt.get("prompt_has_bbox", 1 if prompt.get("bbox") is not None else 0)),
        "prompt_has_point": int(prompt.get("prompt_has_point", 1 if prompt.get("point") is not None else 0)),
        "is_bbox_center_inside_mask": prompt.get("is_bbox_center_inside_mask", float("nan")),
    }


# ── noise case dataclass ────────────────────────────────────────────────

@dataclass(frozen=True)
class NoiseCase:
    noise_type: str
    noise_level: str
    noise_seed: int


@dataclass(frozen=True)
class ProcessedSampleKey:
    """Unique identifier for a single saved stage-1 inference sample."""

    dataset_name: str
    image_id: str
    model_name: str
    noise_type: str
    level: str
    prompt_mode: str
    noise_seed: int
    prompt_variant: str = "default"


@dataclass(frozen=True)
class ResumeState:
    """Tracks whether an existing raw CSV can be reused for resume."""

    processed_keys: Set[ProcessedSampleKey]
    can_append: bool


@dataclass(frozen=True)
class NoisyImageResult:
    image: np.ndarray
    path: Optional[Path]
    source: str


def _build_processed_sample_key(
    *,
    dataset_name: str,
    image_id: str,
    model_name: str,
    noise_type: str,
    level: str,
    prompt_mode: str,
    noise_seed: int,
    prompt_variant: str = "default",
) -> ProcessedSampleKey:
    """Build the canonical resume key for a single sample."""
    return ProcessedSampleKey(
        dataset_name=str(dataset_name),
        image_id=str(image_id),
        model_name=str(model_name),
        noise_type=str(noise_type),
        level=str(level),
        prompt_mode=str(prompt_mode),
        noise_seed=int(noise_seed),
        prompt_variant=str(prompt_variant or "default"),
    )


def _load_resume_state(
    output_path: Path,
    *,
    expected_columns: Optional[Collection[str]] = None,
    prompt_variants_enabled: bool = False,
) -> ResumeState:
    """
    Load processed sample keys from an existing raw CSV.

    The CSV is considered resumable only when it exists, is non-empty, and
    exposes the columns required to reconstruct a sample identity.
    """
    if not output_path.exists() or not output_path.is_file():
        return ResumeState(processed_keys=set(), can_append=False)

    try:
        if output_path.stat().st_size == 0:
            return ResumeState(processed_keys=set(), can_append=False)
    except OSError:
        return ResumeState(processed_keys=set(), can_append=False)

    processed_keys: Set[ProcessedSampleKey] = set()
    try:
        with output_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or [])
            if expected_columns is not None and not set(expected_columns).issubset(fieldnames):
                return ResumeState(processed_keys=set(), can_append=False)
            if not RESUME_KEY_COLUMNS.issubset(fieldnames):
                return ResumeState(processed_keys=set(), can_append=False)
            if prompt_variants_enabled and "prompt_variant" not in fieldnames:
                return ResumeState(processed_keys=set(), can_append=False)

            for row in reader:
                try:
                    processed_keys.add(
                        _build_processed_sample_key(
                            dataset_name=str(row["dataset"]),
                            image_id=str(row["image_id"]),
                            model_name=str(row["model"]),
                            noise_type=str(row["noise_type"]),
                            level=str(row["noise_level"]),
                            prompt_mode=str(row["prompt_mode"]),
                            noise_seed=int(row["noise_seed"]),
                            prompt_variant=str(row.get("prompt_variant", "default")),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    except (csv.Error, OSError):
        return ResumeState(processed_keys=set(), can_append=False)

    return ResumeState(processed_keys=processed_keys, can_append=True)


def is_already_processed(
    *,
    output_path: Path,
    processed_keys: Collection[ProcessedSampleKey],
    dataset_name: str,
    image_id: str,
    model_name: str,
    noise_type: str,
    level: str,
    prompt_mode: str,
    noise_seed: int,
    prompt_variant: str = "default",
) -> bool:
    """
    Return ``True`` when the expected stage-1 output already exists and is valid.

    In this pipeline, the canonical per-sample output is the row stored in the
    prompt-specific raw CSV. A sample is therefore resumable when:

    1. the raw CSV exists,
    2. the file is non-empty, and
    3. a valid row for the requested sample key is already present.
    """
    if not output_path.exists() or not output_path.is_file():
        return False

    try:
        if output_path.stat().st_size == 0:
            return False
    except OSError:
        return False

    sample_key = _build_processed_sample_key(
        dataset_name=dataset_name,
        image_id=image_id,
        model_name=model_name,
        noise_type=noise_type,
        level=level,
        prompt_mode=prompt_mode,
        noise_seed=noise_seed,
        prompt_variant=prompt_variant,
    )
    return sample_key in processed_keys


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

        metrics_cfg = cfg.get("metrics", {})
        self.add_hd95: bool = bool(metrics_cfg.get("add_hd95", False))
        self.add_physical_distance: bool = bool(metrics_cfg.get("add_physical_distance", False))
        self.spacing_source: str = str(metrics_cfg.get("spacing_source", "sample_meta"))
        self.fallback_spacing = metrics_cfg.get("fallback_spacing", None)
        self.keep_legacy_hd: bool = bool(metrics_cfg.get("keep_legacy_hd", True))

        perf_cfg = cfg.get("performance", {})
        self.log_inference_time: bool = bool(perf_cfg.get("log_inference_time", False))
        self.log_fps: bool = bool(perf_cfg.get("log_fps", False))

        pv_cfg = cfg.get("prompt_variants", None)
        self.prompt_variants_cfg: Dict[str, Any] = pv_cfg if isinstance(pv_cfg, dict) else {}
        self.prompt_variants_enabled: bool = bool(self.prompt_variants_cfg.get("enabled", False))
        self.log_prompt_metadata: bool = bool(pv_cfg is not None)
        self.bbox_variant_map: Dict[str, Dict[str, Any]] = self._variant_map("prompt_bbox")
        self.point_variant_map: Dict[str, Dict[str, Any]] = self._variant_map("prompt_point")

        # Stage-1 options
        s1 = cfg.get("stage1", {})
        self.save_artifacts: bool = bool(s1.get("save_artifacts", True))
        self.artifact_samples_per_case: int = int(s1.get("artifact_samples_per_case", 5))
        self.cache_noisy_images: bool = bool(s1.get("cache_noisy_images", True))
        self.clear_noise_cache_on_start: bool = bool(s1.get("clear_noise_cache_on_start", False))
        self.reuse_existing_noisy_images: bool = bool(s1.get("reuse_existing_noisy_images", False))
        self.fallback_generate_noise_if_missing: bool = bool(
            s1.get("fallback_generate_noise_if_missing", True)
        )
        self.save_relative_paths_only: bool = bool(s1.get("save_relative_paths_only", False))
        self.use_output_policy: bool = any(
            key in s1 for key in (
                "save_pred_masks",
                "save_gt_masks",
                "save_original_images",
                "save_noisy_images",
                "save_relative_paths_only",
            )
        )
        self.save_pred_masks: bool = bool(s1.get("save_pred_masks", self.save_artifacts))
        self.save_gt_masks: bool = bool(s1.get("save_gt_masks", self.save_artifacts))
        self.save_original_images: bool = bool(s1.get("save_original_images", self.save_artifacts))
        self.save_noisy_images: bool = bool(s1.get("save_noisy_images", self.save_artifacts))
        self.gc_collect_interval: int = int(max(0, s1.get("gc_collect_interval", 0)))
        self.cuda_cache_clear_interval: int = int(max(0, s1.get("cuda_cache_clear_interval", 0)))
        parallel_cfg = s1.get("parallel", {})
        self.parallel_cfg: Dict[str, Any] = parallel_cfg if isinstance(parallel_cfg, dict) else {}
        self.parallel_enabled: bool = bool(self.parallel_cfg.get("enabled", False))
        self.parallel_num_workers: int = int(max(1, self.parallel_cfg.get("num_workers", 1)))

        existing_root_cfg = s1.get("existing_noisy_root")
        self.existing_noisy_root: Optional[Path] = None
        if existing_root_cfg:
            existing_root = Path(str(existing_root_cfg))
            if not existing_root.is_absolute():
                existing_root = (PROJECT_ROOT / existing_root).resolve()
            self.existing_noisy_root = existing_root

        cache_dir_cfg = s1.get("noise_cache_dir")
        if cache_dir_cfg:
            cache_dir = Path(str(cache_dir_cfg))
            if not cache_dir.is_absolute():
                cache_dir = self.exp_dir / cache_dir
        else:
            cache_dir = self.exp_dir / "noise_cache"
        self.noise_cache_dir: Path = cache_dir
        if self.clear_noise_cache_on_start and self.noise_cache_dir.exists():
            shutil.rmtree(self.noise_cache_dir, ignore_errors=True)
        if self.cache_noisy_images:
            self.noise_cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_columns = self._build_raw_columns()
        self.model_complexity_rows: List[Dict[str, Any]] = []

    # ── public API ───────────────────────────────────────────────────────

    def _build_raw_columns(self) -> List[str]:
        columns = list(BASE_RAW_COLUMNS)
        if self.log_prompt_metadata:
            _append_columns(columns, PROMPT_METADATA_COLUMNS)
        if self.use_output_policy:
            _append_columns(columns, OUTPUT_PATH_COLUMNS)
        if self.add_hd95:
            _append_columns(columns, HD95_COLUMNS)
        if self.add_physical_distance:
            _append_columns(columns, PHYSICAL_DISTANCE_COLUMNS)
        if self.log_inference_time or self.log_fps:
            _append_columns(columns, PERFORMANCE_COLUMNS)
        return columns

    def _variant_map(self, prompt_mode: str) -> Dict[str, Dict[str, Any]]:
        variants = self.prompt_variants_cfg.get(prompt_mode, [])
        if not isinstance(variants, list):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in variants:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                out[name] = dict(item)
        return out

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
                    # Proactive cache cleanup before loading the next heavy model.
                    self._runtime_cache_cleanup(force=True)
                    runner = self.model_manager.get_model(
                        runner_name, prompt_mode=pm, model_cfg=mdl_cfg,
                    )
                    self._record_model_complexity(
                        runner=runner,
                        mdl_name=mdl_name,
                        runner_name=runner_name,
                        prompt_mode=pm,
                    )

                    try:
                        for prompt_variant in self._get_prompt_variants(pm):
                            prompt_variant_name = str(prompt_variant.get("name", "default"))
                            raw_csv = self._raw_csv_path(
                                ds_name=ds_name,
                                mdl_name=mdl_name,
                                runner_name=runner_name,
                                prompt_mode=pm,
                                prompt_variant=prompt_variant_name,
                            )

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
                                prompt_variant=prompt_variant,
                            )

                            manifest_rows.append({
                                "dataset": ds_name,
                                "model": mdl_name,
                                "runner": runner_name,
                                "prompt_mode": pm,
                                "prompt_variant": prompt_variant_name,
                                "raw_csv": str(raw_csv),
                            })
                    finally:
                        self._release_runner(runner)
                        runner = None

        self._write_manifest(manifest_rows)
        self._write_model_complexity()
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
        prompt_variant: Dict[str, Any],
    ) -> None:
        prompt_variant_name = str(prompt_variant.get("name", "default"))
        experiment_type = (
            "prompt_variant_benchmark"
            if self.prompt_variants_enabled
            else "main_prompt_mode_benchmark"
        )
        resume_state = _load_resume_state(
            raw_csv,
            expected_columns=self.raw_columns,
            prompt_variants_enabled=self.prompt_variants_enabled,
        )
        file_mode = "a" if resume_state.can_append else "w"

        with open(raw_csv, file_mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.raw_columns, extrasaction="ignore")
            if not resume_state.can_append:
                writer.writeheader()
                fh.flush()

            processed_keys = set(resume_state.processed_keys)
            clean_result_cache: Dict[tuple[str, int], Dict[str, Any]] = {}
            total = n_samples * len(noise_cases)
            desc = f"{ds_name}/{mdl_name}/{prompt_mode}/{prompt_variant_name}"
            pbar = tqdm(total=total, desc=desc, leave=False)
            step_idx = 0

            for case in noise_cases:
                for idx in range(n_samples):
                    sample = dataset[idx]
                    image_id = str(
                        sample.get("image_id", sample.get("id", idx))
                    )

                    if is_already_processed(
                        output_path=raw_csv,
                        processed_keys=processed_keys,
                        dataset_name=ds_name,
                        image_id=image_id,
                        model_name=mdl_name,
                        noise_type=case.noise_type,
                        level=case.noise_level,
                        prompt_mode=prompt_mode,
                        noise_seed=case.noise_seed,
                        prompt_variant=prompt_variant_name,
                    ):
                        tqdm.write(
                            "SKIP "
                            f"dataset={ds_name} "
                            f"model={mdl_name} "
                            f"prompt={prompt_mode} "
                            f"variant={prompt_variant_name} "
                            f"noise={case.noise_type} "
                            f"level={case.noise_level} "
                            f"seed={case.noise_seed} "
                            f"image_id={image_id}"
                        )
                        pbar.update(1)
                        step_idx += 1
                        self._runtime_cache_cleanup(step_idx=step_idx)
                        continue

                    image = np.asarray(sample["image"], dtype=np.uint8)
                    gt_mask = _to_uint8_mask(
                        sample.get("mask", sample.get("gt_mask"))
                    )
                    sample_meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
                    spacing = self._extract_spacing(sample)

                    clean_cache_key = None
                    cached_clean = None
                    if case.noise_level == "L0":
                        clean_cache_key = (image_id, int(case.noise_seed))
                        cached_clean = clean_result_cache.get(clean_cache_key)

                    if cached_clean is None:
                        noisy_result = self._get_noisy_image(
                            image=image,
                            ds_name=ds_name,
                            image_id=image_id,
                            case=case,
                        )
                        noisy_image = noisy_result.image

                        prompt_payload = {
                            "gt_mask": gt_mask,
                            "noise_seed": case.noise_seed,
                            "noise_level": case.noise_level,
                        }
                        if self.prompt_variants_enabled:
                            resolved_prompt = resolve_prompt_variant(
                                prompt_payload,
                                image.shape[:2],
                                prompt_mode,
                                prompt_variant=prompt_variant,
                                bbox_variants=self.bbox_variant_map,
                                point_variants=self.point_variant_map,
                                deterministic_parts=[
                                    ds_name,
                                    image_id,
                                    case.noise_type,
                                    case.noise_level,
                                    case.noise_seed,
                                    prompt_variant_name,
                                ],
                            )
                        else:
                            resolved_prompt = resolve_prompt(
                                prompt_payload,
                                image.shape[:2],
                                prompt_mode=prompt_mode,
                            )
                            if self.log_prompt_metadata:
                                resolved_prompt = resolve_prompt_variant(
                                    prompt_payload,
                                    image.shape[:2],
                                    prompt_mode,
                                    prompt_variant=None,
                                )
                        prompt_fields = _prompt_row_fields(resolved_prompt)
                        prompt_meta_fields = (
                            _prompt_metadata_row_fields(
                                resolved_prompt,
                                experiment_type=experiment_type,
                            )
                            if self.log_prompt_metadata
                            else {}
                        )

                        t0 = time.perf_counter()
                        pred_mask = _to_uint8_mask(
                            runner.predict(noisy_image, prompt=resolved_prompt)
                        )
                        inference_time_ms = (time.perf_counter() - t0) * 1000.0

                        gt_stats = _fg_stats(gt_mask)
                        pred_stats = _fg_stats(pred_mask)
                        m = self.metric_manager.compute(
                            pred_mask,
                            gt_mask,
                            spacing=spacing,
                            add_hd95=self.add_hd95,
                            add_physical_distance=self.add_physical_distance,
                            keep_legacy_hd=self.keep_legacy_hd,
                        )
                        perf_fields = self._performance_row_fields(inference_time_ms)

                        if clean_cache_key is not None:
                            clean_result_cache[clean_cache_key] = {
                                "prompt_fields": dict(prompt_fields),
                                "prompt_meta_fields": dict(prompt_meta_fields),
                                "pred_mask": pred_mask.copy(),
                                "gt_stats": dict(gt_stats),
                                "pred_stats": dict(pred_stats),
                                "metrics": dict(m),
                                "perf_fields": dict(perf_fields),
                                "noisy_path": noisy_result.path,
                                "noisy_source": noisy_result.source,
                            }
                    else:
                        noisy_result = NoisyImageResult(
                            image=image.copy(),
                            path=cached_clean.get("noisy_path"),
                            source=str(cached_clean.get("noisy_source", "clean_cache")),
                        )
                        noisy_image = noisy_result.image
                        prompt_fields = dict(cached_clean["prompt_fields"])
                        prompt_meta_fields = dict(cached_clean.get("prompt_meta_fields", {}))
                        pred_mask = np.asarray(cached_clean["pred_mask"], dtype=np.uint8).copy()
                        gt_stats = dict(cached_clean["gt_stats"])
                        pred_stats = dict(cached_clean["pred_stats"])
                        m = dict(cached_clean["metrics"])
                        perf_fields = dict(cached_clean.get("perf_fields", {}))

                    saved_paths = self._maybe_save_artifacts(
                        artifact_counts=artifact_counts,
                        ds_name=ds_name,
                        mdl_name=mdl_name,
                        prompt_mode=prompt_mode,
                        prompt_variant=prompt_variant_name,
                        case=case,
                        image_id=image_id,
                        image=image,
                        noisy_result=noisy_result,
                        gt_mask=gt_mask,
                        pred_mask=pred_mask,
                    )
                    path_fields = (
                        self._output_path_row_fields(
                            sample_meta=sample_meta,
                            noisy_result=noisy_result,
                            saved_paths=saved_paths,
                        )
                        if self.use_output_policy
                        else {}
                    )

                    row = {
                        "dataset": ds_name,
                        "model": mdl_name,
                        "prompt_mode": prompt_mode,
                        "noise_type": case.noise_type,
                        "noise_level": case.noise_level,
                        "noise_seed": case.noise_seed,
                        "image_id": image_id,
                        **prompt_meta_fields,
                        **prompt_fields,
                        "gt_fg_pixels": gt_stats["fg_pixels"],
                        "pred_fg_pixels": pred_stats["fg_pixels"],
                        "is_gt_empty": gt_stats["is_empty"],
                        "is_pred_empty": pred_stats["is_empty"],
                        **path_fields,
                        **m,
                        **perf_fields,
                    }
                    writer.writerow(row)
                    fh.flush()
                    processed_keys.add(
                        _build_processed_sample_key(
                            dataset_name=ds_name,
                            image_id=image_id,
                            model_name=mdl_name,
                            noise_type=case.noise_type,
                            level=case.noise_level,
                            prompt_mode=prompt_mode,
                            noise_seed=case.noise_seed,
                            prompt_variant=prompt_variant_name,
                        )
                    )
                    pbar.update(1)
                    step_idx += 1
                    self._runtime_cache_cleanup(step_idx=step_idx)
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

    def _get_prompt_variants(self, prompt_mode: str) -> List[Dict[str, Any]]:
        if not self.prompt_variants_enabled:
            return [{"name": "default", "type": "default"}]
        variants = self.prompt_variants_cfg.get(prompt_mode, [])
        if isinstance(variants, list) and variants:
            return [dict(v) for v in variants if isinstance(v, dict)]
        return [{"name": "default", "type": "default"}]

    def _raw_csv_path(
        self,
        *,
        ds_name: str,
        mdl_name: str,
        runner_name: str,
        prompt_mode: str,
        prompt_variant: str,
    ) -> Path:
        if self.prompt_variants_enabled:
            csv_dir = (
                self.exp_dir
                / "prompt_variants"
                / _slugify(prompt_mode)
                / _slugify(prompt_variant)
                / _slugify(ds_name)
                / _slugify(mdl_name)
            )
            raw_name = (
                f"{_slugify(runner_name)}_{_slugify(ds_name)}_"
                f"{_slugify(prompt_mode)}_{_slugify(prompt_variant)}_raw.csv"
            )
        else:
            csv_dir = self.exp_dir / _slugify(ds_name) / _slugify(mdl_name)
            raw_name = (
                f"{_slugify(runner_name)}_{_slugify(ds_name)}_"
                f"{_prompt_suffix(prompt_mode)}_raw.csv"
            )
        csv_dir.mkdir(parents=True, exist_ok=True)
        return csv_dir / raw_name

    def _performance_row_fields(self, inference_time_ms: float) -> Dict[str, float]:
        fields: Dict[str, float] = {}
        valid_time = np.isfinite(inference_time_ms) and inference_time_ms > 0
        if self.log_inference_time:
            fields["inference_time_ms"] = float(inference_time_ms) if valid_time else float("nan")
        if self.log_fps:
            fields["FPS"] = float(1000.0 / inference_time_ms) if valid_time else float("nan")
        return fields

    def _extract_spacing(self, sample: Dict[str, Any]) -> Optional[Sequence[float]]:
        if not self.add_physical_distance:
            return None
        spacing = None
        if self.spacing_source == "sample_meta":
            meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
            if isinstance(meta, dict):
                for key in ("spacing", "pixel_spacing", "pixdim"):
                    if key in meta:
                        spacing = meta.get(key)
                        break
        if spacing is None:
            spacing = self.fallback_spacing
        if spacing is None:
            return None
        try:
            vals = [float(v) for v in spacing]
        except (TypeError, ValueError):
            return None
        if len(vals) == 1:
            vals = [vals[0], vals[0]]
        if len(vals) < 2 or any(not np.isfinite(v) or v <= 0 for v in vals[:2]):
            return None
        return tuple(vals[:2])

    def _format_path(self, path: Optional[Path | str]) -> str:
        if path is None or str(path) == "":
            return ""
        p = Path(path)
        if self.save_relative_paths_only:
            try:
                return p.resolve().relative_to(PROJECT_ROOT).as_posix()
            except (OSError, ValueError):
                pass
        return str(p)

    def _source_path_from_meta(self, meta: Dict[str, Any], keys: Sequence[str]) -> str:
        if not isinstance(meta, dict):
            return ""
        for key in keys:
            value = meta.get(key)
            if value:
                return self._format_path(value)
        return ""

    def _output_path_row_fields(
        self,
        *,
        sample_meta: Dict[str, Any],
        noisy_result: NoisyImageResult,
        saved_paths: Dict[str, Optional[Path]],
    ) -> Dict[str, str]:
        noisy_path = saved_paths.get("noisy_image_path") or noisy_result.path
        noisy_source = noisy_result.source
        if saved_paths.get("noisy_image_path") is not None:
            noisy_source = "saved_artifact"
        return {
            "pred_mask_path": self._format_path(saved_paths.get("pred_mask_path")),
            "gt_mask_source_path": self._source_path_from_meta(
                sample_meta,
                ("mask_path", "gt_path", "gt_mask_path"),
            ),
            "image_source_path": self._source_path_from_meta(
                sample_meta,
                ("img_path", "image_path"),
            ),
            "noisy_image_path": self._format_path(noisy_path),
            "noisy_image_source": noisy_source,
        }

    def _record_model_complexity(
        self,
        *,
        runner,
        mdl_name: str,
        runner_name: str,
        prompt_mode: str,
    ) -> None:
        total_params = float("nan")
        trainable_params = float("nan")
        model_obj = getattr(runner, "_model", None) or getattr(runner, "_predictor", None)
        if model_obj is not None and not hasattr(model_obj, "parameters"):
            model_obj = getattr(model_obj, "model", model_obj)
        if model_obj is not None and hasattr(model_obj, "parameters"):
            try:
                total = 0
                trainable = 0
                for p in model_obj.parameters():
                    n = int(p.numel())
                    total += n
                    if getattr(p, "requires_grad", False):
                        trainable += n
                total_params = total
                trainable_params = trainable
            except Exception:
                total_params = float("nan")
                trainable_params = float("nan")
        cfg = getattr(runner, "model_cfg", {}) or {}
        gflops = cfg.get("GFLOPs", cfg.get("gflops", cfg.get("glops", float("nan"))))
        self.model_complexity_rows.append({
            "model": mdl_name,
            "runner": runner_name,
            "prompt_mode": prompt_mode,
            "device": self.device,
            "params": total_params,
            "trainable_params": trainable_params,
            "FLOPs": cfg.get("FLOPs", cfg.get("flops", float("nan"))),
            "GFLOPs": gflops,
            "GLOPs": gflops,
        })

    def _write_model_complexity(self) -> None:
        if not self.model_complexity_rows:
            return
        path = self.exp_dir / "model_complexity.csv"
        fieldnames = [
            "model",
            "runner",
            "prompt_mode",
            "device",
            "params",
            "trainable_params",
            "FLOPs",
            "GFLOPs",
            "GLOPs",
        ]
        seen = set()
        rows = []
        for row in self.model_complexity_rows:
            key = (row.get("model"), row.get("runner"), row.get("prompt_mode"), row.get("device"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _release_runner(self, runner) -> None:
        """Best-effort teardown to avoid keeping CUDA memory between model runs."""
        if runner is None:
            return

        for attr in ("_model", "_predictor"):
            obj = getattr(runner, attr, None)
            if obj is None:
                continue
            try:
                if hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
            try:
                setattr(runner, attr, None)
            except Exception:
                pass

        if hasattr(runner, "_processor"):
            try:
                setattr(runner, "_processor", None)
            except Exception:
                pass

        self._runtime_cache_cleanup(force=True)

    def _runtime_cache_cleanup(
        self,
        *,
        force: bool = False,
        step_idx: Optional[int] = None,
    ) -> None:
        """
        Runtime memory cleanup helper.

        - `force=True`: always run `gc.collect()` and CUDA cache cleanup.
        - periodic mode: run based on configured intervals.
        """
        do_gc = force
        do_cuda = force

        if not force and step_idx is not None:
            if self.gc_collect_interval > 0 and step_idx % self.gc_collect_interval == 0:
                do_gc = True
            if self.cuda_cache_clear_interval > 0 and step_idx % self.cuda_cache_clear_interval == 0:
                do_cuda = True

        if do_gc:
            gc.collect()

        if do_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass

    # ── noisy-image cache ────────────────────────────────────────────────

    def _noise_cache_path(
        self,
        *,
        ds_name: str,
        case: NoiseCase,
        image_id: str,
        image: np.ndarray,
    ) -> Path:
        sid = _sanitize_id(image_id)
        h, w = image.shape[:2]
        c = int(image.shape[2]) if image.ndim == 3 else 1
        dims = f"{h}x{w}x{c}"
        params = self.noise_manager.get_params(case.noise_type, case.noise_level) or {}
        sig_key = (
            f"{self.noise_manager.base_seed}|{case.noise_seed}|"
            f"{case.noise_type}|{case.noise_level}|{sorted(params.items())}"
        )
        cfg_sig = hashlib.sha1(sig_key.encode("utf-8")).hexdigest()[:10]
        return (
            self.noise_cache_dir
            / _slugify(ds_name)
            / _slugify(case.noise_type)
            / case.noise_level
            / f"seed{case.noise_seed}"
            / f"{sid}_{dims}_{cfg_sig}.npy"
        )

    def _get_noisy_image(
        self,
        *,
        image: np.ndarray,
        ds_name: str,
        image_id: str,
        case: NoiseCase,
    ) -> NoisyImageResult:
        cache_path = self._noise_cache_path(
            ds_name=ds_name,
            case=case,
            image_id=image_id,
            image=image,
        )

        if self.reuse_existing_noisy_images and self.existing_noisy_root is not None:
            external = self._load_external_noisy_image(
                image=image,
                ds_name=ds_name,
                image_id=image_id,
                case=case,
            )
            if external is not None:
                return external

        if self.cache_noisy_images and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if cached.shape == image.shape and cached.dtype == np.uint8:
                    return NoisyImageResult(
                        image=cached,
                        path=cache_path,
                        source="internal_cache",
                    )
            except Exception:
                pass

        if (
            self.reuse_existing_noisy_images
            and not self.fallback_generate_noise_if_missing
        ):
            raise FileNotFoundError(
                "Missing reusable noisy image and fallback generation is disabled: "
                f"dataset={ds_name}, image_id={image_id}, noise={case.noise_type}, "
                f"level={case.noise_level}, seed={case.noise_seed}"
            )

        noisy = self.noise_manager.apply_noise(
            image,
            noise_type=case.noise_type,
            level=case.noise_level,
            seed=case.noise_seed,
            dataset_name=ds_name,
            image_id=image_id,
        )
        noisy = np.asarray(noisy, dtype=np.uint8)
        saved_cache_path: Optional[Path] = None

        if self.cache_noisy_images:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
            try:
                with open(tmp_path, "wb") as f:
                    np.save(f, noisy, allow_pickle=False)
                if not cache_path.exists():
                    os.replace(tmp_path, cache_path)
                    saved_cache_path = cache_path
                else:
                    tmp_path.unlink(missing_ok=True)
                    saved_cache_path = cache_path
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        return NoisyImageResult(
            image=noisy,
            path=saved_cache_path,
            source="generated",
        )

    def _load_external_noisy_image(
        self,
        *,
        image: np.ndarray,
        ds_name: str,
        image_id: str,
        case: NoiseCase,
    ) -> Optional[NoisyImageResult]:
        if self.existing_noisy_root is None:
            return None
        sid = _sanitize_id(image_id)
        base_dir = (
            self.existing_noisy_root
            / _slugify(ds_name)
            / _slugify(case.noise_type)
            / case.noise_level
            / f"seed{case.noise_seed}"
        )
        candidates = [
            base_dir / f"{sid}_noisy.png",
            base_dir / f"{sid}.png",
            base_dir / f"{sid}_noisy.npy",
            base_dir / f"{sid}.npy",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".npy":
                    arr = np.load(path, allow_pickle=False)
                else:
                    arr = np.asarray(Image.open(path))
                    if arr.ndim == 3 and image.ndim == 2:
                        arr = np.asarray(Image.open(path).convert("L"))
                arr = np.asarray(arr, dtype=np.uint8)
                if arr.shape == image.shape:
                    return NoisyImageResult(
                        image=arr,
                        path=path,
                        source="external_cache",
                    )
            except Exception:
                continue
        return None

    # ── artifact saving ──────────────────────────────────────────────────

    def _maybe_save_artifacts(
        self,
        *,
        artifact_counts: Dict[str, int],
        ds_name: str,
        mdl_name: str,
        prompt_mode: str,
        prompt_variant: str,
        case: NoiseCase,
        image_id: str,
        image: np.ndarray,
        noisy_result: NoisyImageResult,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
    ) -> Dict[str, Optional[Path]]:
        paths: Dict[str, Optional[Path]] = {
            "pred_mask_path": None,
            "noisy_image_path": None,
        }
        if not self.save_artifacts and not self.use_output_policy:
            return paths

        sid = _sanitize_id(image_id)

        if self.use_output_policy:
            if self.save_original_images or self.save_gt_masks or self.save_noisy_images:
                shared_case_key = (
                    f"{ds_name}|{case.noise_type}|{case.noise_level}|{case.noise_seed}"
                )
                shared_count_key = f"shared|{shared_case_key}"
                can_save_shared = (
                    self.artifact_samples_per_case <= 0
                    or artifact_counts.get(shared_count_key, 0) < self.artifact_samples_per_case
                )
                if can_save_shared:
                    shared_dir = (
                        self.exp_dir
                        / "artifacts"
                        / "_shared"
                        / _slugify(ds_name)
                        / _slugify(case.noise_type)
                        / case.noise_level
                        / f"seed{case.noise_seed}"
                    )
                    saved_shared = False
                    if self.save_original_images:
                        saved_shared |= self._save_image_if_missing(
                            shared_dir / f"{sid}_original.png",
                            image.astype(np.uint8),
                        )
                    if self.save_noisy_images:
                        noisy_path = shared_dir / f"{sid}_noisy.png"
                        saved_shared |= self._save_image_if_missing(
                            noisy_path,
                            noisy_result.image.astype(np.uint8),
                        )
                        paths["noisy_image_path"] = noisy_path
                    if self.save_gt_masks:
                        saved_shared |= self._save_image_if_missing(
                            shared_dir / f"{sid}_gt.png",
                            (_to_uint8_mask(gt_mask) * 255),
                        )
                    if saved_shared:
                        artifact_counts[shared_count_key] = artifact_counts.get(shared_count_key, 0) + 1

            if self.save_pred_masks:
                pred_path = (
                    self.exp_dir
                    / "pred_masks"
                    / _slugify(ds_name)
                    / _slugify(mdl_name)
                    / _slugify(prompt_mode)
                    / _slugify(prompt_variant or "default")
                    / _slugify(case.noise_type)
                    / case.noise_level
                    / f"seed{case.noise_seed}"
                    / f"{sid}.png"
                )
                self._save_image_if_missing(
                    pred_path,
                    (_to_uint8_mask(pred_mask) * 255),
                )
                paths["pred_mask_path"] = pred_path
            return paths

        # Save shared original/noisy/gt once per (dataset, noise, level, seed, image_id),
        # then reuse across all model/prompt runs.
        shared_case_key = (
            f"{ds_name}|{case.noise_type}|{case.noise_level}|{case.noise_seed}"
        )
        shared_count_key = f"shared|{shared_case_key}"
        can_save_shared = (
            self.artifact_samples_per_case <= 0
            or artifact_counts.get(shared_count_key, 0) < self.artifact_samples_per_case
        )
        if can_save_shared:
            shared_dir = (
                self.exp_dir
                / "artifacts"
                / "_shared"
                / _slugify(ds_name)
                / _slugify(case.noise_type)
                / case.noise_level
                / f"seed{case.noise_seed}"
            )
            saved_shared = False
            saved_shared |= self._save_image_if_missing(
                shared_dir / f"{sid}_original.png", image.astype(np.uint8)
            )
            saved_shared |= self._save_image_if_missing(
                shared_dir / f"{sid}_noisy.png", noisy_result.image.astype(np.uint8)
            )
            saved_shared |= self._save_image_if_missing(
                shared_dir / f"{sid}_gt.png", (_to_uint8_mask(gt_mask) * 255)
            )
            if saved_shared:
                artifact_counts[shared_count_key] = artifact_counts.get(shared_count_key, 0) + 1

        # Keep prediction masks per model/prompt for comparison/overlay.
        pred_case_key = (
            f"{ds_name}|{mdl_name}|{prompt_mode}|"
            f"{case.noise_type}|{case.noise_level}|{case.noise_seed}"
        )
        pred_count_key = f"pred|{pred_case_key}"
        if (
            self.artifact_samples_per_case > 0
            and artifact_counts.get(pred_count_key, 0) >= self.artifact_samples_per_case
        ):
            return paths

        out_dir = (
            self.exp_dir / "artifacts"
            / _slugify(ds_name) / _slugify(mdl_name)
            / _prompt_suffix(prompt_mode)
            / _slugify(case.noise_type) / case.noise_level
            / f"seed{case.noise_seed}"
        )
        saved_pred = self._save_image_if_missing(
            out_dir / f"{sid}_pred.png", (_to_uint8_mask(pred_mask) * 255)
        )
        paths["pred_mask_path"] = out_dir / f"{sid}_pred.png"
        if saved_pred:
            artifact_counts[pred_count_key] = artifact_counts.get(pred_count_key, 0) + 1
        return paths

    @staticmethod
    def _save_image_if_missing(path: Path, arr: np.ndarray) -> bool:
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
        return True

    # ── manifest ─────────────────────────────────────────────────────────

    def _write_manifest(self, rows: List[Dict[str, str]]) -> None:
        path = self.exp_dir / "raw_files_manifest.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "dataset",
                    "model",
                    "runner",
                    "prompt_mode",
                    "prompt_variant",
                    "raw_csv",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
