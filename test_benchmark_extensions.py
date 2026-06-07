#!/usr/bin/env python3
"""Smoke tests for benchmark extension points without loading real models."""

from __future__ import annotations

import math
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from core.experiment_engine import (
    ExperimentEngine,
    NoiseCase,
    _build_processed_sample_key,
)
from analysis.aggregator import MetricAggregator
from analysis.comprehensive_statistics import ComprehensiveStatistics
from metrics.metric_manager import MetricManager
from models.wrappers.prompt_utils import resolve_prompt_variant


def _mask() -> np.ndarray:
    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 10:22] = 1
    return m


def test_metrics() -> None:
    gt = _mask()
    pred = np.zeros_like(gt)
    pred[9:25, 11:23] = 1
    out = MetricManager.compute(
        pred,
        gt,
        add_hd95=True,
        add_physical_distance=True,
        spacing=None,
    )
    assert "HD" in out
    assert out["HD_px"] == out["HD"]
    assert math.isfinite(out["HD95_px"])
    assert math.isnan(out["HD95_mm"])

    no_legacy_hd = MetricManager.compute(
        pred,
        gt,
        add_hd95=True,
        keep_legacy_hd=False,
    )
    assert "HD" not in no_legacy_hd
    assert "HD_px" in no_legacy_hd
    assert "HD95_px" in no_legacy_hd

    out_mm = MetricManager.compute(
        pred,
        gt,
        spacing=(2.0, 1.0),
        add_hd95=True,
        add_physical_distance=True,
    )
    assert math.isfinite(out_mm["HD_mm"])
    assert math.isfinite(out_mm["HD95_mm"])

    same = MetricManager.compute(gt, gt, add_hd95=True)
    assert same["HD"] == 0.0
    assert same["HD95_px"] == 0.0

    empty = np.zeros_like(gt)
    one_empty = MetricManager.compute(empty, gt, add_hd95=True)
    assert math.isnan(one_empty["HD"])
    assert math.isnan(one_empty["HD95_px"])

    both_empty = MetricManager.compute(empty, empty, add_hd95=True)
    assert both_empty["HD"] == 0.0
    assert both_empty["HD95_px"] == 0.0

    shifted_pred = np.zeros((8, 8), dtype=np.uint8)
    shifted_gt = np.zeros_like(shifted_pred)
    shifted_pred[3, 2] = 1
    shifted_gt[2, 2] = 1
    shifted = MetricManager.compute(
        shifted_pred,
        shifted_gt,
        spacing=(2.0, 1.0),
        add_hd95=True,
        add_physical_distance=True,
    )
    assert shifted["HD_px"] == 1.0
    assert shifted["HD95_px"] == 1.0
    assert shifted["HD_mm"] == 2.0
    assert shifted["HD95_mm"] == 2.0


def test_prompt_variants() -> None:
    gt = _mask()
    image_shape = gt.shape
    bbox_variants = {
        "bbox_expand_10": {"name": "bbox_expand_10", "type": "expand", "expand": 0.10},
        "bbox_shift_10": {
            "name": "bbox_shift_10",
            "type": "translate",
            "shift_x": [-0.10, 0.10],
            "shift_y": [-0.10, 0.10],
            "relative_to": "bbox_size",
        },
        "bbox_shrink_10": {
            "name": "bbox_shrink_10",
            "type": "shrink",
            "shrink": 0.10,
            "min_size": 4,
            "clip_to_image": True,
        },
    }
    for variant in bbox_variants.values():
        resolved = resolve_prompt_variant(
            {"gt_mask": gt},
            image_shape,
            "prompt_bbox",
            prompt_variant=variant,
            deterministic_parts=["BUSI", "img", "gaussian", "L1", 0, variant["name"]],
        )
        x0, y0, x1, y1 = resolved["bbox"]
        assert 0 <= x0 <= x1 < image_shape[1]
        assert 0 <= y0 <= y1 < image_shape[0]

    point_variant = {
        "name": "point_random_inside",
        "type": "random_inside_mask",
        "n_random": 1,
    }
    resolved = resolve_prompt_variant(
        {"gt_mask": gt},
        image_shape,
        "prompt_point",
        prompt_variant=point_variant,
        deterministic_parts=["BUSI", "img", "gaussian", "L1", 0, "point_random_inside"],
    )
    x, y = resolved["point"]
    assert gt[y, x] > 0

    bbox_center = {
        "name": "bbox_gt_5",
        "type": "baseline",
        "expand": 0.05,
    }
    resolved = resolve_prompt_variant(
        {"gt_mask": gt},
        image_shape,
        "prompt_bbox",
        prompt_variant=bbox_center,
    )
    assert resolved["is_bbox_center_inside_mask"] == 1.0


def test_resume_key_and_paths() -> None:
    a = _build_processed_sample_key(
        dataset_name="BUSI",
        image_id="img",
        model_name="SAM2",
        noise_type="gaussian",
        level="L1",
        prompt_mode="prompt_bbox",
        noise_seed=0,
        prompt_variant="bbox_gt_5",
    )
    b = _build_processed_sample_key(
        dataset_name="BUSI",
        image_id="img",
        model_name="SAM2",
        noise_type="gaussian",
        level="L1",
        prompt_mode="prompt_bbox",
        noise_seed=0,
        prompt_variant="bbox_expand_10",
    )
    assert a != b

    with tempfile.TemporaryDirectory() as td:
        old_engine = ExperimentEngine({
            "exp": {"name": "bench", "out_root": td},
            "protocols": {"coupled_presets": {}},
            "noise_config": {"base_seed": 42, "n_noise_seeds": 1},
        })
        old_path = old_engine._raw_csv_path(
            ds_name="BUSI",
            mdl_name="SAM2",
            runner_name="SAM2",
            prompt_mode="prompt_bbox",
            prompt_variant="default",
        )
        assert old_path.name == "sam2_busi_bbox_raw.csv"

        new_engine = ExperimentEngine({
            "exp": {"name": "bench", "out_root": td},
            "prompt_variants": {"enabled": True},
            "protocols": {"coupled_presets": {}},
            "noise_config": {"base_seed": 42, "n_noise_seeds": 1},
        })
        new_path = new_engine._raw_csv_path(
            ds_name="BUSI",
            mdl_name="SAM2",
            runner_name="SAM2",
            prompt_mode="prompt_bbox",
            prompt_variant="bbox_gt_5",
        )
        assert "prompt_variants" in new_path.as_posix()
        assert new_path.name == "sam2_busi_prompt_bbox_bbox_gt_5_raw.csv"


def test_reuse_noisy_image() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "external"
        case_dir = root / "busi" / "gaussian" / "L1" / "seed0"
        case_dir.mkdir(parents=True)
        arr = np.full((8, 8), 77, dtype=np.uint8)
        Image.fromarray(arr).save(case_dir / "img_noisy.png")

        engine = ExperimentEngine({
            "exp": {"name": "bench", "out_root": str(Path(td) / "out")},
            "stage1": {
                "cache_noisy_images": False,
                "reuse_existing_noisy_images": True,
                "existing_noisy_root": str(root),
                "fallback_generate_noise_if_missing": False,
            },
            "protocols": {"coupled_presets": {}},
            "noise_config": {"base_seed": 42, "n_noise_seeds": 1},
        })
        result = engine._get_noisy_image(
            image=np.zeros((8, 8), dtype=np.uint8),
            ds_name="BUSI",
            image_id="img",
            case=NoiseCase("gaussian", "L1", 0),
        )
        assert result.source == "external_cache"
        assert np.array_equal(result.image, arr)


def test_aggregate_fallback_and_variant_grouping() -> None:
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "legacy_raw.csv"
        with raw.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "dataset",
                    "model",
                    "prompt_mode",
                    "noise_type",
                    "noise_level",
                    "noise_seed",
                    "image_id",
                    "Dice",
                    "IoU",
                    "Recall",
                    "Precision",
                    "F1",
                    "HD",
                    "params",
                    "trainable_params",
                    "FLOPs",
                    "GFLOPs",
                    "GLOPs",
                    "is_gt_empty",
                    "is_pred_empty",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_bbox",
                "noise_type": "gaussian",
                "noise_level": "L1",
                "noise_seed": 0,
                "image_id": "a",
                "Dice": 0.4,
                "IoU": 0.3,
                "Recall": 0.5,
                "Precision": 0.6,
                "F1": 0.55,
                "HD": 2.0,
                "params": 1000,
                "trainable_params": 100,
                "FLOPs": 2000000000,
                "GFLOPs": 2.0,
                "GLOPs": 2.0,
                "is_gt_empty": 0,
                "is_pred_empty": 0,
            })

        stats = MetricAggregator().aggregate_file(raw)
        assert stats.loc[0, "experiment_type"] == "main_prompt_mode_benchmark"
        assert stats.loc[0, "prompt_variant"] == "default"
        assert stats.loc[0, "failure_rate_dice_lt_0_5"] == 1.0
        assert stats.loc[0, "params"] == 1000
        assert stats.loc[0, "GFLOPs"] == 2.0

        stats_csv = Path(td) / "statistics_merged.csv"
        stats.to_csv(stats_csv, index=False)
        comprehensive = ComprehensiveStatistics(stats_csv, Path(td) / "statistics")
        profile = comprehensive.model_profile_summary()
        assert profile.loc[0, "params"] == 1000
        model_summary = comprehensive.model_summary()
        assert "GFLOPs" in model_summary.columns

        variant_raw = Path(td) / "variant_raw.csv"
        with variant_raw.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "experiment_type",
                    "dataset",
                    "model",
                    "prompt_mode",
                    "prompt_variant",
                    "noise_type",
                    "noise_level",
                    "noise_seed",
                    "image_id",
                    "Dice",
                    "HD95_px",
                    "is_bbox_center_inside_mask",
                ],
            )
            writer.writeheader()
            for variant, dice in [("bbox_gt_5", 0.8), ("bbox_expand_10", 0.7)]:
                writer.writerow({
                    "experiment_type": "prompt_variant_benchmark",
                    "dataset": "BUSI",
                    "model": "SAM2",
                    "prompt_mode": "prompt_bbox",
                    "prompt_variant": variant,
                    "noise_type": "gaussian",
                    "noise_level": "L1",
                    "noise_seed": 0,
                    "image_id": variant,
                    "Dice": dice,
                    "HD95_px": 3.0,
                    "is_bbox_center_inside_mask": 1,
                })

        variant_stats = MetricAggregator().aggregate_file(variant_raw)
        assert set(variant_stats["prompt_variant"]) == {"bbox_gt_5", "bbox_expand_10"}


def test_prompt_variant_summary_adds_filtered_defaults() -> None:
    with tempfile.TemporaryDirectory() as td:
        stats_csv = Path(td) / "statistics_merged.csv"
        rows = [
            {
                "experiment_type": "main_prompt_mode_benchmark",
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_bbox",
                "prompt_variant": "default",
                "noise_type": "gaussian",
                "noise_level": "L1",
                "Dice": 0.90,
            },
            {
                "experiment_type": "main_prompt_mode_benchmark",
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_bbox",
                "prompt_variant": "default",
                "noise_type": "gaussian",
                "noise_level": "L2",
                "Dice": 0.10,
            },
            {
                "experiment_type": "main_prompt_mode_benchmark",
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_point",
                "prompt_variant": "default",
                "noise_type": "gaussian",
                "noise_level": "L1",
                "Dice": 0.85,
            },
            {
                "experiment_type": "main_prompt_mode_benchmark",
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_point",
                "prompt_variant": "default",
                "noise_type": "gaussian",
                "noise_level": "L2",
                "Dice": 0.15,
            },
            {
                "experiment_type": "prompt_variant_benchmark",
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_bbox",
                "prompt_variant": "bbox_expand_10",
                "noise_type": "gaussian",
                "noise_level": "L1",
                "Dice": 0.70,
            },
            {
                "experiment_type": "prompt_variant_benchmark",
                "dataset": "BUSI",
                "model": "SAM2",
                "prompt_mode": "prompt_point",
                "prompt_variant": "point_centroid",
                "noise_type": "gaussian",
                "noise_level": "L1",
                "Dice": 0.60,
            },
        ]
        pd.DataFrame(rows).to_csv(stats_csv, index=False)

        comprehensive = ComprehensiveStatistics(stats_csv, Path(td) / "statistics")
        summary = comprehensive.prompt_variant_summary()
        observed = set(zip(summary["prompt_mode"], summary["prompt_variant"], summary["noise_level"]))
        assert ("prompt_bbox", "bbox_default", "L1") in observed
        assert ("prompt_point", "point_default", "L1") in observed
        assert ("prompt_bbox", "bbox_default", "L2") not in observed
        assert ("prompt_point", "point_default", "L2") not in observed

        comparison = comprehensive.prompt_variant_comparison()
        labels = set(comparison["prompt_variant_comparison"])
        assert "prompt_bbox:bbox_default" in labels
        assert "prompt_point:point_default" in labels


if __name__ == "__main__":
    test_metrics()
    test_prompt_variants()
    test_resume_key_and_paths()
    test_reuse_noisy_image()
    test_aggregate_fallback_and_variant_grouping()
    test_prompt_variant_summary_adds_filtered_defaults()
    print("benchmark extension smoke tests passed")
