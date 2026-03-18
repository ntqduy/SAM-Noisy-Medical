"""Main entry point for the 3-stage segmentation robustness pipeline.

    Stage 1  (run)        – ExperimentEngine: run inference under noise.
    Stage 1b (aggregate)  – StatisticsMerger: aggregate raw CSVs → stats.
    Stage 2  (visualize)  – Viz classes: generate PDF figures & tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── stage runners ────────────────────────────────────────────────────────

def _run_stage1(
    cfg: Dict[str, Any],
    *,
    max_samples,
    dataset_filter,
    model_filter,
    device: Optional[str] = None,
):
    from core.experiment_engine import ExperimentEngine

    engine = ExperimentEngine(cfg, device=device)
    summary = engine.run(
        max_samples=max_samples,
        dataset_filter=dataset_filter,
        model_filter=model_filter,
    )
    print(f"[Stage1] Completed on {engine.device}: {summary}")
    return summary


def _gpu_worker(
    cfg: Dict[str, Any],
    device: str,
    model_names: List[str],
    max_samples: Optional[int],
    dataset_filter: Optional[List[str]],
) -> Dict[str, Any]:
    """Worker function executed in a child process for multi-GPU parallel run."""
    return _run_stage1(
        cfg,
        max_samples=max_samples,
        dataset_filter=dataset_filter,
        model_filter=model_names,
        device=device,
    )


def _run_stage1_parallel(
    cfg: Dict[str, Any],
    devices: List[str],
    *,
    max_samples: Optional[int],
    dataset_filter: Optional[List[str]],
    model_filter: Optional[List[str]],
):
    """Split models across *devices* and run each subset in a separate process."""
    import multiprocessing as mp

    models_cfg = cfg.get("models", [])
    mdl_filter = set(model_filter or [])

    # Collect model names that will actually run
    model_names: List[str] = []
    for m in models_cfg:
        name = str(m.get("name"))
        runner = str(m.get("runner", name))
        if mdl_filter and name not in mdl_filter and runner not in mdl_filter:
            continue
        model_names.append(name)

    if not model_names:
        print("[Stage1] No models matched the filter.")
        return

    # Round-robin assignment of models → devices
    n_devices = len(devices)
    assignments: Dict[str, List[str]] = {d: [] for d in devices}
    for i, name in enumerate(model_names):
        assignments[devices[i % n_devices]].append(name)

    # Remove empty assignments
    assignments = {d: names for d, names in assignments.items() if names}

    if len(assignments) == 1:
        # Only one device has work → run directly, no subprocess overhead
        dev, names = next(iter(assignments.items()))
        _run_stage1(
            cfg,
            max_samples=max_samples,
            dataset_filter=dataset_filter,
            model_filter=names,
            device=dev,
        )
        return

    print(f"[Stage1] Multi-GPU: splitting {len(model_names)} model(s) across {len(assignments)} device(s)")
    for dev, names in assignments.items():
        print(f"  {dev}: {names}")

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    for dev, names in assignments.items():
        p = ctx.Process(
            target=_gpu_worker,
            args=(cfg, dev, names, max_samples, dataset_filter),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(
            f"[Stage1] {len(failed)}/{len(processes)} GPU worker(s) failed."
        )


def _run_stage1b(cfg: Dict[str, Any], exp_dir: Path):
    from analysis.stats_merger import StatisticsMerger

    merger = StatisticsMerger()
    summary = merger.run(exp_dir)
    print(f"[Stage1b] Completed: {summary}")
    return summary


def _run_stage2(cfg: Dict[str, Any], exp_dir: Path):
    from viz import (
        MetricPlotter,
        ModelComparisonPlotter,
        NoiseGalleryGenerator,
        PredictionVisualizer,
        PromptComparisonPlotter,
        StatisticalTableGenerator,
    )

    merged_csv = exp_dir / "statistics_merged.csv"
    if not merged_csv.exists():
        raise RuntimeError(
            f"Missing {merged_csv}. Run Stage 1b first."
        )

    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = exp_dir / "artifacts"

    # Level descriptors from config (falls back to defaults in each viz class)
    level_names = cfg.get("level_names")

    outputs: Dict[str, str] = {}

    # Metric curves & robustness
    mp = MetricPlotter(merged_csv, figures_dir, level_names=level_names)
    outputs["metric_curves_dice"] = str(mp.plot_metric_curves("Dice", "metric_curves_dice.pdf"))
    outputs["metric_curves_iou"] = str(mp.plot_metric_curves("IoU", "metric_curves_iou.pdf"))
    outputs["robustness_dice"] = str(mp.plot_robustness("Dice", "robustness_dice_drop.pdf"))

    # Model comparison
    mc = ModelComparisonPlotter(merged_csv, figures_dir, level_names=level_names)
    outputs["model_comparison_dice"] = str(mc.plot("Dice", "model_comparison_dice.pdf"))

    # Prompt visualization & comparison
    pc = PromptComparisonPlotter(figures_dir, merged_csv)
    outputs["prompt_visualization"] = str(pc.plot_schematic("prompt_visualization.pdf"))
    outputs["prompt_comparison"] = str(pc.plot_comparison("Dice", "prompt_comparison_dice.pdf"))

    # Noise gallery
    ng = NoiseGalleryGenerator(artifact_root, figures_dir, level_names=level_names)
    outputs["noise_gallery"] = str(ng.generate(filename="noise_gallery.pdf"))

    # Prediction overlay
    pv = PredictionVisualizer(artifact_root, figures_dir)
    outputs["prediction_overlay"] = str(pv.generate(filename="prediction_overlay.pdf"))

    # Statistical tables
    st = StatisticalTableGenerator(merged_csv, figures_dir, level_names=level_names)
    outputs["statistical_tables"] = str(st.generate("statistical_tables.pdf"))

    print("[Stage2] Completed:")
    for key, value in outputs.items():
        print(f"  - {key}: {value}")
    return outputs


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Segmentation robustness pipeline")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["run", "aggregate", "visualize", "all"],
        help="Pipeline stage to execute",
    )
    ap.add_argument("--max_samples", type=int, default=None, help="Optional sample limit per dataset")
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names to run in stage 'run'",
    )
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names/runners to run in stage 'run'",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda:0', 'cuda:1', 'cpu'). "
             "Overrides the config file setting.",
    )
    ap.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for parallel Stage 1. "
             "Models are split round-robin across cuda:0 … cuda:N-1.",
    )
    return ap.parse_args()


def main():
    from core.config_manager import ConfigManager

    args = parse_args()
    cm = ConfigManager(args.config)

    # CLI overrides for device selection
    if args.device is not None:
        cm.override_devices([args.device])
    elif args.num_gpus is not None and args.num_gpus >= 2:
        cm.override_devices([f"cuda:{i}" for i in range(args.num_gpus)])

    cfg = cm.cfg
    exp_dir = cm.exp_dir
    devices = cm.devices
    dataset_filter = [x.strip() for x in args.datasets.split(",")] if args.datasets else None
    model_filter = [x.strip() for x in args.models.split(",")] if args.models else None

    print(f"[Config] Device(s): {devices}")

    if args.stage in {"run", "all"}:
        if len(devices) >= 2:
            _run_stage1_parallel(
                cfg,
                devices,
                max_samples=args.max_samples,
                dataset_filter=dataset_filter,
                model_filter=model_filter,
            )
        else:
            _run_stage1(
                cfg,
                max_samples=args.max_samples,
                dataset_filter=dataset_filter,
                model_filter=model_filter,
                device=devices[0],
            )

    if args.stage in {"aggregate", "all"}:
        _run_stage1b(cfg, exp_dir)

    if args.stage in {"visualize", "all"}:
        _run_stage2(cfg, exp_dir)


if __name__ == "__main__":
    main()
