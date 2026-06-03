"""Main entry point for the 3-stage segmentation robustness pipeline.

    Stage 1  (run)        – ExperimentEngine: run inference under noise.
    Stage 1b (aggregate)  – StatisticsMerger: aggregate raw CSVs → stats.
    Stage 2  (visualize)  – Viz classes: generate PDF figures & tables.
"""
from __future__ import annotations

import argparse
import copy
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_project_path(path_str: str) -> Path:
    """Resolve relative config paths against the project root, not the cwd."""
    path = Path(str(path_str))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _canonicalize_out_root(cfg: Dict[str, Any]) -> None:
    """Make exp.out_root stable regardless of the shell's current directory."""
    exp_cfg = cfg.setdefault("exp", {})
    out_root = _resolve_project_path(str(exp_cfg.get("out_root", "outputs")))
    exp_cfg["out_root"] = str(out_root)


def _legacy_exp_dir_candidates(exp_dir: Path) -> List[Path]:
    """Return likely legacy experiment directories for previously nested outputs."""
    candidates: List[Path] = []
    try:
        rel = exp_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        return candidates

    legacy = (PROJECT_ROOT / "outputs" / rel).resolve()
    if legacy != exp_dir:
        candidates.append(legacy)
    return candidates


def _has_raw_outputs(exp_dir: Path) -> bool:
    return any(exp_dir.rglob("*_raw.csv"))


def _slugify_for_worker(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    return out.strip("_") or "worker"


def _parallel_mode(cfg: Dict[str, Any]) -> str:
    parallel_cfg = cfg.get("stage1", {}).get("parallel", {})
    if isinstance(parallel_cfg, dict):
        return str(parallel_cfg.get("mode", "model")).strip().lower()
    return "model"


def _collect_model_names(
    cfg: Dict[str, Any],
    model_filter: Optional[List[str]],
) -> List[str]:
    models_cfg = cfg.get("models", [])
    mdl_filter = set(model_filter or [])
    model_names: List[str] = []
    for m in models_cfg:
        name = str(m.get("name"))
        runner = str(m.get("runner", name))
        if mdl_filter and name not in mdl_filter and runner not in mdl_filter:
            continue
        model_names.append(name)
    return model_names


def _collect_prompt_modes(
    cfg: Dict[str, Any],
    model_filter: Optional[List[str]],
) -> List[str]:
    from models.wrappers.prompt_utils import normalize_prompt_mode

    models_cfg = cfg.get("models", [])
    mdl_filter = set(model_filter or [])
    prompt_modes: List[str] = []
    for m in models_cfg:
        name = str(m.get("name"))
        runner = str(m.get("runner", name))
        if mdl_filter and name not in mdl_filter and runner not in mdl_filter:
            continue
        raw = m.get("prompt_modes", m.get("mode", cfg.get("prompt_modes", [])))
        if isinstance(raw, str):
            raw = [raw]
        if not raw:
            raw = ["prompt_bbox"]
        for pm in raw:
            norm = normalize_prompt_mode(pm)
            if norm not in prompt_modes:
                prompt_modes.append(norm)
    return prompt_modes


def _round_robin_assign(items: List[str], devices: List[str]) -> Dict[str, List[str]]:
    assignments: Dict[str, List[str]] = {d: [] for d in devices}
    for i, item in enumerate(items):
        assignments[devices[i % len(devices)]].append(item)
    return {d: vals for d, vals in assignments.items() if vals}


def _experiment_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    exp = cfg.get("exp", cfg.get("experiment", {}))
    return Path(str(exp.get("out_root", "outputs"))) / str(exp.get("name", "experiment"))


def _merge_worker_csvs(
    exp_dir: Path,
    *,
    pattern: str,
    output_name: str,
) -> None:
    parts = sorted(exp_dir.glob(pattern))
    if not parts:
        return

    output_path = exp_dir / output_name
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    seen = set()
    for path in parts:
        try:
            with path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for name in reader.fieldnames or []:
                    if name not in fieldnames:
                        fieldnames.append(name)
                for row in reader:
                    key = tuple(sorted(row.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
        except OSError:
            continue

    if not fieldnames:
        return
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merge_parallel_worker_outputs(cfg: Dict[str, Any]) -> None:
    exp_dir = _experiment_dir_from_cfg(cfg)
    _merge_worker_csvs(
        exp_dir,
        pattern="raw_files_manifest_*.csv",
        output_name="raw_files_manifest.csv",
    )
    _merge_worker_csvs(
        exp_dir,
        pattern="model_complexity_*.csv",
        output_name="model_complexity.csv",
    )


def _clear_parallel_worker_outputs(cfg: Dict[str, Any]) -> None:
    exp_dir = _experiment_dir_from_cfg(cfg)
    for pattern in ("raw_files_manifest_*.csv", "model_complexity_*.csv"):
        for path in exp_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def _find_existing_exp_dir(exp_dir: Path, *, require: str) -> Path:
    """
    Find the best available experiment directory.

    This keeps compatibility with old runs where `out_root: outputs` was
    interpreted from inside the outer `outputs/` folder, producing
    `outputs/outputs/<experiment>`.
    """
    candidates = [exp_dir, *_legacy_exp_dir_candidates(exp_dir)]
    for candidate in candidates:
        if require == "merged" and (candidate / "statistics_merged.csv").exists():
            return candidate
        if require == "raw" and _has_raw_outputs(candidate):
            return candidate
    return exp_dir


def _iter_output_paths(value: Any):
    """Yield Paths from nested dict/list visualization outputs."""
    if isinstance(value, Path):
        yield value
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_output_paths(child)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_output_paths(child)


# ── stage runners ────────────────────────────────────────────────────────

def _run_stage1(
    cfg: Dict[str, Any],
    *,
    max_samples,
    dataset_filter,
    model_filter,
    device: Optional[str] = None,
    prompt_filter: Optional[List[str]] = None,
):
    from core.experiment_engine import ExperimentEngine

    engine = ExperimentEngine(cfg, device=device)
    summary = engine.run(
        max_samples=max_samples,
        dataset_filter=dataset_filter,
        model_filter=model_filter,
        prompt_filter=prompt_filter,
    )
    print(f"[Stage1] Completed on {engine.device}: {summary}")
    return summary


def _gpu_worker(
    cfg: Dict[str, Any],
    device: str,
    model_names: List[str],
    prompt_modes: Optional[List[str]],
    max_samples: Optional[int],
    dataset_filter: Optional[List[str]],
) -> Dict[str, Any]:
    """Worker function executed in a child process for multi-GPU parallel run."""
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg["_parallel_worker_suffix"] = _slugify_for_worker(device)
    return _run_stage1(
        worker_cfg,
        max_samples=max_samples,
        dataset_filter=dataset_filter,
        model_filter=model_names,
        prompt_filter=prompt_modes,
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
    return _run_stage1_parallel_impl(
        cfg,
        devices,
        max_samples=max_samples,
        dataset_filter=dataset_filter,
        model_filter=model_filter,
    )


def _run_stage1_parallel_impl(
    cfg: Dict[str, Any],
    devices: List[str],
    *,
    max_samples: Optional[int],
    dataset_filter: Optional[List[str]],
    model_filter: Optional[List[str]],
):
    """Run Stage 1 across devices using model or prompt-mode partitioning."""
    import multiprocessing as mp

    mode = _parallel_mode(cfg)
    model_names = _collect_model_names(cfg, model_filter)
    if not model_names:
        print("[Stage1] No models matched the filter.")
        return

    if mode in {"prompt", "prompts", "prompt_mode", "prompt_modes"}:
        prompt_modes = _collect_prompt_modes(cfg, model_filter)
        if not prompt_modes:
            print("[Stage1] No prompt modes matched the selected model(s).")
            return

        assignments = _round_robin_assign(prompt_modes, devices)
        if len(assignments) == 1:
            dev, prompts = next(iter(assignments.items()))
            _run_stage1(
                cfg,
                max_samples=max_samples,
                dataset_filter=dataset_filter,
                model_filter=model_filter,
                prompt_filter=prompts,
                device=dev,
            )
            return

        print(
            f"[Stage1] Multi-GPU prompt-mode split: "
            f"{len(prompt_modes)} prompt mode(s), {len(model_names)} model(s), "
            f"{len(assignments)} device(s)"
        )
        for dev, prompts in assignments.items():
            print(f"  {dev}: prompts={prompts}, models={model_filter or model_names}")

        _clear_parallel_worker_outputs(cfg)
        ctx = mp.get_context("spawn")
        processes: List[mp.Process] = []
        for dev, prompts in assignments.items():
            p = ctx.Process(
                target=_gpu_worker,
                args=(cfg, dev, model_filter or model_names, prompts, max_samples, dataset_filter),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"[Stage1] {len(failed)}/{len(processes)} prompt GPU worker(s) failed."
            )
        _merge_parallel_worker_outputs(cfg)
        return

    assignments = _round_robin_assign(model_names, devices)
    if len(assignments) == 1:
        dev, names = next(iter(assignments.items()))
        _run_stage1(
            cfg,
            max_samples=max_samples,
            dataset_filter=dataset_filter,
            model_filter=names,
            device=dev,
        )
        return

    print(f"[Stage1] Multi-GPU model split: {len(model_names)} model(s), {len(assignments)} device(s)")
    for dev, names in assignments.items():
        print(f"  {dev}: models={names}")

    _clear_parallel_worker_outputs(cfg)
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    for dev, names in assignments.items():
        p = ctx.Process(
            target=_gpu_worker,
            args=(cfg, dev, names, None, max_samples, dataset_filter),
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
    _merge_parallel_worker_outputs(cfg)


def _run_stage1_parallel_legacy_unreachable(
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
    from analysis import generate_comprehensive_statistics

    merger = StatisticsMerger()
    source_exp_dir = _find_existing_exp_dir(exp_dir, require="raw")
    summary = merger.run(source_exp_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = source_exp_dir / "statistics_merged.csv"
    canonical_merged_csv = exp_dir / "statistics_merged.csv"
    if merged_csv.exists() and merged_csv != canonical_merged_csv:
        shutil.copy2(merged_csv, canonical_merged_csv)
        stage1b_summary = source_exp_dir / "stage1b_summary.csv"
        if stage1b_summary.exists():
            shutil.copy2(stage1b_summary, exp_dir / "stage1b_summary.csv")
    else:
        canonical_merged_csv = merged_csv

    stats_dir = exp_dir / "statistics"
    generate_comprehensive_statistics(canonical_merged_csv, stats_dir)
    complexity_csv = source_exp_dir / "model_complexity.csv"
    if complexity_csv.exists():
        shutil.copy2(complexity_csv, stats_dir / "model_complexity.csv")

    summary = {
        **summary,
        "source_exp_dir": str(source_exp_dir),
        "output_exp_dir": str(exp_dir),
        "statistics_dir": str(stats_dir),
        "merged_statistics_csv": str(canonical_merged_csv),
    }
    print(f"[Stage1b] Completed: {summary}")
    return summary


def _run_stage2(cfg: Dict[str, Any], exp_dir: Path):
    import os

    backend_env = os.environ.get("MPLBACKEND", "").strip().lower()
    # In headless/script mode, inline backends from notebooks can be invalid.
    if not backend_env or backend_env.startswith("module://"):
        os.environ["MPLBACKEND"] = "Agg"

    from analysis import generate_comprehensive_visualizations

    source_exp_dir = _find_existing_exp_dir(exp_dir, require="merged")
    merged_csv = source_exp_dir / "statistics_merged.csv"
    if not merged_csv.exists():
        raise RuntimeError(
            f"Missing {merged_csv}. Run Stage 1b first."
        )

    output_root = exp_dir / "visualizations"
    artifact_root = exp_dir / "artifacts"
    if not artifact_root.exists():
        artifact_root = source_exp_dir / "artifacts"

    suite_outputs = generate_comprehensive_visualizations(
        merged_csv,
        output_root,
        artifact_root=artifact_root if artifact_root.exists() else None,
    )
    generated_files = sorted({str(p) for p in _iter_output_paths(suite_outputs)})
    outputs: Dict[str, Any] = {
        "input_exp_dir": str(source_exp_dir),
        "output_root": str(output_root),
        "n_generated_files": len(generated_files),
    }

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
        "--model",
        type=str,
        default=None,
        help="Alias of --models for a single model name",
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
    _canonicalize_out_root(cm.cfg)

    # CLI overrides for device selection
    if args.device is not None:
        cm.override_devices([args.device])
    elif args.num_gpus is not None and args.num_gpus >= 1:
        cm.override_devices([f"cuda:{i}" for i in range(args.num_gpus)])

    cfg = cm.cfg
    exp_dir = cm.exp_dir
    devices = cm.devices
    dataset_filter = [x.strip() for x in args.datasets.split(",")] if args.datasets else None
    raw_models = args.models or args.model
    model_filter = [x.strip() for x in raw_models.split(",")] if raw_models else None

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
