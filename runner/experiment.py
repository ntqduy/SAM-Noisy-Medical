"""
Main experiment runner for AIO25 NoisySAM benchmark.

Extended features:
  - Noise intensity tracking with full metadata
  - Prediction caching for efficient re-runs
  - Uncertainty metrics (confidence, entropy)
  - Debug/subset selection options
  - Report-only mode from cached results
"""
import warnings
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from runner.io_utils import (
    ensure_dir, save_json, save_yaml_config, load_json, get_device,
    env_seed_everything, PredictionCache, get_command_string
)
from runner.protocols import build_protocol_cases, ProtocolCase
from runner.aggregate import aggregate_results, compute_stability
from runner.config_schema import validate_config

from datasets.registry import build_dataset
from model.registry import build_model_runner
from noises.registry import build_noise
from noises.base import NoiseResult

from viz.grids import save_preview_pdf, save_noise_gallery
from viz.plots import (
    plot_metric_vs_level, plot_ofat_sensitivity, plot_grid_heatmap,
    plot_global_sensitivity, plot_summary_heatmap
)
from viz.failure_cases import export_failure_cases
from reports.pdf_builder import build_report_pdf


def compute_uncertainty_metrics(
    pred: np.ndarray,
    prob_map: Optional[np.ndarray],
    gt: np.ndarray
) -> Dict[str, float]:
    """
    Compute uncertainty metrics from prediction.
    
    Args:
        pred: Binary prediction mask
        prob_map: Probability/confidence map (optional)
        gt: Ground truth mask
        
    Returns:
        Dict with uncertainty metrics
    """
    metrics = {}
    
    if prob_map is not None and prob_map.size > 0:
        # Ensure prob_map is in [0, 1]
        if prob_map.max() > 1.0:
            prob_map = prob_map / 255.0
        
        # Mean confidence
        metrics["mean_confidence"] = float(np.mean(prob_map))
        
        # Entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-7
        p = np.clip(prob_map, eps, 1 - eps)
        entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        metrics["mean_entropy"] = float(np.mean(entropy))
        
        # Boundary entropy (thin band around prediction boundary)
        try:
            from scipy.ndimage import binary_dilation, binary_erosion
            pred_bool = pred.astype(bool)
            boundary = binary_dilation(pred_bool, iterations=2) ^ binary_erosion(pred_bool, iterations=2)
            if boundary.sum() > 0:
                metrics["boundary_entropy"] = float(np.mean(entropy[boundary]))
            else:
                metrics["boundary_entropy"] = 0.0
        except ImportError:
            metrics["boundary_entropy"] = None
    else:
        # Fallback: distance-to-boundary proxy for uncertainty
        try:
            from scipy.ndimage import distance_transform_edt
            pred_bool = pred.astype(bool)
            if pred_bool.sum() > 0 and pred_bool.sum() < pred_bool.size:
                dt_inside = distance_transform_edt(pred_bool)
                dt_outside = distance_transform_edt(~pred_bool)
                dt_boundary = np.minimum(dt_inside, dt_outside)
                # Normalize: closer to boundary = higher uncertainty proxy
                max_dist = max(dt_boundary.max(), 1.0)
                uncertainty_proxy = 1.0 - (dt_boundary / max_dist)
                metrics["mean_confidence_proxy"] = float(1.0 - np.mean(uncertainty_proxy))
                metrics["boundary_uncertainty_proxy"] = float(np.mean(uncertainty_proxy[dt_boundary < 5]))
            else:
                metrics["mean_confidence_proxy"] = 1.0 if pred_bool.sum() > 0 else 0.0
                metrics["boundary_uncertainty_proxy"] = 0.0
        except ImportError:
            metrics["mean_confidence_proxy"] = None
            metrics["boundary_uncertainty_proxy"] = None
    
    return metrics


def filter_by_debug_config(items: List, cfg: dict, item_type: str) -> List:
    """
    Filter items based on debug configuration.
    
    Args:
        items: List of items to filter
        cfg: Config with debug settings
        item_type: Type of items ('samples', 'models', 'modes')
        
    Returns:
        Filtered list
    """
    debug = cfg.get("debug", {})
    
    if item_type == "samples":
        sample_ids = debug.get("sample_ids")
        max_samples = debug.get("max_samples")
        
        if sample_ids:
            items = [i for i in items if i in sample_ids]
        if max_samples and max_samples > 0:
            items = items[:max_samples]
    
    return items


def run_experiment(cfg: Dict[str, Any]):
    """
    Main experiment runner for SAM benchmark under noisy conditions.
    
    Extended for AIO25 NoisySAM:
      - Full noise metadata tracking
      - Prediction caching
      - Uncertainty metrics
      - Debug subset selection
      - Report-only mode
    """
    # Validate and fill defaults
    cfg = validate_config(cfg)
    
    seed = int(cfg.get("seed", 42))
    env_seed_everything(seed)

    phase = int(cfg.get("phase", 1))
    dry_run = bool(cfg.get("dry_run", False))
    limit_n = int(cfg.get("limit_n", 0) or 0)

    exp_name = cfg["exp"]["name"]
    
    # Cache configuration
    cache_cfg = cfg.get("cache", {})
    use_cache = cache_cfg.get("enabled", False)
    overwrite_cache = cache_cfg.get("clear_on_start", False)
    only_report = cfg.get("only_report", False)
    
    # Debug configuration
    debug_cfg = cfg.get("debug", {})
    max_samples = debug_cfg.get("max_samples")
    sample_ids_filter = debug_cfg.get("sample_ids")
    models_filter = debug_cfg.get("models")
    modes_filter = debug_cfg.get("modes")
    
    # Noise configuration
    noise_cfg = cfg.get("noise_config", {})
    compute_distortion = noise_cfg.get("compute_distortion", True)
    
    # Normalize output directory
    exp_dir = ensure_dir(Path(cfg["exp"].get("out_root", "outputs")) / exp_name)

    figures_dir = ensure_dir(exp_dir / "figures")
    pred_dir = ensure_dir(exp_dir / "pred_masks")
    meta_dir = ensure_dir(exp_dir / "meta")
    
    # Handle cache_dir - use default "cache" if None or not specified
    cache_subdir = cache_cfg.get("cache_dir") or "cache"
    cache_dir = ensure_dir(exp_dir / cache_subdir)

    device = get_device(cfg.get("device", "cpu"))
    
    # Initialize prediction cache
    pred_cache = PredictionCache(cache_dir, use_cache=use_cache, overwrite=overwrite_cache)

    # Save config snapshot and command
    save_json(meta_dir / "config_snapshot.json", cfg)
    save_yaml_config(meta_dir / "config_snapshot.yaml", cfg)
    
    try:
        import sys
        cmd_str = " ".join(sys.argv)
        with open(meta_dir / "command.txt", "w") as f:
            f.write(cmd_str)
    except:
        pass

    datasets_cfg = cfg.get("datasets", [])
    models_cfg = cfg.get("models", [])
    outputs_cfg = cfg.get("outputs", {})

    # Check for report-only mode
    results_csv = exp_dir / "results.csv"
    if only_report and results_csv.exists():
        print("[INFO] Report-only mode: regenerating report from existing results.csv")
        df = pd.read_csv(results_csv)
        _generate_outputs(df, cfg, exp_dir, figures_dir, outputs_cfg)
        return

    # Build protocol cases
    protocol_cases = build_protocol_cases(cfg)

    # Handle dry_run - print cases and exit early
    if dry_run:
        print(f"\n[DRY RUN] Would run {len(protocol_cases)} protocol cases:")
        for i, case in enumerate(protocol_cases[:20]):  # Show first 20
            print(f"  {i+1}. {case.protocol}:{case.noise_name}:{case.level} (p={case.p}, seed={case.noise_seed})")
        if len(protocol_cases) > 20:
            print(f"  ... and {len(protocol_cases) - 20} more cases")
        print(f"\n[DRY RUN] Datasets: {[d['name'] for d in datasets_cfg]}")
        print(f"[DRY RUN] Models: {[m['name'] for m in models_cfg]}")
        print("[DRY RUN] Exiting without running inference.")
        return

    all_rows: List[Dict[str, Any]] = []

    # Build datasets
    datasets = []
    for dcfg in datasets_cfg:
        ds = build_dataset(dcfg)
        datasets.append((dcfg["name"], ds))

    # Build model runners with filtering
    model_runners = []
    for mcfg in models_cfg:
        model_name = mcfg["name"]
        
        # Apply model filter
        if models_filter and model_name not in models_filter:
            continue
            
        runner_key = mcfg["runner"]
        modes = list(mcfg.get("mode", ["prompt_bbox"]))
        
        # Apply mode filter
        if modes_filter:
            modes = [m for m in modes if m in modes_filter]
        
        weights = list(mcfg.get("weights", []))
        for w in weights:
            ckpt = w.get("checkpoint", "")
            if not ckpt or not Path(ckpt).exists():
                warnings.warn(f"[WARN] Missing checkpoint for {model_name}/{w.get('id','?')}: {ckpt} -> skip")
                continue
            for mode in modes:
                try:
                    runner = build_model_runner(runner_key, w, mode, device=device)
                    model_runners.append((model_name, w, mode, runner))
                except Exception as e:
                    warnings.warn(f"[WARN] Failed to build {model_name}/{mode}: {e}")

    if len(model_runners) == 0:
        raise RuntimeError("No valid models/weights found (all checkpoints missing?).")

    # Main experiment loop
    for dataset_name, ds in datasets:
        n = len(ds)
        idxs = list(range(n))
        
        # Apply sample filters
        if sample_ids_filter:
            idxs = [i for i in idxs if ds[i]["id"] in sample_ids_filter]
        if max_samples and max_samples > 0:
            idxs = idxs[:max_samples]
        if limit_n > 0:
            idxs = idxs[:min(limit_n, len(idxs))]

        for model_name, weight_cfg, mode, model in model_runners:
            weight_id = weight_cfg["id"]
            
            # For stability: store clean prediction per id (from P0)
            clean_pred_cache: Dict[str, np.ndarray] = {}
            clean_img_cache: Dict[str, np.ndarray] = {}

            for case in protocol_cases:
                noise_name = case.noise_name
                protocol = case.protocol
                level = case.level
                noise_seed = case.noise_seed

                # Build noise with full metadata tracking
                noise_instance = build_noise(
                    name=noise_name,
                    p=case.p,
                    params=case.params,
                    seed=noise_seed,
                    level=level,
                    protocol=protocol,
                    compute_distortion=compute_distortion
                )

                desc = f"{dataset_name} | {model_name}/{weight_id} | {mode} | {protocol}:{noise_name}:{level}"
                
                for i in tqdm(idxs, desc=desc):
                    sample = ds[i]
                    sid = sample["id"]
                    img = sample["image"]      # uint8 HxW
                    gt = sample["gt_mask"]     # uint8 HxW {0,1}
                    meta = sample.get("meta", {})

                    # Store clean image for distortion metrics
                    if protocol == "P0" or noise_name == "clean":
                        clean_img_cache[sid] = img.copy()

                    # Apply noise and get metadata
                    if noise_instance is None:
                        img_noisy = img.copy()
                        noise_meta = {
                            "noise_type": "clean",
                            "protocol": protocol,
                            "level": level,
                            "p": 0.0,
                            "severity_scalar": 0.0,
                            "intensity_scalar": 0.0,
                            "severity_params": {},
                            "noise_seed": noise_seed,
                            "applied": False
                        }
                    else:
                        noise_result = noise_instance(img, return_meta=True)
                        img_noisy = noise_result.noisy_image
                        noise_meta = noise_result.meta

                    # Check cache
                    cache_key = pred_cache.get_key(
                        dataset=dataset_name,
                        sample_id=sid,
                        model_name=model_name,
                        weight_id=weight_id,
                        mode=mode,
                        noise_type=noise_name,
                        protocol=protocol,
                        level=level,
                        p=case.p,
                        severity_params=case.params,
                        noise_seed=noise_seed
                    )

                    cached = pred_cache.load(cache_key) if use_cache else None
                    
                    if cached is not None:
                        pred = cached["pred_mask"]
                        prob_map = cached.get("prob_map")
                        extra = cached.get("extra", {})
                    else:
                        # Run inference
                        pred, extra = model.predict(img_noisy, gt_mask=gt, meta=meta)
                        prob_map = extra.get("prob_map") if isinstance(extra, dict) else None
                        
                        # Save to cache
                        pred_cache.save(
                            cache_key=cache_key,
                            pred_mask=pred,
                            prob_map=prob_map,
                            extra=extra if isinstance(extra, dict) else {},
                            metadata={"noise_meta": noise_meta}
                        )

                    # Build result row
                    row = {
                        "phase": phase,
                        "dataset": dataset_name,
                        "id": sid,
                        "model": model_name,
                        "weight": weight_id,
                        "mode": mode,
                        "protocol": protocol,
                        "noise": noise_name,
                        "level": level,
                        "noise_seed": noise_seed,
                        "img_path": meta.get("img_path", ""),
                        "mask_path": meta.get("mask_path", ""),
                        # Noise metadata
                        "p": noise_meta.get("p", case.p),
                        "severity_scalar": noise_meta.get("severity_scalar", 0.0),
                        "intensity_scalar": noise_meta.get("intensity_scalar", case.intensity_scalar),
                        "severity_params": json.dumps(noise_meta.get("severity_params", case.params)),
                        "noise_applied": noise_meta.get("applied", True),
                    }

                    # Distortion metrics (PSNR, SSIM)
                    if "psnr" in noise_meta:
                        row["psnr"] = noise_meta["psnr"]
                    if "ssim" in noise_meta:
                        row["ssim"] = noise_meta["ssim"]

                    # Segmentation metrics
                    from metrics.seg import dice, iou, hd95
                    row["dice"] = float(dice(pred, gt))
                    row["iou"] = float(iou(pred, gt))
                    hd95_val = hd95(pred, gt)
                    row["hd95"] = float(hd95_val) if hd95_val is not None else None

                    # Model confidence if available
                    if isinstance(extra, dict) and "pred_iou_score" in extra:
                        row["pred_iou_score"] = float(extra["pred_iou_score"])

                    # Uncertainty metrics
                    uncertainty = compute_uncertainty_metrics(pred, prob_map, gt)
                    row.update(uncertainty)

                    # Mask consistency with clean prediction
                    if protocol == "P0" and noise_name == "clean":
                        clean_pred_cache[sid] = pred.copy()
                        row["mask_consistency_iou"] = 1.0
                    else:
                        if sid in clean_pred_cache:
                            row["mask_consistency_iou"] = float(iou(pred, clean_pred_cache[sid]))
                        else:
                            row["mask_consistency_iou"] = None

                    all_rows.append(row)

                    # Save pred mask optionally
                    if outputs_cfg.get("save_pred_masks", True):
                        out_sub = ensure_dir(
                            pred_dir / dataset_name / model_name / weight_id / mode / 
                            protocol / noise_name / str(level) / f"seed{noise_seed}"
                        )
                        from PIL import Image
                        Image.fromarray((pred.astype("uint8") * 255)).save(out_sub / f"{sid}.png")

    # Create results DataFrame
    df = pd.DataFrame(all_rows)
    df.to_csv(results_csv, index=False)

    # Generate outputs
    _generate_outputs(df, cfg, exp_dir, figures_dir, outputs_cfg)


def _generate_outputs(
    df: pd.DataFrame,
    cfg: Dict,
    exp_dir: Path,
    figures_dir: Path,
    outputs_cfg: Dict
):
    """
    Generate all output files and visualizations.
    
    Args:
        df: Results DataFrame
        cfg: Configuration dictionary
        exp_dir: Experiment output directory
        figures_dir: Figures output directory
        outputs_cfg: Output configuration
    """
    from runner.aggregate import aggregate_results, compute_stability_extended
    
    # Aggregate results
    agg = aggregate_results(df)
    aggregate_csv = exp_dir / "aggregate.csv"
    agg.to_csv(aggregate_csv, index=False)

    # Compute extended stability metrics
    stability = compute_stability_extended(df, cfg)
    stability_csv = exp_dir / "stability.csv"
    stability.to_csv(stability_csv, index=False)

    # Generate plots
    plot_paths = []
    if len(df) > 0:
        # Standard plots
        plot_paths += plot_metric_vs_level(df, figures_dir, protocols=["P1"], metrics=["dice", "iou"])
        plot_paths += plot_ofat_sensitivity(df, figures_dir, metrics=["dice"], protocols=["P2a", "P2b"])
        plot_paths += plot_grid_heatmap(df, figures_dir)
        
        # Global comparative plots
        plot_paths += plot_global_sensitivity(df, figures_dir, metric="dice")
        plot_paths += plot_summary_heatmap(df, figures_dir, stability)

    # Noise gallery
    try:
        gallery_paths = save_noise_gallery(
            df=df,
            cfg=cfg,
            out_dir=figures_dir / "noise_gallery",
            num_samples=int(outputs_cfg.get("noise_gallery_samples", 3))
        )
        plot_paths += gallery_paths
    except Exception as e:
        warnings.warn(f"[WARN] Failed to generate noise gallery: {e}")

    # Preview PDF
    preview_pdf = exp_dir / "preview.pdf"
    try:
        save_preview_pdf(
            df=df,
            cfg=cfg,
            out_pdf=preview_pdf,
            num_samples=int(outputs_cfg.get("num_preview_samples", 8)),
            levels=list(outputs_cfg.get("preview_levels", ["L0", "L2", "L4"])),
        )
    except Exception as e:
        warnings.warn(f"[WARN] Failed to generate preview PDF: {e}")

    # Failure cases
    failure_dir = ensure_dir(exp_dir / "failure_cases")
    try:
        failure_imgs = export_failure_cases(
            df, cfg, failure_dir, 
            top_k=int(outputs_cfg.get("num_failure_cases", 8))
        )
    except Exception as e:
        warnings.warn(f"[WARN] Failed to export failure cases: {e}")
        failure_imgs = []

    # Summary JSON
    summary = {
        "exp_name": cfg["exp"]["name"],
        "phase": cfg.get("phase", 1),
        "device": cfg.get("device", "cpu"),
        "n_rows": int(len(df)),
        "n_samples": int(df["id"].nunique()) if len(df) > 0 else 0,
        "n_models": int(df[["model", "weight"]].drop_duplicates().shape[0]) if len(df) > 0 else 0,
        "results_csv": str(exp_dir / "results.csv"),
        "aggregate_csv": str(aggregate_csv),
        "stability_csv": str(stability_csv),
        "preview_pdf": str(preview_pdf),
        "figures_dir": str(figures_dir),
        "cache_stats": PredictionCache(exp_dir / "cache", use_cache=False).get_stats() if (exp_dir / "cache").exists() else {}
    }
    save_json(exp_dir / "summary.json", summary)

    # Build final report PDF
    try:
        report_pdf = build_report_pdf(
            df=df,
            agg_df=agg,
            stability_df=stability,
            cfg=cfg,
            exp_dir=exp_dir,
            figure_paths=plot_paths,
            failure_paths=failure_imgs,
        )
    except Exception as e:
        warnings.warn(f"[WARN] Failed to build report PDF: {e}")
        report_pdf = None

    print("\n[DONE]")
    print(f"- results: {exp_dir / 'results.csv'}")
    print(f"- aggregate: {aggregate_csv}")
    print(f"- stability: {stability_csv}")
    print(f"- preview: {preview_pdf}")
    if report_pdf:
        print(f"- report: {report_pdf}")
    print(f"- summary: {exp_dir / 'summary.json'}")
