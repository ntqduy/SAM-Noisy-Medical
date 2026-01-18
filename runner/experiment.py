import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from runner.io_utils import ensure_dir, save_json, get_device, env_seed_everything
from runner.protocols import build_protocol_cases, ProtocolCase
from runner.aggregate import aggregate_results, compute_stability

from datasets.registry import build_dataset
from model.registry import build_model_runner
from noises.registry import build_noise
from viz.grids import save_preview_pdf
from viz.plots import plot_metric_vs_level, plot_ofat_sensitivity, plot_grid_heatmap
from viz.failure_cases import export_failure_cases
from reports.pdf_builder import build_report_pdf


def run_experiment(cfg: Dict[str, Any]):
    """
    Main experiment runner for SAM benchmark under noisy conditions.
    
    Handles:
      - Dataset loading
      - Model initialization
      - Protocol execution (P0, P1, P2a, P2b, P3)
      - Noise injection
      - Metric computation
      - Visualization generation
      - Report building
    """
    seed = int(cfg.get("seed", 42))
    env_seed_everything(seed)

    phase = int(cfg.get("phase", 1))
    dry_run = bool(cfg.get("dry_run", False))
    limit_n = int(cfg.get("limit_n", 0) or 0)

    exp_name = cfg["exp"]["name"]
    
    # Normalize output directory
    exp_dir = ensure_dir(Path(cfg["exp"].get("out_root", "outputs")) / exp_name)

    figures_dir = ensure_dir(exp_dir / "figures")
    pred_dir = ensure_dir(exp_dir / "pred_masks")
    meta_dir = ensure_dir(exp_dir / "meta")

    device = get_device(cfg.get("device", "cpu"))

    # save config snapshot
    save_json(meta_dir / "config_snapshot.json", cfg)

    datasets_cfg = cfg.get("datasets", [])
    models_cfg = cfg.get("models", [])
    outputs_cfg = cfg.get("outputs", {})

    protocol_cases = build_protocol_cases(cfg)

    all_rows: List[Dict[str, Any]] = []

    # build datasets
    datasets = []
    for dcfg in datasets_cfg:
        ds = build_dataset(dcfg)
        datasets.append((dcfg["name"], ds))

    # iterate models x weights x modes
    model_runners = []
    for mcfg in models_cfg:
        model_name = mcfg["name"]
        runner_key = mcfg["runner"]
        modes = list(mcfg.get("mode", ["prompt_bbox"]))
        weights = list(mcfg.get("weights", []))
        for w in weights:
            ckpt = w.get("checkpoint", "")
            if not ckpt or not Path(ckpt).exists():
                warnings.warn(f"[WARN] Missing checkpoint for {model_name}/{w.get('id','?')}: {ckpt} -> skip")
                continue
            for mode in modes:
                model_runners.append((model_name, w, mode, build_model_runner(runner_key, w, mode, device=device)))

    if len(model_runners) == 0:
        raise RuntimeError("No valid models/weights found (all checkpoints missing?).")

    # main loop
    for dataset_name, ds in datasets:
        n = len(ds)
        idxs = list(range(n))
        if limit_n > 0:
            idxs = idxs[: min(limit_n, n)]

        for model_name, weight_cfg, mode, model in model_runners:
            # For stability: store clean prediction per id (from P0)
            clean_pred_cache: Dict[str, Any] = {}

            for case in protocol_cases:
                noise_name = case.noise_name
                protocol = case.protocol
                level = case.level

                if dry_run and protocol not in ("P0", "P1"):
                    continue

                noise = build_noise(noise_name, p=case.p, params=case.params)

                for i in tqdm(idxs, desc=f"{dataset_name} | {model_name}/{weight_cfg['id']} | {mode} | {protocol}:{noise_name}:{level}"):
                    sample = ds[i]  # {id, image, gt_mask, meta}
                    sid = sample["id"]
                    img = sample["image"]      # uint8 HxW
                    gt = sample["gt_mask"]     # uint8 HxW {0,1}
                    meta = sample.get("meta", {})

                    img_noisy = img if noise is None else noise(img)

                    pred, extra = model.predict(img_noisy, gt_mask=gt, meta=meta)

                    row = {
                        "phase": phase,
                        "dataset": dataset_name,
                        "id": sid,
                        "model": model_name,
                        "weight": weight_cfg["id"],
                        "mode": mode,
                        "protocol": protocol,
                        "noise": noise_name,
                        "level": level,
                        "img_path": meta.get("img_path", ""),
                        "mask_path": meta.get("mask_path", ""),
                    }

                    # metrics
                    from metrics.seg import dice, iou, hd95
                    row["dice"] = float(dice(pred, gt))
                    row["iou"] = float(iou(pred, gt))
                    row["hd95"] = float(hd95(pred, gt)) if hd95(pred, gt) is not None else None

                    # optional confidence proxy
                    if isinstance(extra, dict) and "pred_iou_score" in extra:
                        row["pred_iou_score"] = extra["pred_iou_score"]

                    # stability: MaskConsistency = IoU(mask_L0, mask_Lk)
                    if protocol == "P0" and noise_name == "clean":
                        clean_pred_cache[sid] = pred.copy()
                    else:
                        if sid in clean_pred_cache:
                            row["mask_consistency_iou"] = float(iou(pred, clean_pred_cache[sid]))
                        else:
                            row["mask_consistency_iou"] = None

                    all_rows.append(row)

                    # save pred mask optionally
                    if outputs_cfg.get("save_pred_masks", True):
                        out_sub = ensure_dir(pred_dir / dataset_name / model_name / weight_cfg["id"] / mode / protocol / noise_name / str(level))
                        from PIL import Image
                        Image.fromarray((pred.astype("uint8") * 255)).save(out_sub / f"{sid}.png")

    df = pd.DataFrame(all_rows)
    results_csv = exp_dir / "results.csv"
    df.to_csv(results_csv, index=False)

    agg = aggregate_results(df)
    aggregate_csv = exp_dir / "aggregate.csv"
    agg.to_csv(aggregate_csv, index=False)

    stability = compute_stability(df)
    stability_csv = exp_dir / "stability.csv"
    stability.to_csv(stability_csv, index=False)

    # plots
    plot_paths = []
    if len(df) > 0:
        plot_paths += plot_metric_vs_level(df, figures_dir, protocols=["P1"], metrics=["dice", "iou"])
        plot_paths += plot_ofat_sensitivity(df, figures_dir, metrics=["dice"], protocols=["P2a", "P2b"])
        plot_paths += plot_grid_heatmap(df, figures_dir)

    # preview.pdf (clean vs L2 vs L4)
    preview_pdf = exp_dir / "preview.pdf"
    save_preview_pdf(
        df=df,
        cfg=cfg,
        out_pdf=preview_pdf,
        num_samples=int(outputs_cfg.get("num_preview_samples", 8)),
        levels=list(outputs_cfg.get("preview_levels", ["L0", "L2", "L4"])),
    )

    # failure cases
    failure_dir = ensure_dir(exp_dir / "failure_cases")
    failure_imgs = export_failure_cases(df, cfg, failure_dir, top_k=int(outputs_cfg.get("num_failure_cases", 8)))

    # summary.json
    summary = {
        "exp_name": cfg["exp"]["name"],
        "phase": phase,
        "device": device,
        "n_rows": int(len(df)),
        "results_csv": str(results_csv),
        "aggregate_csv": str(aggregate_csv),
        "stability_csv": str(stability_csv),
        "preview_pdf": str(preview_pdf),
        "figures_dir": str(figures_dir),
    }
    save_json(exp_dir / "summary.json", summary)

    # report.pdf
    report_pdf = build_report_pdf(
        df=df,
        agg_df=agg,
        stability_df=stability,
        cfg=cfg,
        exp_dir=exp_dir,
        figure_paths=plot_paths,
        failure_paths=failure_imgs,
    )

    print("\n[DONE]")
    print(f"- preview: {preview_pdf}")
    print(f"- report : {report_pdf}")
    print(f"- results: {results_csv}")
    print(f"- agg    : {aggregate_csv}")
    print(f"- summary: {exp_dir / 'summary.json'}")
