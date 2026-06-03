# SAM-Noisy-Medical

Benchmark pipeline for medical image segmentation robustness under controlled image noise.

## Overview

This benchmark evaluates 9 SAM-based models and SAM variants under a unified robustness protocol.

- 9 models: SAM, SAM2, SAM3, MedSAM, MedicoSAM, MedSAM2, MedSAM3, SAM-Med2D, UltraSAM
- 12 noise types
- 10 severity levels from L0 (clean) to L9 (most severe), with increasing corruption strength
- 3 prompt types: prompt_point, prompt_bbox, prompt_point_box
- 3 primary datasets: BUSI, CAMUS, DDTI

The goal is to measure segmentation performance degradation consistently across models, noise conditions, prompt modes, and key medical datasets.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use UltraSAM, install extra dependencies:

```bash
pip install -r requirements_ultra.txt
```

Run full pipeline:

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

## Run By Stage

```bash
python main.py --config configs/full_benchmark.yaml --stage run
python main.py --config configs/full_benchmark.yaml --stage aggregate
python main.py --config configs/full_benchmark.yaml --stage visualize
```

## Common CLI Options

Limit number of samples:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 50
```

Run selected datasets:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --datasets Montgomery,BUSI
```

Run selected models:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --models SAM2,MedSAM3
```

Single model alias:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --model SAM2
```

Device override:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --device cpu
python main.py --config configs/full_benchmark.yaml --stage run --device cuda:1
```

Multi-GPU model split:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 2
```

Notes:

- datasets and models are comma-separated strings.
- run and all execute inference; aggregate and visualize consume saved outputs.

## Default Full Benchmark Setup

Main config: configs/full_benchmark.yaml

- models: 9 wrappers
  - SAM, SAM2, SAM3, MedSAM, MedicoSAM, MedSAM2, MedSAM3, SAM-Med2D, UltraSAM
- dataset entries: 6 entries in config
  - Montgomery, BUSI, CAMUS (nift), CAMUS (2D slice), TN3K, TG3K, DDTI
- prompt modes: prompt_point, prompt_bbox, prompt_point_box
- noise types (enabled):
  - gaussian, speckle, salt_pepper, motion_blur, jpeg, pixelation,
    low_brightness, high_brightness, low_contrast, high_contrast,
    rician, poisson
- levels: L0 to L9
- metrics: IoU, Dice, Recall, Precision, F1, HD, HD_px, HD95_px, HD_mm, HD95_mm, inference_time_ms, FPS

To rerun the old full setup with the new metrics:

```bash
python main.py --config configs/full_benchmark.yaml --stage run
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

## Output Structure

Typical files under outputs/full_benchmark/:

- statistics_merged.csv
- stage1b_summary.csv
- statistics/
  - overall/model/mode/noise/level summaries
  - per-metric matrices
  - robustness analysis
- visualizations/
  - line plots by level
  - model-vs-noise and model-vs-level heatmaps
  - model ranking and noise difficulty ranking
  - noise gallery and optional segmentation gallery

Raw files are written per dataset/model/prompt combination as *_raw.csv and *_stats.csv.

When `prompt_variants.enabled: true`, variant raw files are separated under:

```text
outputs/full_benchmark/prompt_variants/<prompt_mode>/<prompt_variant>/<dataset>/<model>/
```

The main benchmark layout remains:

```text
outputs/full_benchmark/<dataset>/<model>/
```

## Metrics And Aggregation

Main prompt mode statistics are grouped by:

dataset, model, prompt_mode, noise_type, noise_level

Prompt variant statistics are grouped by:

dataset, model, prompt_mode, prompt_variant, noise_type, noise_level

For each metric, aggregate stage reports:

- mean
- std
- cv percent
- count of valid values

Additional fields:

- gt_empty_rate
- pred_empty_rate
- failure_rate_dice_lt_0_5
- failure_rate_dice_lt_0_7
- bbox_center_inside_mask_percentage
- n_images
- n_rows

Important: do not mix `main_prompt_mode_benchmark` and `prompt_variant_benchmark` summaries in a paper. The former compares prompt modes; the latter compares variants inside one prompt mode.

## New Metrics

Enable the metric additions in config:

```yaml
metrics:
  add_hd95: true
  add_physical_distance: true
  spacing_source: "sample_meta"
  fallback_spacing: null
  keep_legacy_hd: true

performance:
  log_inference_time: true
  log_fps: true
```

`HD` is preserved as the legacy pixel Hausdorff distance. `HD_px` is an alias of `HD`, and `HD95_px` is the 95th percentile surface distance in pixels. `HD_mm` and `HD95_mm` use `sample["meta"]["spacing"]` as `(spacing_y, spacing_x)`; when spacing is missing they are `nan`.

## Prompt Variants

Keep `prompt_modes` unchanged. Turn on variants only when you want to compare bbox or point choices within a prompt mode:

```yaml
prompt_variants:
  enabled: true
```

`prompt_bbox` variants include `bbox_gt_5`, `bbox_expand_10`, `bbox_expand_20`, `bbox_shift_10`, and `bbox_shrink_10`. `prompt_point` variants include `point_center_bbox`, `point_centroid`, and `point_random_inside`.

`point_center_bbox` uses the center of the GT-derived bbox. `point_centroid` uses the GT mask centroid, which can differ from the bbox center and can land differently for asymmetric masks.

Variant raw filenames include the variant:

```text
<runner>_<dataset>_<prompt_mode>_<prompt_variant>_raw.csv
```

## Noisy Image Reuse And Saving

Stage 1 can reuse already saved noisy images:

```yaml
stage1:
  reuse_existing_noisy_images: true
  existing_noisy_root: "/data2/Medical/StrokeCT-outputs/full_benchmark/artifacts/_shared"
  fallback_generate_noise_if_missing: true
```

Reuse is logged in raw CSV with `noisy_image_source=external_cache`; generated images use `generated`, and internal `.npy` cache hits use `internal_cache`.

By default in `configs/full_benchmark.yaml`, GT masks, original images, and noisy images are not copied again:

```yaml
stage1:
  save_pred_masks: true
  save_gt_masks: false
  save_original_images: false
  save_noisy_images: false
```

Pred masks are saved only when `save_pred_masks: true`, under:

```text
outputs/full_benchmark/pred_masks/<dataset>/<model>/<prompt_mode>/<prompt_variant>/<noise_type>/<level>/<seed>/<image_id>.png
```

Use pred masks for metric recomputation or debugging; otherwise turn them off to save disk.

## Parallel Runs

Single device:

```yaml
device: "cuda:0"
```

Explicit multi-GPU model split:

```yaml
devices: ["cuda:0", "cuda:1"]
```

Safe preprocessing/cache settings:

```yaml
stage1:
  parallel:
    enabled: true
    num_workers: 4
    mode: "prompt_mode"
    preserve_order: true
```

With `mode: "prompt_mode"`, multiple GPUs split prompt jobs. When `prompt_variants.enabled: false`, each prompt mode is one job. When `prompt_variants.enabled: true`, only the configured variants are jobs, and no default prompt variants are added.

```text
--num_gpus 1: cuda:0 runs all jobs sequentially
--num_gpus 2: jobs are assigned round-robin to cuda:0 and cuda:1
--num_gpus 3: jobs are assigned round-robin to cuda:0, cuda:1, and cuda:2
```

Example:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 3 --models SAM2
```

This loads the selected model once per GPU/prompt worker, so it reduces wall-clock time but uses more total VRAM. The per-row `FPS` is still single-predict latency-derived FPS, not total multi-GPU throughput.

To go back to the older model-split behavior:

```yaml
stage1:
  parallel:
    mode: "model"
```

## Legacy Output Compatibility

If you have old runs in nested paths like outputs/outputs/full_benchmark, current aggregate and visualize stages can auto-detect and reuse those files.

## Troubleshooting

No raw CSV files found:

```bash
python main.py --config configs/full_benchmark.yaml --stage run
```

Missing statistics_merged.csv:

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

CUDA issues:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --device cpu
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Out of memory:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 20
python main.py --config configs/full_benchmark.yaml --stage run --datasets Montgomery
python main.py --config configs/full_benchmark.yaml --stage run --models SAM2
```

If some datasets or checkpoints are missing locally, filter with CLI options or edit configs/full_benchmark.yaml.
