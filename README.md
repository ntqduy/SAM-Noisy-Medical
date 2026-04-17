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
- metrics: IoU, Dice, Recall, Precision, F1, HD

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

## Metrics And Aggregation

Statistics are grouped by:

dataset, model, prompt_mode, noise_type, noise_level

For each metric, aggregate stage reports:

- mean
- std
- cv percent
- count of valid values

Additional fields:

- gt_empty_rate
- pred_empty_rate
- n_images
- n_rows

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
