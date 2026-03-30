# SAM-Noisy-Medical

Robustness benchmark pipeline for medical image segmentation under controlled noise.

The project is organized as a 3-stage pipeline:

1. `run`: run model inference on noisy images and write per-image raw CSVs
2. `aggregate`: convert raw CSVs into merged statistics and summary tables
3. `visualize`: generate publication-style PDF figures from merged statistics

The canonical experiment output directory is:

```text
outputs/<experiment_name>/
```

For `configs/full_benchmark.yaml`, that is:

```text
outputs/full_benchmark/
```

The current code also auto-detects older nested runs under `outputs/outputs/...` when you run `aggregate` or `visualize`.

---

## What The Current Full Benchmark Config Enables

`configs/full_benchmark.yaml` currently defines:

- 9 model wrappers: `SAM`, `SAM2`, `SAM3`, `MedSAM`, `MedicoSAM`, `MedSAM2`, `MedSAM3`, `SAM-Med2D`, `UltraSAM`
- 6 datasets in config: `Montgomery`, `BUSI`, `CAMUS`, `TN3K`, `TG3K`, `DDTI`
- 11 enabled noise types in the full benchmark config:
  `gaussian`, `speckle`, `salt_pepper`, `motion_blur`, `jpeg`, `pixelation`,
  `low_brightness`, `high_brightness`, `low_contrast`, `high_contrast`, `rician`
- 10 noise levels: `L0` to `L9`
- 3 prompt modes in the full benchmark config:
  `prompt_point`, `prompt_bbox`, `prompt_point_box`
- 6 metrics: `IoU`, `Dice`, `Recall`, `Precision`, `F1`, `HD`

Notes:

- The noise registry supports more noises than the full benchmark config currently enables, for example `poisson`, `bias_field`, `uniform`, `defocus_blur`, `gridmask`, and others.
- Prompt utilities also support `prompt_multi_point` and `autogen`, but `configs/full_benchmark.yaml` uses the 3 prompt modes above.
- Outputs are only created for datasets and models that are actually run.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you want to run `UltraSAM`, you will typically also need extra OpenMMLab dependencies:

```bash
pip install -r requirements_ultra.txt
```

### 2. Run the full pipeline

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

### 3. Or run stage by stage

```bash
python main.py --config configs/full_benchmark.yaml --stage run
python main.py --config configs/full_benchmark.yaml --stage aggregate
python main.py --config configs/full_benchmark.yaml --stage visualize
```

### 4. Inspect outputs

```bash
# Stage 1 / 1b
ls outputs/full_benchmark
ls outputs/full_benchmark/statistics

# Stage 2
ls outputs/full_benchmark/visualizations/busi
ls outputs/full_benchmark/visualizations/montgomery
```

---

## Pipeline Architecture

### Stage Summary

| Stage | Command | Main Input | Main Output |
|---|---|---|---|
| `run` | `--stage run` | datasets + model configs + noise presets | per-image raw CSVs, artifacts, noise cache |
| `aggregate` | `--stage aggregate` | raw CSVs under experiment output | per-file stats CSVs, `statistics_merged.csv`, `statistics/` summaries |
| `visualize` | `--stage visualize` | `statistics_merged.csv` and optional artifacts | PDF figures under `visualizations/` |
| `all` | `--stage all` | config | run all 3 stages in sequence |

### Data Flow

```text
Dataset image + GT mask
    -> prompt generation from GT
    -> noise injection on-the-fly
    -> model inference
    -> per-image metrics
    -> *_raw.csv
    -> *_stats.csv
    -> statistics_merged.csv
    -> statistics/*.csv
    -> visualizations/*.pdf
```

---

## Supported Components

### Dataset Adapters

Registered in `datasets/dataset_registry.py`:

- `ImageMaskFolderAdapter`
- `BUSIAdapter`
- `CAMUSAdapter`

`ImageMaskFolderAdapter` supports:

- `root/image_dir` + `root/mask_dir`
- `root/<split>/image_dir` + `root/<split>/mask_dir`
- explicit `sources:` entries in YAML

### Model Registry

Registered in `core/model_manager.py`:

- `SAM`, `SAM1`
- `SAM2`
- `SAM3`
- `MEDSAM`, `MEDSAM1`
- `MEDICOSAM`
- `MEDSAM2`
- `MEDSAM3`
- `SAM-MED2D`
- `ULTRASAM`

Important behavior:

- The runner must load real model weights.
- If loading fails and `allow_fallback` is not enabled, the benchmark stops with an error.
- Heuristic fallback is only allowed when `model_cfg.allow_fallback=true`, and is not recommended for benchmarking.

### Prompt Modes

Supported by `models/wrappers/prompt_utils.py`:

- `prompt_point`
- `prompt_multi_point`
- `prompt_bbox`
- `prompt_point_box`
- `autogen`

Current `full_benchmark.yaml` uses:

- `prompt_point`
- `prompt_bbox`
- `prompt_point_box`

Prompt resolution behavior:

- `prompt_point`: point only, no bbox
- `prompt_bbox`: bbox only, no points
- `prompt_point_box`: one point plus one bbox
- bbox is derived from GT with adaptive margin
- points are sampled deterministically from foreground

### Noise Registry

Registered in `noises/noise_registry.py`.

Enabled in `configs/full_benchmark.yaml`:

- `gaussian`
- `speckle`
- `salt_pepper`
- `motion_blur`
- `jpeg`
- `pixelation`
- `low_brightness`
- `high_brightness`
- `low_contrast`
- `high_contrast`
- `rician`

Available in code but disabled in `full_benchmark.yaml`:

- `poisson`
- `bias_field`
- several optional extras such as `uniform`, `quantization`, `defocus_blur`, `coarse_dropout`, `gridmask`

Noise behavior:

- noise is applied on-the-fly, never pre-generated into the dataset
- `L0` returns the original image unchanged
- noise seeds are deterministic per dataset, image, noise type, level, and `noise_seed`
- noisy images can be cached to `noise_cache/`

---

## Metrics

Metrics are implemented in `metrics/metric_manager.py`:

- `IoU`
- `Dice`
- `Recall`
- `Precision`
- `F1`
- `HD` (Hausdorff Distance)

### Empty-Mask Handling

- If GT is empty, `IoU`, `Dice`, `Recall`, `Precision`, and `F1` return `NaN`
- If GT is non-empty and prediction is empty, `Precision` returns `0.0`
- If GT is empty, `HD` returns `NaN`
- If GT is non-empty and prediction is empty, `HD` returns `inf`
- If SciPy is unavailable, `HD` falls back to `NaN`

### Aggregation

Stage `aggregate` computes grouped statistics by:

```text
dataset, model, prompt_mode, noise_type, noise_level
```

For each metric it writes:

- mean
- standard deviation
- coefficient of variation (`*_cv_pct`)
- number of valid values (`*_n_valid`)

It also records:

- `gt_empty_rate`
- `n_gt_non_empty`
- `pred_empty_rate`
- `n_images`
- `n_rows`

---

## Configuration

The main config file is:

```text
configs/full_benchmark.yaml
```

### Core Sections

```yaml
exp:
  name: "full_benchmark"
  out_root: "outputs"

device: "cuda"

stage1:
  save_artifacts: true
  artifact_samples_per_case: 3
  cache_noisy_images: true
  noise_cache_dir: "noise_cache"
  clear_noise_cache_on_start: false
  gc_collect_interval: 0
  cuda_cache_clear_interval: 0

noise_config:
  base_seed: 42
  n_noise_seeds: 1
```

### Device Selection

You can choose devices in 3 ways:

```yaml
device: "cuda:0"
devices: ["cuda:0", "cuda:1"]
num_gpus: 2
```

CLI overrides:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --device cuda:1
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 2
```

### Important Config Notes

- `ConfigManager` requires non-empty `exp`, `datasets`, and `models`
- if `noise_config.n_noise_seeds` is missing, the config manager defaults it to `3`
- `full_benchmark.yaml` explicitly sets `n_noise_seeds: 1`
- relative `exp.out_root` is normalized against the project root by `main.py`

---

## Running The Benchmark

### Full Run

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

### Stage 1: Inference

```bash
python main.py --config configs/full_benchmark.yaml --stage run
```

Useful overrides:

```bash
# Limit dataset size for quick testing
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 50

# Run only selected datasets
python main.py --config configs/full_benchmark.yaml --stage run --datasets Montgomery,BUSI

# Run only selected models
python main.py --config configs/full_benchmark.yaml --stage run --models SAM2,MedSAM3

# Alias for a single model
python main.py --config configs/full_benchmark.yaml --stage run --model SAM2

# Select a device
python main.py --config configs/full_benchmark.yaml --stage run --device cpu

# Split models across multiple GPUs
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 2
```

Important CLI note:

- `--datasets` and `--models` are comma-separated strings, for example `Montgomery,BUSI`, not space-separated lists.

### Stage 1b: Aggregate

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

This stage:

- finds all `*/*/*_raw.csv` files under the experiment directory
- creates matching `*_stats.csv` files alongside them
- merges them into `statistics_merged.csv`
- creates `stage1b_summary.csv`
- creates comprehensive summary tables under `statistics/`

### Stage 2: Visualize

```bash
python main.py --config configs/full_benchmark.yaml --stage visualize
```

This stage:

- reads `statistics_merged.csv`
- optionally uses saved artifacts
- writes PDF figures under `visualizations/`

If you have an older run under `outputs/outputs/full_benchmark`, `aggregate` and `visualize` will auto-detect it and write canonical outputs back to `outputs/full_benchmark`.

---

## Output Structure

Canonical layout for `full_benchmark`:

```text
outputs/full_benchmark/
+-- raw_files_manifest.csv
+-- statistics_merged.csv
+-- stage1b_summary.csv
+-- busi/
|   +-- sam/
|   |   +-- sam1_busi_point_raw.csv
|   |   +-- sam1_busi_point_stats.csv
|   |   +-- sam1_busi_bbox_raw.csv
|   |   +-- sam1_busi_bbox_stats.csv
|   |   +-- sam1_busi_pointbox_raw.csv
|   |   `-- sam1_busi_pointbox_stats.csv
|   +-- sam2/
|   +-- medsam/
|   `-- ...
+-- montgomery/
|   `-- ...
+-- statistics/
|   +-- 01_overall_summary.csv
|   +-- 02_model_summary.csv
|   +-- 03_mode_summary.csv
|   +-- 04_noise_summary.csv
|   +-- 05_level_summary.csv
|   +-- 06_model_noise_matrix_*.csv
|   +-- 07_model_level_matrix_*.csv
|   +-- 08_noise_level_matrix_*.csv
|   +-- 10_robustness_analysis.csv
|   +-- statistics_manifest.csv
|   `-- by_metric/
|       +-- summary_dice.csv
|       +-- summary_f1.csv
|       +-- summary_hd.csv
|       +-- summary_iou.csv
|       +-- summary_precision.csv
|       `-- summary_recall.csv
+-- visualizations/
|   +-- schematics/
|   |   `-- prompt_modes_illustration_point_bbox_pointbox.pdf
|   +-- busi/
|   |   +-- busi_iou_lineplot_by_level_point.pdf
|   |   +-- busi_iou_lineplot_by_level_bbox.pdf
|   |   +-- busi_iou_lineplot_by_level_point_bbox.pdf
|   |   +-- busi_iou_heatmap_model_vs_noise.pdf
|   |   +-- busi_iou_heatmap_model_vs_level.pdf
|   |   +-- busi_iou_mode_comparison_all_levels.pdf
|   |   +-- busi_iou_ranking_models.pdf
|   |   +-- busi_iou_ranking_noise_difficulty.pdf
|   |   `-- ...
|   `-- montgomery/
|       `-- ...
+-- artifacts/
|   +-- _shared/
|   |   `-- busi/
|   |       `-- gaussian/
|   |           `-- L1/
|   |               `-- seed0/
|   |                   +-- sample_original.png
|   |                   +-- sample_noisy.png
|   |                   `-- sample_gt.png
|   `-- busi/
|       `-- sam/
|           `-- point/
|               `-- gaussian/
|                   `-- L1/
|                       `-- seed0/
|                           `-- sample_pred.png
`-- noise_cache/
    `-- busi/
        `-- gaussian/
            `-- L1/
                `-- seed0/
                    `-- sample_256x256x1_<hash>.npy
```

### Raw CSV Columns

Stage 1 raw files contain:

```text
dataset, model, prompt_mode, noise_type, noise_level, noise_seed, image_id,
prompt_x, prompt_y,
bbox_x0, bbox_y0, bbox_x1, bbox_y1, bbox_w, bbox_h, bbox_area,
gt_fg_pixels, pred_fg_pixels, is_gt_empty, is_pred_empty,
IoU, Dice, Recall, Precision, F1, HD
```

### Merged Statistics Columns

`statistics_merged.csv` contains grouped statistics such as:

```text
dataset, model, prompt_mode, noise_type, noise_level,
gt_empty_rate, n_gt_non_empty, pred_empty_rate,
Dice, Dice_std, Dice_cv_pct, Dice_n_valid,
IoU, IoU_std, IoU_cv_pct, IoU_n_valid,
Recall, Recall_std, Recall_cv_pct, Recall_n_valid,
Precision, Precision_std, Precision_cv_pct, Precision_n_valid,
F1, F1_std, F1_cv_pct, F1_n_valid,
HD, HD_std, HD_cv_pct, HD_n_valid,
n_images, n_rows, source_stats_file
```

---

## Statistics And Visualization Outputs

### Statistics

When all 6 metrics are present, `aggregate` writes 31 CSV files under `statistics/`:

1. `01_overall_summary.csv`
2. `02_model_summary.csv`
3. `03_mode_summary.csv`
4. `04_noise_summary.csv`
5. `05_level_summary.csv`
6. `06_model_noise_matrix_<metric>.csv` for each metric
7. `07_model_level_matrix_<metric>.csv` for each metric
8. `08_noise_level_matrix_<metric>.csv` for each metric
9. `10_robustness_analysis.csv`
10. `by_metric/summary_<metric>.csv` for each metric
11. `statistics_manifest.csv`

### Visualizations

`visualize` generates:

- 1 prompt schematic PDF under `visualizations/schematics/`
- per-dataset line plots:
  6 metrics x 3 prompt modes = 18 PDFs when all 3 modes are present
- per-dataset mode-comparison plots:
  6 PDFs
- per-dataset heatmaps:
  6 metrics x 2 heatmaps = 12 PDFs
- per-dataset ranking plots:
  6 metrics x 2 rankings = 12 PDFs
- per-dataset noise gallery:
  1 PDF
- optional per-dataset segmentation gallery:
  generated only when artifact coverage allows it

With 6 metrics and 3 prompt modes, the standard output is 49 PDFs per dataset plus 1 schematic PDF. If segmentation-gallery conditions are met, you may get 1 additional PDF per dataset.

### What The Current Figures Actually Show

The current `analysis/comprehensive_visualization.py` generates:

- line plots of metric vs noise level, averaged across noise types, one PDF per `(dataset, metric, prompt_mode)`
- mode-comparison plots that compare prompt modes across levels
- heatmaps for:
  - model vs noise type
  - model vs noise level
- ranking plots for:
  - models
  - noise difficulty
- a noise gallery over `L0` to `L9`

All figure titles are omitted; filenames are intended to describe the plot.

---

## Troubleshooting

### `No raw CSV files found under ...`

Run Stage 1 first:

```bash
python main.py --config configs/full_benchmark.yaml --stage run
```

If you have an older nested run under `outputs/outputs/full_benchmark`, the current `aggregate` command should auto-detect it.

### `Missing .../statistics_merged.csv. Run Stage 1b first.`

Run:

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

### CUDA / device issues

Try CPU:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --device cpu
```

Or inspect Torch CUDA visibility:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### Out of memory

Use a smaller run:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 20
python main.py --config configs/full_benchmark.yaml --stage run --datasets Montgomery
python main.py --config configs/full_benchmark.yaml --stage run --models SAM2
```

You can also increase cleanup frequency in config:

```yaml
stage1:
  gc_collect_interval: 10
  cuda_cache_clear_interval: 20
```

### Missing datasets

`full_benchmark.yaml` references 6 datasets. If you only have a subset locally, either:

- run with `--datasets` and only include available datasets
- or edit the YAML to remove unavailable datasets

### Missing weights

Every configured model must load real weights unless you explicitly opt into fallback behavior. Verify checkpoint paths in `configs/full_benchmark.yaml`.

---

## References

Model wrappers and external code live under:

- `models/wrappers/`
- `models/external/`

Core pipeline modules:

- `main.py`
- `core/experiment_engine.py`
- `analysis/aggregator.py`
- `analysis/stats_merger.py`
- `analysis/comprehensive_statistics.py`
- `analysis/comprehensive_visualization.py`
- `metrics/metric_manager.py`
- `noises/noise_manager.py`
- `models/wrappers/prompt_utils.py`

---

## Current Repository Note

The repository currently contains example/legacy outputs under `outputs/outputs/full_benchmark/`. New runs from the current `main.py` write to the canonical location:

```text
outputs/full_benchmark/
```

If you re-run `aggregate` and `visualize`, the code will read the legacy run if needed and regenerate canonical outputs in the new location.
