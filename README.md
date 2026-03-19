# SAM-Noisy-Medical

Benchmark pipeline for segmentation robustness under controlled noise.

## Pipeline Overview

1. Stage `run`: inference + per-image metrics (raw CSV)
2. Stage `aggregate`: merge per-image CSV into grouped stats
3. Stage `visualize`: generate PDF plots/tables from stats

Noise is always applied on-the-fly (never pre-generated into dataset files).

## Prompt Standard (Synchronized Across Models)

Prompt building is centralized in [models/wrappers/prompt_utils.py](models/wrappers/prompt_utils.py).

### Modes

- `prompt_point`
- `prompt_bbox`
- `prompt_point_box`
- `autogen`

### Coordinate convention

- Point: `(x, y)` in pixel coordinates.
- Box: `XYXY = (x0, y0, x1, y1)`, inclusive endpoints.
- Width/height in logs are computed as:
  - `bbox_w = x1 - x0 + 1`
  - `bbox_h = y1 - y0 + 1`
  - `bbox_area = bbox_w * bbox_h`

### How bbox/point are selected

Priority order:

1. Use explicit prompt values if provided by caller.
2. Otherwise derive from `gt_mask`:
   - `point` = mask centroid (rounded)
   - `bbox` = tight foreground box with `margin_ratio=0.04`, then clipped to image bounds
3. If no foreground exists, fallback to deterministic center prompt:
   - `point = (W//2, H//2)`
   - `bbox = (W//4, H//4, 3W//4, 3H//4)`

All wrappers now use this same resolved prompt path.

## Stage-1 CSV now includes prompt debug fields

Each raw row includes:

- `prompt_x`, `prompt_y`
- `bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1`
- `bbox_w`, `bbox_h`, `bbox_area`

This lets you directly verify what bbox/point was used for each image/noise-level/model.

## Models

Model runners are in [models/wrappers](models/wrappers) and are managed by [core/model_manager.py](core/model_manager.py).

Current registry keys:

- `SAM1` / `SAM`
- `SAM2`
- `SAM3`
- `MEDSAM1` / `MEDSAM`
- `MEDSAM2`
- `MEDSAM3`
- `SAM-MED2D`
- `ULTRASAM`

## Noise and Levels

Defined in config (`noises`, `levels`, `protocols.coupled_presets`) and executed by [noises/noise_manager.py](noises/noise_manager.py).

- Recommended levels: `L0...L9`
- `L0` is always clean baseline (no perturbation)
- Non-`L0` params are taken from `protocols.coupled_presets[noise_type][level]`
- Deterministic seed key uses:
  - `base_seed`, `noise_seed`, `dataset_name`, `image_id`, `noise_type`, `level`

## Installation

Use your conda env (example: `sam1`) and install base deps:

```bash
pip install -r requirements.txt
```

If you run UltraSAM-only workflows, use:

```bash
pip install -r requirements_ultra.txt
```

Important for GPU:

- Install a `torch/torchvision` build compatible with your CUDA + GPU architecture.
- On unsupported architectures, some wrappers auto-fallback to CPU for stability.

## Run

### Stage 1 (inference)

```bash
python main.py --config configs/full_benchmark.yaml --stage run
```

Optional Stage-1 optimization flags in config:

- `stage1.cache_noisy_images: true`  
  Reuse noisy images across models/prompt modes (same dataset/image/noise/level/seed).
- `stage1.noise_cache_dir: "noise_cache"`  
  Relative path under `outputs/{exp}` (or absolute path).

Example targeted run:

```bash
python main.py --config configs/full_benchmark.yaml --stage run --datasets Montgomery --models SAM2 --max_samples 10
```

### Stage 1b (aggregate)

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

### Stage 2 (visualize)

```bash
python main.py --config configs/full_benchmark.yaml --stage visualize
```

### End-to-end

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

## Output Layout

- Raw CSV:
  - `outputs/{exp}/{dataset}/{model}/{runner}_{dataset}_{prompt}_raw.csv`
- Stats CSV:
  - `.../{runner}_{dataset}_{prompt}_stats.csv`
- Merged stats:
  - `outputs/{exp}/statistics_merged.csv`
- Noisy image cache:
  - `outputs/{exp}/noise_cache/{dataset}/{noise}/{level}/seed{n}/{image_id}_{shape}_{sig}.npy`
- Shared artifacts (saved once, reused across models):
  - `outputs/{exp}/artifacts/_shared/{dataset}/{noise}/{level}/seed{n}/{image_id}_{original|noisy|gt}.png`
- Model-specific artifacts:
  - `outputs/{exp}/artifacts/{dataset}/{model}/{prompt}/{noise}/{level}/seed{n}/{image_id}_pred.png`
- Publication-ready visualization suite (auto long/wide CSV normalization):
  - `outputs/visualizations/{dataset_name}/*.pdf`

## Notes

- Stage 2 never reruns inference.
- Prompt mode normalization accepts aliases (`bbox`, `point`, `point_box`, etc.).
- If results look wrong, first inspect prompt columns in raw CSV to verify bbox/point are sensible.
- Stage 1 supports noisy-image caching (`stage1.cache_noisy_images: true`) to avoid re-applying the same perturbation across models/prompts.
- Stage 1 artifacts now store shared `original/noisy/gt` once under `artifacts/_shared/...`; model folders keep `*_pred.png`.
