# Medical

This project evaluates segmentation robustness under image perturbations with **three independent stages**:

1. `Stage 1` Run experiments (inference + per-image metrics)
2. `Stage 1b` Aggregate raw metrics (mean/std/CV%)
3. `Stage 2` Analysis and visualization (PDF only, no inference)

## Key Design Rules

- Noise is applied **on-the-fly** during inference.
- Noise is **not pre-generated** into datasets.
- Noise is deterministic with `noise_config.base_seed` and `noise_config.n_noise_seeds`.
- Evaluation uses **all available dataset images** (split folders are merged automatically if present).
- Stage 2 reads aggregated CSV outputs for analysis plots/tables and may use pre-saved Stage-1 artifacts for image galleries.
- Stage 2 never reruns inference.

## Project Structure

```
main.py                          # CLI entrypoint
core/
    config_manager.py            # YAML config loading, validation, device resolution
    dataset_manager.py           # Centralised dataset construction & caching
    model_manager.py             # Model runner factory & registry
    experiment_engine.py         # Stage 1 – full inference loop
analysis/
    aggregator.py                # Stage 1b – per-CSV stats (mean/std/CV%)
    stats_merger.py              # Stage 1b – merge all stats into one CSV
datasets/
    base_dataset.py              # DatasetAdapter ABC
    dataset_registry.py          # Adapter registry & builder
    adapters/
        image_mask_folder_adapter.py
        busi_adapter.py
        camus_adapter.py
models/
    external/                    # Third-party model source code (untouched)
        sam/
        sam2/
        sam3/
        MedSAM/
        MedSAM2/
        MedSAM3/
        SAM-Med2D/
        UltraSam/
    wrappers/                    # Internal model wrappers
        base_model.py            # ModelRunner ABC + shared CV utilities
        prompt_utils.py          # Prompt normalisation & building
        sam_runner.py
        sam2_runner.py
        sam3_runner.py
        medsam_runner.py
        medsam2_runner.py
        medsam3_runner.py
        sam_med2d_runner.py
        ultrasam_runner.py
noises/                          # Noise implementations
    base.py                      # NoiseBase ABC
    noise_registry.py            # Noise type registry
    noise_manager.py             # Apply noise by name + level
    gaussian.py, poisson.py, salt_pepper.py, motion_blur.py,
    bias_field.py, low_contrast.py, optional_extras.py
metrics/
    metric_manager.py            # IoU, Dice, Recall, Precision, F1, HD
runner/
    result_writer.py             # Optional CSV writer utility
viz/                             # Stage 2 – all output as multi-page PDF
    plot_metrics.py
    model_comparison.py
    noise_gallery.py
    prediction_overlay.py
    prompt_visualization.py
    statistical_tables.py
configs/
    full_benchmark.yaml
    phase1.yaml
    phase2.yaml
```

## Configuration

Experiments are controlled by YAML (see `configs/phase1.yaml`, `configs/phase2.yaml`, `configs/full_benchmark.yaml`).

Minimal pattern:

```yaml
exp:
  name: "exp_name"
  out_root: "outputs"

device: "cuda"          # or "cuda:0", "cpu"

datasets:
  - name: "CAMUS"
    adapter: "CAMUSAdapter"
    root: "data/CAMUS/CAMUS_public/database_nifti"

models:
  - name: "SAM"
    runner: "SAM1"
    prompt_modes: ["prompt_point", "prompt_bbox", "prompt_point_box"]

noises: [gaussian, poisson, salt_pepper, motion_blur, bias_field, low_contrast]
levels: [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]

noise_config:
  base_seed: 42
  n_noise_seeds: 3

protocols:
  coupled_presets:
    gaussian:
      L1: {p: 1, sigma: 3}
      ...
      L9: {p: 1, sigma: 48}
```

### Device / GPU Configuration

There are **3 ways** to specify which GPU(s) to use:

| Method | Config key | Example |
|---|---|---|
| Single GPU | `device` | `device: "cuda:0"` |
| Explicit multi-GPU | `devices` | `devices: ["cuda:0", "cuda:1"]` |
| Auto N GPUs | `num_gpus` | `num_gpus: 2` |

**CLI overrides** (take priority over config):

```bash
# Force a specific GPU
python main.py --config configs/full_benchmark.yaml --device cuda:1

# Run on 2 GPUs in parallel
python main.py --config configs/full_benchmark.yaml --num_gpus 2
```

When using **multi-GPU**, Stage 1 automatically:
- Splits models round-robin across GPUs (e.g. 8 models / 2 GPUs → 4 models per GPU)
- Spawns a separate process per GPU (`multiprocessing`, start method `spawn`)
- Each process loads its assigned models on its own GPU and runs inference in parallel
- All CSV results are written to the same `exp_dir`; Stage 1b and Stage 2 work as usual

## CLI

### Stage 1: Run Experiments

```bash
python main.py --config configs/phase1.yaml --stage run
```

Optional filters:

```bash
python main.py --config configs/phase1.yaml --stage run --max_samples 50
python main.py --config configs/phase1.yaml --stage run --datasets TN3K --models SAM
python main.py --config configs/phase1.yaml --stage run --device cuda:0
python main.py --config configs/full_benchmark.yaml --stage run --num_gpus 2
```

### Stage 1b: Aggregate Metrics

```bash
python main.py --config configs/phase1.yaml --stage aggregate
```

### Stage 2: Visualize (No Inference)

```bash
python main.py --config configs/phase1.yaml --stage visualize
```

### Run End-to-End

```bash
python main.py --config configs/phase1.yaml --stage all
```

## Outputs

### Stage 1 – raw per-image CSV

Path pattern:

`outputs/{experiment}/{dataset}/{model}/{runner}_{dataset}_{prompt}_raw.csv`

Columns:

- `dataset, model, prompt_mode, noise_type, noise_level, noise_seed, image_id`
- `IoU, Dice, Recall, Precision, F1, HD`

### Stage 1b – aggregated stats CSV

Per raw file:

`.../{runner}_{dataset}_{prompt}_stats.csv`

Grouped by:

- `noise_type, noise_level`

Metrics include:

- mean (e.g. `Dice`)
- std (e.g. `Dice_std`)
- CV% (e.g. `Dice_cv_pct`)

Global merged file:

`outputs/{experiment}/statistics_merged.csv`

### Stage 2 – figures (PDF)

`outputs/{experiment}/figures/`

- `metric_curves_dice.pdf`
- `metric_curves_iou.pdf`
- `robustness_dice_drop.pdf`
- `prompt_visualization.pdf`
- `prompt_comparison_dice.pdf`
- `model_comparison_dice.pdf`
- `noise_gallery.pdf`
- `prediction_overlay.pdf`
- `statistical_tables.pdf`

## Extensibility

- **New dataset**: implement adapter in `datasets/adapters/` and register in `datasets/dataset_registry.py`.
- **New model**: implement runner in `models/wrappers/` and register in `core/model_manager.py`.
- **New noise**: implement in `noises/` and register in `noises/noise_registry.py`.
- **New metrics**: extend `metrics/metric_manager.py` and update aggregation keys.
- **New perturbation protocols**: extend `protocols.coupled_presets` in YAML.
