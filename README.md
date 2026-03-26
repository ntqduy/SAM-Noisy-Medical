# SAM-Noisy-Medical: Robustness Benchmark for Medical Image Segmentation

A comprehensive benchmark pipeline for evaluating segmentation model robustness under 12+ controlled noise types and 10 severity levels (L0-L9).

**Status**: ✅ Full benchmark completed with statistics & visualizations

- **8 models** tested (SAM, SAM2, SAM3, MedSAM, MedSAM2, MedSAM3, SAM-Med2D, UltraSAM)
- **2 datasets** (BUSI, Montgomery)
- **12 noise types** (Gaussian, Poisson, Rician, Salt-Pepper, JPEG, Pixelation, Motion Blur, Speckle, Low/High Brightness, Low/High Contrast)
- **10 severity levels** (L0=clean, L1=mild, ..., L9=catastrophic)
- **3 prompt modes** (point, bbox, point+bbox)
- **6 metrics** (IoU, Dice, Recall, Precision, F1, Hausdorff Distance)

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Noise System](#noise-system)
4. [Metrics](#metrics)
5. [Models](#models)
6. [Configuration](#configuration)
7. [Installation](#installation)
8. [Running the Benchmark](#running-the-benchmark)
9. [Output Structure](#output-structure)
10. [Visualization & Statistics](#visualization--statistics)
11. [Known Issues & Fixes](#known-issues--fixes)
12. [Results Interpretation](#results-interpretation)
13. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### One-liner: Full benchmark

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

### Generation only (statistics & plots):

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
python main.py --config configs/full_benchmark.yaml --stage visualize
```

### Inspect results:

```bash
# Statistics
ls outputs/full_benchmark/statistics/*.csv

# Visualizations (PDF)
ls outputs/full_benchmark/visualizations/busi/*.pdf
```

---

## 🏗️ Pipeline Architecture

### Three-Stage Pipeline

| Stage | Command | Input | Output | Time |
|-------|---------|-------|--------|------|
| **run** | `--stage run` | Config + datasets | Raw CSV per model/prompt | ~8-12h (GPU) |
| **aggregate** (1b) | `--stage aggregate` | Raw CSVs | Merged statistics | ~5 min |
| **visualize** (2) | `--stage visualize` | Merged stats | PDFs + heatmaps | ~10 min |

### Data Flow

```
Dataset images + GT masks
        ↓
   [Apply noise] (on-the-fly)
        ↓
   [Model inference] (8 models × 3 prompts)
        ↓
   [Compute metrics] (IoU, Dice, ..., HD)
        ↓
   Raw CSV (per model/prompt/noise/level)
        ↓
   [Merge & aggregate]
        ↓
   statistics_merged.csv
        ↓
   [Generate visualizations]
        ↓
   PDF plots (line plots, heatmaps, rankings, galleries)
```

---

## 🎨 Noise System

### Architecture

- **Base class**: `noises/base.py::NoiseBase`
- **Implementations**: 12+ noise types in `noises/`
- **Registry**: `noises/noise_registry.py` (plugin-based)
- **Severity mapping**: `noises/severity_mapping.py` (direction per noise type)

### Noise Types & Severity Direction

| Noise Type | Parameter | Range | Direction | Notes |
|-----------|-----------|-------|-----------|-------|
| **Gaussian** | sigma | 0-90 | Standard | Higher σ → more noise |
| **Poisson** | peak | 1-100 | **Inverted** | Lower peak → more noise (photon counting) |
| **Rician** (MRI) | sigma | 0-80 | Standard | Magnitude noise for MRI |
| **Salt & Pepper** | amount | 0-0.5 | Standard | Fraction of pixels affected |
| **JPEG** | quality | 5-90 | **Inverted** | Lower quality → more compression |
| **Pixelation** | block_size | 2-50 | Standard | Larger blocks → more degradation |
| **Motion Blur** | kernel | 3-40 | Standard | Larger kernel → stronger blur |
| **Speckle** | sigma | 0-0.9 | Standard | Multiplicative Gaus sian noise |
| **Low Brightness** | factor | 0.05-1.0 | **Inverted** | Lower factor → darker (more severe) |
| **High Brightness** | factor | 1.05-5.0 | Standard | Higher factor → brighter |
| **Low Contrast** | alpha | 0.1-1.0 | **Inverted** | Lower α → less contrast (inverted) |
| **High Contrast** | factor | 1.5-8.0 | Standard | Higher factor → more contrast |

### Severity Levels (L0-L9)

```yaml
L0: clean (no noise)
L1: very mild
L2: mild
L3: moderate
L4: strong
L5: severe
L6: extreme
L7: destructive
L8: near failure
L9: catastrophic
```

Each level has pre-configured parameters in `configs/full_benchmark.yaml::protocols.coupled_presets`.

### Key Features

✅ **Signal-dependent**: Noise accounts for image intensity
✅ **Reproducible**: Deterministic seed per image/noise/level
✅ **Cached**: Optional on-the-fly caching of noisy images (reuse across models)
✅ **Metadata tracking**: Full provenance in CSV (seed, params, PSNR, SSIM)

---

## 📊 Metrics

### Six Segmentation Metrics

| Metric | Formula | Range | Direction | Notes |
|--------|---------|-------|-----------|-------|
| **IoU** | TP/(TP+FP+FN) | [0, 1] | ↑ Higher better | Jaccard index |
| **Dice** | 2TP/(2TP+FP+FN) | [0, 1] | ↑ Higher better | F1-like, symmetric |
| **Recall** | TP/(TP+FN) | [0, 1] | ↑ Higher better | Sensitivity, coverage |
| **Precision** | TP/(TP+FP) | [0, 1] | ↑ Higher better | Specificity, accuracy |
| **F1** | 2PR/(P+R) | [0, 1] | ↑ Higher better | Harmonic mean of P & R |
| **HD** | max(d(A,B), d(B,A)) | [0, ∞] | ↓ **Lower better** | Hausdorff distance (boundary) |

### Empty Predictions

- **IoU, Dice, Recall**: Return `NaN` if GT is empty (no foreground to segment)
- **Precision**: Returns `0.0` if prediction is empty (no positive predictions → 0% precision)
- **HD**: Returns `∞` (infinite distance between empty sets)

### Metric Direction in Analysis

- **Higher-is-better metrics** (IoU, Dice, Recall, Precision, F1): Lower noise severity → higher metric
- **Lower-is-better metric** (HD): Lower noise severity → lower distance (better)

Statistics correctly negate HD drop sign so "robustness" is always "higher is better".

---

## 🤖 Models

### Model Runners

Located in `models/wrappers/` and registered in `core/model_manager.py`.

| Model | Runner | Checkpoint | Config | Notes |
|-------|--------|------------|--------|-------|
| **SAM (v1)** | `SAM1` | `weights/sam_b.pt` | — | Original Segment Anything |
| **SAM2** | `SAM2` | `weights/sam2_b.pt` | `sam2/sam2_hiera_b+.yaml` | Video SAM, adapted to 2D |
| **SAM3** | `SAM3` | `weights/sam3.pt` | — | Latest SAM (academic release) |
| **MedSAM** | `MEDSAM1` | `weights/medsam_vit_b.pth` | — | SAM fine-tuned on medical data |
| **MedSAM2** | `MEDSAM2` | `weights/MedSAM2/MedSAM2_latest.pt` | `sam2.1_hiera_t512.yaml` | MedSAM on SAM2 backbone |
| **MedSAM3** | `MEDSAM3` | `weights/sam3.pt` + LoRA | `weights/MedSAM3/best_lora_weights.pt` | LoRA-adapted SAM3 |
| **SAM-Med2D** | `SAM-MED2D` | `weights/sam-med2d_b.pth` | — | SAM fine-tuned on 2D medical |
| **UltraSAM** | `ULTRASAM` | `weights/UltraSam.pth` | Mode-specific configs | Efficient high-res SAM variant |

### Model Features

✅ **Unified interface**: All wrappers inherit from `BaseModel`
✅ **Prompt standardization**: All use centralized `prompt_utils.py`
✅ **Mode-specific configs**: UltraSAM can use different configs per prompt mode
✅ **Multi-mask selection**: Best-of-3 masks selected by IoU score
✅ **Automatic fallback**: CPU fallback on unsupported GPU architectures

---

## ⚙️ Configuration

### Main Config File

`configs/full_benchmark.yaml` controls all benchmark parameters:

```yaml
exp:
  name: "full_benchmark"
  out_root: "outputs"

device: "cuda"  # or "cuda:0", ["cuda:0", "cuda:1"], "cpu"

datasets:
  - name: "Montgomery"
    adapter: "ImageMaskFolderAdapter"
    root: "data/Montgomery"
    # ... (BUSI, CAMUS, TN3K, TG3K, DDTI also configured)

models:
  - name: "SAM"
    runner: "SAM1"
    checkpoint: "weights/sam_b.pt"
    prompt_modes: ["prompt_point", "prompt_bbox", "prompt_point_box"]
  # ... (7 more models)

noises: [gaussian, speckle, salt_pepper, motion_blur, jpeg, ...]
levels: [L0, L1, ..., L9]

protocols:
  coupled_presets:
    gaussian:
      L1: {p: 1, sigma: 5}
      L2: {p: 1, sigma: 10}
      # ...
    speckle:
      L1: {p: 1, sigma: 0.05}
      # ...
```

### Stage 1 Options

```yaml
stage1:
  save_artifacts: true              # Save original/noisy/pred images
  artifact_samples_per_case: 3      # Per noise/level/model
  cache_noisy_images: true          # Reuse noisy images across models
  noise_cache_dir: "noise_cache"    # Relative to exp_out/{exp}/
  clear_noise_cache_on_start: false
  gc_collect_interval: 0            # Periodic gc.collect() every N
  cuda_cache_clear_interval: 0      # Periodic torch.cuda.empty_cache()
```

---

## 💾 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (GPU) or CPU-only
- 24+ GB VRAM (for full benchmark with all models)

### Setup

1. Clone/enter repo:
```bash
cd "d:/AI/TA_TH/project SAM/project1"
```

2. Create conda environment:
```bash
conda create -n sam-bench python=3.10
conda activate sam-bench
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model weights:
- Create `weights/` directory
- Download SAM, MedSAM, SAM2, SAM3 checkpoints
- Place in appropriate subdirectories

5. Prepare datasets:
```bash
# Organize data/Montgomery/, data/BUSI/, etc.
mkdir -p data/{Montgomery,BUSI}
# Copy images and masks following expected folder structure
```

---

## 🏃 Running the Benchmark

### Full Pipeline (All Stages)

```bash
python main.py --config configs/full_benchmark.yaml --stage all
```

### Individual Stages

#### Stage 1: Inference

```bash
# Full benchmark (8 models × 2 datasets × 3 modes × 12 noises × 10 levels)
python main.py --config configs/full_benchmark.yaml --stage run

# Targeted run (testing)
python main.py --config configs/full_benchmark.yaml --stage run \
  --datasets Montgomery \
  --models SAM2 MedSAM3 \
  --max_samples 50
```

**Duration**: ~8-12 hours (full, single GPU)

#### Stage 1b: Aggregate Statistics

```bash
python main.py --config configs/full_benchmark.yaml --stage aggregate
```

Produces:
- `statistics_merged.csv` (merged with all images/models/noises/levels)
- Per-model/prompt aggregated statistics

#### Stage 2: Visualizations

```bash
python main.py --config configs/full_benchmark.yaml --stage visualize
```

Generates:
- Line plots (metrics vs. levels, per prompt mode)
- Heatmaps (model × noise, model × level)
- Ranking charts (model robustness, noise difficulty)
- Noise gallery (visual examples L0-L9)
- Prompt mode illustrations

All outputs are **PDF** with **no titles** (descriptive filenames instead).

### CLI Overrides

```bash
python main.py \
  --config configs/full_benchmark.yaml \
  --stage run \
  --device cuda:1 \
  --num_gpus 2 \
  --datasets Montgomery BUSI \
  --models SAM SAM2 \
  --max_samples 100
```

---

## 📁 Output Structure

```
outputs/full_benchmark/
├── Montgomery/
│   └── MedSAM/
│       ├── runner_Montgomery_prompt_point_raw.csv
│       ├── runner_Montgomery_prompt_point_stats.csv
│       ├── runner_Montgomery_prompt_bbox_raw.csv
│       └── runner_Montgomery_prompt_point_box_raw.csv
├── BUSI/
│   ├── SAM/runner_BUSI_prompt_point_raw.csv
│   ├── SAM2/runner_BUSI_prompt_point_raw.csv
│   └── ...
├── statistics_merged.csv               # Merged all stats
├── statistics/
│   ├── 01_overall_summary.csv
│   ├── 02_model_summary.csv
│   ├── 03_mode_summary.csv
│   ├── 04_noise_summary.csv
│   ├── 05_level_summary.csv            # ← Includes all L0-L9
│   ├── 06_model_noise_matrix_*.csv     # For each metric
│   ├── 07_model_level_matrix_*.csv
│   ├── 08_noise_level_matrix_*.csv
│   ├── 10_robustness_analysis.csv
│   ├── by_metric/
│   │   ├── summary_iou.csv
│   │   ├── summary_dice.csv
│   │   └── ...
│   └── statistics_manifest.csv
├── visualizations/
│   ├── schematics/
│   │   └── prompt_modes_illustration_point_bbox_pointbox.pdf
│   ├── busi/
│   │   ├── busi_iou_lineplot_by_level_point.pdf
│   │   ├── busi_iou_heatmap_model_vs_noise.pdf
│   │   ├── busi_iou_ranking_models.pdf
│   │   ├── busi_noise_gallery_all_types_L0_to_L9.pdf
│   │   └── ...
│   └── montgomery/
│       └── (same structure)
├── artifacts/
│   ├── _shared/
│   │   └── Montgomery/gaussian/L1/seed42/image001_original.png
│   │   └── Montgomery/gaussian/L1/seed42/image001_noisy.png
│   │   └── Montgomery/gaussian/L1/seed42/image001_gt.png
│   └── Montgomery/MedSAM/prompt_point/gaussian/L1/seed42/image001_pred.png
└── noise_cache/
    └── Montgomery/gaussian/L1/seed42/image001_256x256.npy
```

### CSV Columns (statistics_merged.csv)

```
dataset, model, prompt_mode, noise_type, noise_level, gt_empty_rate, n_gt_non_empty,
pred_empty_rate, Dice, Dice_std, IoU, IoU_std, Recall, Recall_std, Precision,
Precision_std, F1, F1_std, HD, HD_std, n_images, n_rows, source_stats_file
```

---

## 📈 Visualization & Statistics

### Generated Statistics (31 CSV files)

1. **01_overall_summary.csv**: Global metrics (mean, std, min, max, drop)
2. **02_model_summary.csv**: Per-model performance & rankings
3. **03_mode_summary.csv**: Per-prompt-mode (point/bbox/point+bbox)
4. **04_noise_summary.csv**: Per-noise-type difficulty
5. **05_level_summary.csv**: Per-level (L0-L9) aggregates
6. **06_model_noise_matrix_*.csv**: Heatmaps (8 models × 12 noises, per metric)
7. **07_model_level_matrix_*.csv**: Heatmaps (8 models × 10 levels)
8. **08_noise_level_matrix_*.csv**: Heatmaps (12 noises × 10 levels)
9. **10_robustness_analysis.csv**: Stability ranks, AUC, degradation slope
10. **by_metric/*.csv**: Detailed per-metric summaries (6 metrics)

### Generated Visualizations (99 PDF files)

**Per Dataset**: BUSI (49 PDFs) + Montgomery (49 PDFs) + Schematics (1 PDF)

- **Line plots** (18 per dataset = 6 metrics × 3 modes): Metric vs. level per noise type
- **Mode comparison** (6 per dataset): Point vs. bbox vs. point+bbox overlay
- **Heatmaps** (12 per dataset = 6 metrics × 2 types):
  - Model × Noise (noise difficulty heatmap)
  - Model × Level (robustness across severity)
- **Rankings** (12 per dataset = 6 metrics × 2 types):
  - Model ranking (by robustness)
  - Noise ranking (by difficulty)
- **Noise gallery** (1 per dataset): Visual examples of all noise types, L0-L9

**Features**:
✅ No titles → descriptive filenames
✅ All levels L0-L9 preserved
✅ HD metric correctly handled (lower is better)
✅ PDF export for publication

---

## 🐛 Known Issues & Fixes

### Fixed Bugs (v1.1)

#### 1. Speckle Noise Not Applied (CRITICAL)

**Issue**: `configs/full_benchmark.yaml` speckle presets missing `p: 1`
**Symptom**: Speckle noise completely skipped (always p=0)
**Fix**: Added `p: 1` to all speckle L1-L9 levels

```yaml
speckle:
  L1: {p: 1, sigma: 0.05}  # ← Added p: 1
  L2: {p: 1, sigma: 0.1}
```

#### 2. Low Brightness L1 = No Change

**Issue**: `low_brightness` L1 had `factor: 1` (multiply by 1 = no change)
**Symptom**: L1 identical to L0
**Fix**: Changed L1 to `factor: 0.98`

#### 3. Low Contrast L1 = No Change

**Issue**: `low_contrast` L1 had `alpha: 1` (no contrast reduction)
**Symptom**: L1 identical to L0
**Fix**: Changed L1 to `alpha: 0.98`

#### 4. Precision = 1.0 for Empty Predictions

**Issue**: When model predicts empty (TP=0, FP=0), precision returned 1.0
**Symptom**: Misleading perfect precision when no predictions made
**Fix**: Changed to return 0.0 (no positive predictions = 0% precision)

```python
# metrics/metric_manager.py:precision_score()
denom = c["tp"] + c["fp"]
return 0.0 if denom == 0 else float(c["tp"] / denom)  # ← Was 1.0
```

### Known Limitations

⚠️ **Hausdorff Distance on small structures**: HD can be unstable with <5 pixel objects
⚠️ **Memory**: Full benchmark needs 24+ GB VRAM; use `--max_samples` to test
⚠️ **CAMUS dataset**: Required changes to multi-frame handling for 2D benchmark

---

## 🔍 Results Interpretation

### Robustness Metrics

**Relative Drop %**: How much metric degrades from clean (L0) to noisy
```
drop = (clean_val - noisy_val) / clean_val × 100%
```

- **Low drop** (< 5%): Model robust to noise
- **High drop** (> 20%): Model sensitive to noise

**Degradation Slope**: Rate of performance decay across L0→L9

- **Gentle slope**: Graceful degradation
- **Steep slope**: Sudden failure at high levels

**AUC Robustness**: Integrated robustness across all levels (trapezoid rule)

- **Higher AUC**: More stable across levels

### Ranking Interpretation

**Model Ranking** (by robustness across all conditions):
- Rank 1 = Most robust (lowest drop, gentlest slope)
- Rank 8 = Least robust

**Noise Ranking** (by difficulty):
- Noise causing largest drop = "most difficult"
- Useful for identifying model weaknesses

### Prompt Mode Comparison

- **Point mode**: Minimal spatial info → model must infer full object
- **BBox mode**: Spatial constraint → easier task
- **Point+BBox mode**: Combined hints → typically easiest

Expected pattern: `Point ≥ Point+BBox ≥ BBox` (in terms of difficulty)

---

## 🔧 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution 1**: Use sample limiting
```bash
python main.py --config configs/full_benchmark.yaml --stage run --max_samples 50
```

**Solution 2**: Enable noise caching & GC
```yaml
stage1:
  cache_noisy_images: true
  gc_collect_interval: 10
  cuda_cache_clear_interval: 20
```

**Solution 3**: Run per-dataset
```bash
python main.py --config configs/full_benchmark.yaml --stage run --datasets Montgomery
```

### Issue: Model Initialization Fails

**Check**: Checkpoint path exists and format matches runner
**Fix**: Verify weight file path in config and filename extension

```bash
ls -la weights/sam_b.pt  # Should exist
```

### Issue: CUDA/GPU Not Found

**Check**: `device` setting in config
```yaml
device: "cuda"  # or "cpu" to force CPU
```

**Verify**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: CSV Missing All L0-L9

**Check**: Ensure all levels L0-L9 defined in config:
```yaml
levels: [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]
```

**Verify**:
```bash
grep "noise_level" outputs/*/statistics_merged.csv | cut -d, -f5 | sort -u
# Should show: L0, L1, L2, ..., L9
```

### Issue: Metric NaN or Inf Values

**Common causes**:
- Empty GT mask → IoU/Dice/Recall return NaN
- Empty prediction + non-empty GT → Precision returns 0.0
- Empty prediction + empty GT → HD returns NaN (mathematically undefined)

**Check**: `gt_empty_rate` and `pred_empty_rate` columns in CSV

---

## 📚 References

### Paper

> [Your paper citation here]

### Noise Models

- **Gaussian**: Standard additive noise
- **Poisson/Shot**: Foi et al. (2008), Makitalo & Foi (2011) - photon counting
- **Rician**: Gudbjartsson & Patz (1995) - MRI magnitude estimation
- **Speckle**: Multiplicative noise in radar/ultrasound

### SAM Variants

- SAM: Kirillov et al. (2023)
- SAM2: Ravi et al. (2024)
- SAM3: Academic release
- MedSAM: Wang et al. (2023)
- SAM-Med2D: Specificity adapted

## 📝 Citation

If you use this benchmark, please cite:

```bibtex
@article{sam-noisy-medical,
  title={Robustness Benchmark for Medical Image Segmentation under Controlled Noise},
  author={...},
  journal={...},
  year={2026}
}
```

---

**Last Updated**: 2026-03-26
**Benchmark Status**: ✅ Complete (Statistics + Visualizations Generated)
