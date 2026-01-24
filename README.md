# SAM Benchmark under Noisy Medical Radiology Imaging Conditions

A comprehensive benchmarking framework for evaluating **SAM-based models** (SAM, SAM2, MedSAM) on 2D medical imaging datasets under various noise and artifact conditions.

**Extended for AIO25 NoisySAM Project** with noise intensity tracking, uncertainty metrics, prediction caching, and comprehensive PDF reports.

## 🎯 Objectives

- Evaluate **robustness** of SAM variants under simulated acquisition artifacts
- Measure **stability** metrics (performance drop, slope, AUC, coefficient of variation)
- Track **noise intensity** with PSNR/SSIM quality metrics
- Compute **uncertainty** metrics (confidence, entropy) for predictions
- Identify **failure modes** across different noise types and severity levels
- Generate comprehensive **PDF reports** with noise galleries and global sensitivity plots

## ✨ New Features (AIO25 Extension)

- **Noise Intensity Tracking**: PSNR/SSIM computed for each noisy image
- **Extended Stability Metrics**: drop_Lmax, slope, AUC, CV, seed stability
- **Uncertainty Analysis**: Mean confidence, entropy, boundary entropy
- **Prediction Caching**: Skip repeated inferences with `--use_cache`
- **Debug Mode**: Quick testing with `--debug --max_samples N`
- **Report-Only Mode**: Regenerate reports without re-running inference
- **Noise Gallery**: Visual comparison across noise types and levels
- **Global Sensitivity Plots**: Heatmaps and ranking charts

## 📁 Project Structure

```
project1/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── configs/
│   ├── phase1.yaml        # Phase 1: Controlled analysis (2 datasets)
│   └── phase2.yaml        # Phase 2: Extended (4-5 datasets)
├── datasets/              # Dataset adapters
│   ├── base.py           # BaseDatasetAdapter ABC
│   ├── image_mask_folder.py  # Generic folder adapter
│   └── registry.py       # Dataset registry
├── noises/                # Noise injection modules
│   ├── base.py           # NoiseBase ABC + NoiseResult dataclass
│   ├── gaussian.py       # Gaussian noise
│   ├── poisson.py        # Poisson noise
│   ├── salt_pepper.py    # Salt & pepper noise
│   ├── motion_blur.py    # Motion blur
│   ├── bias_field.py     # Intensity inhomogeneity
│   ├── low_contrast.py   # Low contrast degradation
│   ├── optional_extras.py # Phase 2 optional noises
│   ├── presets.py        # Noise level presets
│   └── registry.py       # Noise registry with metadata
├── model/                 # Model runners
│   ├── base.py           # BaseModelRunner ABC
│   ├── sam1.py           # SAM (original) runner
│   ├── sam2.py           # SAM2 runner
│   ├── sam3.py           # SAM3 placeholder
│   ├── medsam.py         # MedSAM runner
│   ├── prompts.py        # Prompt generation utilities
│   └── registry.py       # Model registry
├── metrics/               # Evaluation metrics
│   ├── seg.py            # Dice, IoU, HD95
│   └── stability.py      # PerfDrop, MaskConsistency, AUC
├── viz/                   # Visualization
│   ├── overlays.py       # Mask overlay utilities
│   ├── grids.py          # Comparison grids
│   ├── plots.py          # Metric plots
│   └── failure_cases.py  # Failure analysis
├── reports/               # PDF report generation
│   └── pdf_builder.py    # ReportLab-based PDF builder
├── runner/                # Experiment orchestration
│   ├── experiment.py     # Main experiment runner
│   ├── protocols.py      # Protocol case builder
│   ├── aggregate.py      # Results aggregation
│   ├── io_utils.py       # I/O utilities
│   └── config_schema.py  # Config validation
└── weight/                # Model checkpoints
    ├── sam_vit_b_01ec64.pth
    ├── sam_vit_l_0b3195.pth
    ├── sam_vit_h_4b8939.pth
    ├── sam2_t.pt
    └── sam2_s.pt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Install PyTorch (select your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional: Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 2. Prepare Dataset

Organize your dataset as:
```
your_dataset/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── masks/
    ├── img001.png
    ├── img002.png
    └── ...
```

### 3. Configure and Run

```bash
# Full benchmark run
python main.py --config configs/phase1.yaml

# Quick debug with caching (recommended for testing)
python main.py --config configs/phase1.yaml --debug --max_samples 5

# Run with prediction caching (skip repeated inferences)
python main.py --config configs/phase1.yaml --use_cache

# Regenerate report from existing results (no inference)
python main.py --config configs/phase1.yaml --only_report

# Run specific noise types only
python main.py --config configs/phase1.yaml --max_noise gaussian,poisson --use_cache

# Run specific levels only
python main.py --config configs/phase1.yaml --max_level L1,L2 --use_cache

# Clear cache and rerun
python main.py --config configs/phase1.yaml --clear_cache --use_cache

# Phase 2 with all noises and grid search
python main.py --config configs/phase2.yaml --use_cache
```

## 🔧 CLI Options

| Flag | Description |
|------|-------------|
| `--config` | Path to config file (required) |
| `--phase` | Override phase (1 or 2) |
| `--dry_run` | Print protocol cases without running |
| `--use_cache` | Enable prediction caching |
| `--clear_cache` | Clear existing cache before running |
| `--only_report` | Skip inference, generate report from results.csv |
| `--max_samples` | Limit samples per protocol case |
| `--max_noise` | Comma-separated noise types (e.g., gaussian,poisson) |
| `--max_level` | Comma-separated levels (e.g., L1,L2) |
| `--max_models` | Limit number of models |
| `--debug` | Enable debug mode (implies --use_cache) |
| `--verbose` | Enable verbose output |

## 📊 Protocols

| Protocol | Description |
|----------|-------------|
| **P0** | Clean baseline (L0, no noise) |
| **P1** | Coupled levels - p and severity increase together (L1→L4) |
| **P2a** | OFAT - sweep severity with fixed probability |
| **P2b** | OFAT - sweep probability with fixed severity |
| **P3** | Grid search over p × severity combinations |

## 🔊 Noise Types

### Phase 1 (Required)
- Gaussian noise
- Poisson noise
- Salt & pepper noise
- Motion blur
- Bias field (intensity inhomogeneity)
- Low contrast degradation

### Phase 2 (Optional)
- Speckle noise
- Uniform noise
- JPEG compression artifacts
- Quantization noise
- Defocus blur
- Coarse dropout
- GridMask

## 📈 Metrics

### Segmentation Metrics
| Metric | Description |
|--------|-------------|
| **Dice** | Dice coefficient (F1 score) |
| **IoU** | Intersection over Union (Jaccard) |
| **HD95** | 95th percentile Hausdorff distance |

### Stability Metrics
| Metric | Description |
|--------|-------------|
| **drop_Lmax** | Dice(L0) - Dice(L4) — performance drop at maximum severity |
| **slope** | Linear regression slope of Dice vs intensity_scalar |
| **AUC** | Area under curve (normalized) — higher is better |
| **CV** | Coefficient of variation (std/mean) across levels |
| **seed_std** | Standard deviation across noise seeds |
| **seed_cv** | Coefficient of variation across noise seeds |
| **MaskConsistency** | IoU(mask_L0, mask_Lk) |

### Noise Quality Metrics
| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | Structural Similarity Index |
| **intensity_scalar** | Normalized severity (0.0 to 1.0) |

### Uncertainty Metrics
| Metric | Description |
|--------|-------------|
| **mean_confidence** | Average prediction confidence |
| **mean_entropy** | Average prediction entropy |
| **boundary_entropy** | Entropy at mask boundaries |

## 📄 Output Files

After running an experiment, outputs are saved to `outputs/<exp_name>/`:

```
outputs/<exp_name>/
├── report.pdf           # Comprehensive PDF report with all sections
├── preview.pdf          # Visual samples (clean vs noisy)
├── results.csv          # Per-sample results with all metrics
├── aggregate.csv        # Aggregated by group
├── stability.csv        # Extended stability metrics
├── summary.json         # Experiment summary
├── figures/             # All generated plots
│   ├── dice_vs_level__*.png
│   ├── OFAT_P2a__*.png
│   ├── global_sensitivity_heatmap__*.png
│   ├── sensitivity_curves__*.png
│   ├── impact_ranking__*.png
│   ├── summary_heatmap_*.png
│   └── noise_gallery/
│       ├── noise_gallery_*.png
│       └── noise_gallery_summary.png
├── failure_cases/       # Top failure visualizations
├── pred_masks/          # Saved predictions
├── meta/                # Config snapshots
│   ├── config_snapshot.json
│   └── config_snapshot.yaml
└── cache/               # Prediction cache (if --use_cache)
```

## ⚙️ Configuration

### Key Config Sections

```yaml
# Experiment info
exp:
  name: "my_experiment"
  out_root: "outputs"

# Device and reproducibility
device: "cuda"
seed: 42
verbose: true

# Cache configuration (for efficient re-runs)
cache:
  enabled: false          # Enable via --use_cache
  cache_dir: null         # Default: cache/
  clear_on_start: false

# Debug configuration (for quick testing)
debug:
  enabled: false
  max_samples: null       # Limit samples per case
  noise_types: null       # Filter noise types
  levels: null            # Filter levels

# Datasets (supports multiple)
datasets:
  - name: "my_dataset"
    adapter: "ImageMaskFolderAdapter"
    root: "/path/to/dataset"
    image_dir: "images"
    mask_dir: "masks"

# Models (multi-model + multi-weights)
models:
  - name: "SAM"
    runner: "SAM1"
    mode: ["prompt_bbox", "automatic"]
    weights:
      - id: "sam_b"
        checkpoint: "weight/sam_vit_b.pth"
        model_type: "vit_b"

# Noise levels with intensity scalars
levels:
  names: ["L0", "L1", "L2", "L3", "L4"]
  intensity_scalars:
    L0: 0.0
    L1: 0.25
    L2: 0.50
    L3: 0.75
    L4: 1.0

# Noise seed configuration
noise_config:
  n_noise_seeds: 1        # Increase for seed stability analysis
  base_seed: 42

# Protocols
protocols:
  enabled: ["P0", "P1", "P2"]
  coupled_presets:
    gaussian: {L1: {p: 0.3, sigma: 5}, ...}

# Output settings
outputs:
  save_pred_masks: true
  num_preview_samples: 8
  num_gallery_samples: 3
  num_failure_cases: 8
  generate_global_plots: true
```

## 🤝 Extending

### Add New Dataset Adapter

```python
# datasets/my_adapter.py
from datasets.base import BaseDatasetAdapter

class MyAdapter(BaseDatasetAdapter):
    def __init__(self, cfg: dict):
        # Initialize from config
        pass
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "id": "sample_id",
            "image": np.ndarray,  # uint8 HxW
            "gt_mask": np.ndarray,  # uint8 HxW {0,1}
            "meta": {}
        }

# Register in datasets/registry.py
```

### Add New Noise Type

```python
# noises/my_noise.py
from noises.base import NoiseBase, NoiseResult

class MyNoise(NoiseBase):
    # Define parameter ranges for intensity computation
    PARAM_RANGES = {
        "my_param": (0.0, 100.0)  # (min, max) for normalization
    }
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        # Apply noise transformation
        return noisy_image

# Register in noises/registry.py
# Add presets in noises/presets.py
```

### Add New Model

```python
# model/my_model.py
from model.base import BaseModelRunner

class MyModelRunner(BaseModelRunner):
    def __init__(self, weight_cfg: dict, mode: str, device: str):
        # Initialize model
        pass
    
    def predict(self, image_gray, gt_mask, meta) -> Tuple[np.ndarray, dict]:
        # Return prediction and extras
        return pred_mask, {"pred_iou_score": 0.95}

# Register in model/registry.py
```

## 📝 License

MIT License

## 📚 References

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
