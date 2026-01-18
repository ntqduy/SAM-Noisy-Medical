# SAM Benchmark under Noisy Medical Radiology Imaging Conditions

A comprehensive benchmarking framework for evaluating **SAM-based models** (SAM, SAM2, MedSAM) on 2D medical imaging datasets under various noise and artifact conditions.

## 🎯 Objectives

- Evaluate **robustness** of SAM variants under simulated acquisition artifacts
- Measure **stability** metrics (performance drop, mask consistency)
- Identify **failure modes** across different noise types and severity levels
- Generate comprehensive **PDF reports** with visualizations and analysis

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
│   ├── base.py           # NoiseBase ABC
│   ├── gaussian.py       # Gaussian noise
│   ├── poisson.py        # Poisson noise
│   ├── salt_pepper.py    # Salt & pepper noise
│   ├── motion_blur.py    # Motion blur
│   ├── bias_field.py     # Intensity inhomogeneity
│   ├── low_contrast.py   # Low contrast degradation
│   ├── optional_extras.py # Phase 2 optional noises
│   ├── presets.py        # Noise level presets
│   └── registry.py       # Noise registry
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
# Phase 1: Controlled analysis
python main.py --config configs/phase1.yaml

# Phase 2: Extended multi-modality
python main.py --config configs/phase2.yaml

# Debug mode (limited samples)
python main.py --config configs/phase1.yaml --limit_n 10 --dry_run
```

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

| Metric | Description |
|--------|-------------|
| **Dice** | Dice coefficient (F1 score) |
| **IoU** | Intersection over Union (Jaccard) |
| **HD95** | 95th percentile Hausdorff distance |
| **PerfDrop** | Dice(L0) - Dice(L4) |
| **MaskConsistency** | IoU(mask_L0, mask_Lk) |

## 📄 Output Files

After running an experiment, outputs are saved to `outputs/<exp_name>/`:

```
outputs/<exp_name>/
├── preview.pdf          # Visual samples (clean vs noisy)
├── report.pdf           # Comprehensive PDF report
├── results.csv          # Per-sample results
├── aggregate.csv        # Aggregated by group
├── stability.csv        # Stability metrics
├── summary.json         # Experiment summary
├── figures/             # All generated plots
│   ├── dice_vs_level__*.png
│   ├── OFAT_P2a__*.png
│   └── ...
├── failure_cases/       # Top failure visualizations
└── pred_masks/          # Saved predictions
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

# Protocols
protocols:
  enabled: ["P0", "P1", "P2"]
  coupled_presets:
    gaussian: {L1: {p: 0.3, sigma: 5}, ...}
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
from noises.base import NoiseBase

class MyNoise(NoiseBase):
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
