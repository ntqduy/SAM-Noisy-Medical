"""
Config schema validation and defaults for SAM benchmark.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ExpConfig:
    """Experiment configuration."""
    name: str
    out_root: str = "outputs"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    adapter: str
    root: str
    image_dir: str = "images"
    mask_dir: str = "masks"
    image_exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    mask_exts: List[str] = field(default_factory=lambda: [".png"])
    mask_type: str = "binary_0_255"
    class_id: int = 1


@dataclass
class WeightConfig:
    """Model weight configuration."""
    id: str
    checkpoint: str
    model_type: str = "vit_b"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    runner: str
    mode: List[str] = field(default_factory=lambda: ["prompt_bbox"])
    weights: List[WeightConfig] = field(default_factory=list)


@dataclass
class InferenceConfig:
    """Inference settings."""
    image_rgb: str = "repeat_gray_3ch"
    prompt_default: str = "BOX_FROM_GT_BBOX"


@dataclass 
class OFATConfig:
    """One-Factor-At-a-Time settings."""
    p_fixed: float = 0.8
    severity_ref_level: str = "L3"


@dataclass
class GridConfig:
    """Grid search settings."""
    enabled: bool = False
    p_values: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    severity_levels: List[str] = field(default_factory=lambda: ["L1", "L2", "L3"])


@dataclass
class ProtocolsConfig:
    """Protocol configuration."""
    enabled: List[str] = field(default_factory=lambda: ["P0", "P1", "P2"])
    coupled_presets: Dict[str, Dict] = field(default_factory=dict)
    ofat: OFATConfig = field(default_factory=OFATConfig)
    grid: GridConfig = field(default_factory=GridConfig)


@dataclass
class OutputsConfig:
    """Output settings."""
    save_pred_masks: bool = True
    num_preview_samples: int = 8
    preview_levels: List[str] = field(default_factory=lambda: ["L0", "L2", "L4"])
    num_failure_cases: int = 8
    figures_dpi: int = 160


# Default level names
DEFAULT_LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Default coupled presets for Phase 1
DEFAULT_COUPLED_PRESETS = {
    "gaussian": {
        "L1": {"p": 0.3, "sigma": 5},
        "L2": {"p": 0.6, "sigma": 10},
        "L3": {"p": 0.8, "sigma": 18},
        "L4": {"p": 1.0, "sigma": 28},
    },
    "poisson": {
        "L1": {"p": 0.3, "lam": 8},
        "L2": {"p": 0.6, "lam": 15},
        "L3": {"p": 0.8, "lam": 25},
        "L4": {"p": 1.0, "lam": 40},
    },
    "salt_pepper": {
        "L1": {"p": 0.3, "amount": 0.005},
        "L2": {"p": 0.6, "amount": 0.01},
        "L3": {"p": 0.8, "amount": 0.02},
        "L4": {"p": 1.0, "amount": 0.04},
    },
    "motion_blur": {
        "L1": {"p": 0.3, "k": 5, "angle": 10},
        "L2": {"p": 0.6, "k": 9, "angle": 15},
        "L3": {"p": 0.8, "k": 13, "angle": 20},
        "L4": {"p": 1.0, "k": 17, "angle": 25},
    },
    "bias_field": {
        "L1": {"p": 0.3, "strength": 0.2, "smooth": 48},
        "L2": {"p": 0.6, "strength": 0.35, "smooth": 64},
        "L3": {"p": 0.8, "strength": 0.55, "smooth": 96},
        "L4": {"p": 1.0, "strength": 0.8, "smooth": 128},
    },
    "low_contrast": {
        "L1": {"p": 0.3, "alpha": 0.85, "beta": 0.0},
        "L2": {"p": 0.6, "alpha": 0.75, "beta": 0.0},
        "L3": {"p": 0.8, "alpha": 0.65, "beta": 0.0},
        "L4": {"p": 1.0, "alpha": 0.50, "beta": 0.0},
    },
}


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill defaults for config dictionary.
    
    Args:
        cfg: Raw config dictionary
        
    Returns:
        Validated config with defaults filled
    """
    # Ensure required sections exist
    if "exp" not in cfg:
        cfg["exp"] = {"name": "default_exp", "out_root": "outputs"}
    
    if "name" not in cfg["exp"]:
        cfg["exp"]["name"] = "default_exp"
    
    if "out_root" not in cfg["exp"]:
        cfg["exp"]["out_root"] = "outputs"
    
    # Defaults
    cfg.setdefault("device", "cuda")
    cfg.setdefault("seed", 42)
    cfg.setdefault("limit_n", 0)
    cfg.setdefault("dry_run", False)
    cfg.setdefault("phase", 1)
    
    # Datasets
    cfg.setdefault("datasets", [])
    
    # Models
    cfg.setdefault("models", [])
    
    # Levels
    if "levels" not in cfg:
        cfg["levels"] = {"names": DEFAULT_LEVELS}
    elif "names" not in cfg["levels"]:
        cfg["levels"]["names"] = DEFAULT_LEVELS
    
    # Protocols
    if "protocols" not in cfg:
        cfg["protocols"] = {
            "enabled": ["P0", "P1", "P2"],
            "coupled_presets": DEFAULT_COUPLED_PRESETS,
            "ofat": {"p_fixed": 0.8, "severity_ref_level": "L3"},
            "grid": {"enabled": False, "p_values": [0.2, 0.5, 0.8], "severity_levels": ["L1", "L2", "L3"]}
        }
    else:
        cfg["protocols"].setdefault("enabled", ["P0", "P1", "P2"])
        cfg["protocols"].setdefault("coupled_presets", DEFAULT_COUPLED_PRESETS)
        cfg["protocols"].setdefault("ofat", {"p_fixed": 0.8, "severity_ref_level": "L3"})
        cfg["protocols"].setdefault("grid", {"enabled": False})
    
    # Inference
    if "inference" not in cfg:
        cfg["inference"] = {
            "image_rgb": "repeat_gray_3ch",
            "prompt": {"default": "BOX_FROM_GT_BBOX"}
        }
    
    # Outputs
    if "outputs" not in cfg:
        cfg["outputs"] = {
            "save_pred_masks": True,
            "num_preview_samples": 8,
            "preview_levels": ["L0", "L2", "L4"],
            "num_failure_cases": 8,
            "figures_dpi": 160
        }
    
    return cfg


def get_default_config() -> Dict[str, Any]:
    """Get a complete default configuration."""
    return validate_config({})
