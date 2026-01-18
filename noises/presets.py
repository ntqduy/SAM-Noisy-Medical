"""
Noise presets for systematic level variation (L0-L4).
Provides coupled presets and OFAT (One-Factor-At-a-Time) configurations.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class NoisePreset:
    """Single noise preset configuration."""
    noise_name: str
    level: str
    p: float
    params: Dict[str, Any]


# ============================================================================
# DEFAULT COUPLED PRESETS (Phase 1 required noises)
# L0: Clean baseline (no noise)
# L1-L4: Mild to Severe with increasing p and severity
# ============================================================================

DEFAULT_COUPLED_PRESETS = {
    # Gaussian noise: sigma controls intensity
    "gaussian": {
        "L1": {"p": 0.3, "sigma": 5},
        "L2": {"p": 0.6, "sigma": 10},
        "L3": {"p": 0.8, "sigma": 18},
        "L4": {"p": 1.0, "sigma": 28},
    },
    
    # Poisson noise: lam controls intensity (higher = more noise)
    "poisson": {
        "L1": {"p": 0.3, "lam": 8},
        "L2": {"p": 0.6, "lam": 15},
        "L3": {"p": 0.8, "lam": 25},
        "L4": {"p": 1.0, "lam": 40},
    },
    
    # Salt & pepper: amount controls density
    "salt_pepper": {
        "L1": {"p": 0.3, "amount": 0.005},
        "L2": {"p": 0.6, "amount": 0.01},
        "L3": {"p": 0.8, "amount": 0.02},
        "L4": {"p": 1.0, "amount": 0.04},
    },
    
    # Motion blur: k=kernel size, angle=direction
    "motion_blur": {
        "L1": {"p": 0.3, "k": 5, "angle": 10},
        "L2": {"p": 0.6, "k": 9, "angle": 15},
        "L3": {"p": 0.8, "k": 13, "angle": 20},
        "L4": {"p": 1.0, "k": 17, "angle": 25},
    },
    
    # Bias field (intensity inhomogeneity): strength, smooth
    "bias_field": {
        "L1": {"p": 0.3, "strength": 0.2, "smooth": 48},
        "L2": {"p": 0.6, "strength": 0.35, "smooth": 64},
        "L3": {"p": 0.8, "strength": 0.55, "smooth": 96},
        "L4": {"p": 1.0, "strength": 0.8, "smooth": 128},
    },
    
    # Low contrast: alpha < 1 reduces contrast
    "low_contrast": {
        "L1": {"p": 0.3, "alpha": 0.85, "beta": 0.0},
        "L2": {"p": 0.6, "alpha": 0.75, "beta": 0.0},
        "L3": {"p": 0.8, "alpha": 0.65, "beta": 0.0},
        "L4": {"p": 1.0, "alpha": 0.50, "beta": 0.0},
    },
}


# ============================================================================
# PHASE 2 OPTIONAL PRESETS
# ============================================================================

PHASE2_OPTIONAL_PRESETS = {
    # Speckle noise: multiplicative noise
    "speckle": {
        "L1": {"p": 0.3, "sigma": 0.04},
        "L2": {"p": 0.6, "sigma": 0.08},
        "L3": {"p": 0.8, "sigma": 0.12},
        "L4": {"p": 1.0, "sigma": 0.18},
    },
    
    # Uniform noise: additive uniform
    "uniform": {
        "L1": {"p": 0.3, "a": -5, "b": 5},
        "L2": {"p": 0.6, "a": -10, "b": 10},
        "L3": {"p": 0.8, "a": -18, "b": 18},
        "L4": {"p": 1.0, "a": -25, "b": 25},
    },
    
    # JPEG compression artifacts
    "jpeg": {
        "L1": {"p": 0.3, "quality": 70},
        "L2": {"p": 0.6, "quality": 50},
        "L3": {"p": 0.8, "quality": 30},
        "L4": {"p": 1.0, "quality": 15},
    },
    
    # Quantization noise
    "quantization": {
        "L1": {"p": 0.3, "step": 4},
        "L2": {"p": 0.6, "step": 8},
        "L3": {"p": 0.8, "step": 16},
        "L4": {"p": 1.0, "step": 32},
    },
    
    # Defocus blur
    "defocus_blur": {
        "L1": {"p": 0.3, "k": 3},
        "L2": {"p": 0.6, "k": 5},
        "L3": {"p": 0.8, "k": 9},
        "L4": {"p": 1.0, "k": 13},
    },
    
    # Coarse dropout
    "coarse_dropout": {
        "L1": {"p": 0.3, "holes": 4, "size": 16},
        "L2": {"p": 0.6, "holes": 8, "size": 24},
        "L3": {"p": 0.8, "holes": 12, "size": 32},
        "L4": {"p": 1.0, "holes": 16, "size": 48},
    },
    
    # GridMask
    "gridmask": {
        "L1": {"p": 0.3, "d": 64, "r": 16},
        "L2": {"p": 0.6, "d": 48, "r": 16},
        "L3": {"p": 0.8, "d": 32, "r": 16},
        "L4": {"p": 1.0, "d": 24, "r": 12},
    },
}


def get_all_presets(include_phase2: bool = False) -> Dict[str, Dict]:
    """Get all noise presets."""
    presets = dict(DEFAULT_COUPLED_PRESETS)
    if include_phase2:
        presets.update(PHASE2_OPTIONAL_PRESETS)
    return presets


def get_preset_for_level(noise_name: str, level: str, presets: Optional[Dict] = None) -> Optional[NoisePreset]:
    """
    Get preset for a specific noise and level.
    
    Args:
        noise_name: Name of noise type
        level: Level string (L0-L4)
        presets: Custom presets dict or None for defaults
        
    Returns:
        NoisePreset or None if not found
    """
    if presets is None:
        presets = get_all_presets(include_phase2=True)
    
    if level == "L0":
        return NoisePreset(noise_name="clean", level="L0", p=0.0, params={})
    
    if noise_name not in presets:
        return None
    
    noise_levels = presets[noise_name]
    if level not in noise_levels:
        return None
    
    cfg = dict(noise_levels[level])
    p = float(cfg.pop("p", 1.0))
    
    return NoisePreset(noise_name=noise_name, level=level, p=p, params=cfg)


def get_severity_params_at_level(noise_name: str, level: str, presets: Optional[Dict] = None) -> Dict[str, Any]:
    """Get severity parameters for a noise at specific level (without p)."""
    preset = get_preset_for_level(noise_name, level, presets)
    if preset is None:
        return {}
    return dict(preset.params)


def get_p_at_level(noise_name: str, level: str, presets: Optional[Dict] = None) -> float:
    """Get probability for a noise at specific level."""
    preset = get_preset_for_level(noise_name, level, presets)
    if preset is None:
        return 0.0
    return preset.p


# ============================================================================
# MIXED NOISE SETS (Phase 2 optional)
# ============================================================================

MIXED_NOISE_SETS = {
    "acquisition_combo": ["gaussian", "motion_blur"],
    "medical_artifacts": ["bias_field", "low_contrast", "poisson"],
    "compression_artifacts": ["jpeg", "quantization"],
    "heavy_degradation": ["gaussian", "motion_blur", "low_contrast"],
}


def get_mixed_noise_set(set_name: str) -> List[str]:
    """Get list of noise names for a mixed set."""
    return list(MIXED_NOISE_SETS.get(set_name, []))


def build_mixed_preset(set_name: str, level: str, presets: Optional[Dict] = None) -> List[NoisePreset]:
    """
    Build list of presets for a mixed noise set at given level.
    
    Args:
        set_name: Name of mixed noise set
        level: Level string (L1-L4)
        presets: Custom presets dict
        
    Returns:
        List of NoisePreset objects
    """
    noise_names = get_mixed_noise_set(set_name)
    result = []
    for name in noise_names:
        preset = get_preset_for_level(name, level, presets)
        if preset is not None:
            result.append(preset)
    return result
