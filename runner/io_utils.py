"""
I/O utilities for AIO25 NoisySAM benchmark.
Includes prediction caching for efficient re-runs and debugging.
"""
import json
import os
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(path: Path, cfg: Dict[str, Any]):
    """Save configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def save_json(path: Path, obj: Any):
    """Save object as JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, s: str):
    """Save text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def get_device(device: str) -> str:
    """Get available device (cuda or cpu)."""
    device = (device or "cpu").lower()
    if device.startswith("cuda"):
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
    return "cpu"


def env_seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ============================================================================
# Prediction Caching System
# ============================================================================

def compute_cache_key(
    dataset: str,
    sample_id: str,
    model_name: str,
    weight_id: str,
    mode: str,
    noise_type: str,
    protocol: str,
    level: str,
    p: float,
    severity_params: Dict,
    noise_seed: int,
    prompt_cfg: Dict = None
) -> str:
    """
    Compute unique cache key for a prediction.
    
    Args:
        dataset: Dataset name
        sample_id: Sample identifier
        model_name: Model name
        weight_id: Weight/checkpoint identifier
        mode: Inference mode (prompt_bbox, automatic, etc.)
        noise_type: Noise type name
        protocol: Protocol (P0/P1/P2a/P2b/P3)
        level: Noise level
        p: Noise probability
        severity_params: Noise parameters
        noise_seed: Noise random seed
        prompt_cfg: Optional prompt configuration
        
    Returns:
        Hash string for cache key
    """
    key_dict = {
        "dataset": dataset,
        "sample_id": sample_id,
        "model": model_name,
        "weight": weight_id,
        "mode": mode,
        "noise_type": noise_type,
        "protocol": protocol,
        "level": level,
        "p": round(p, 4),
        "severity_params": severity_params,
        "noise_seed": noise_seed,
        "prompt_cfg": prompt_cfg or {}
    }
    
    # Create deterministic string representation
    key_str = json.dumps(key_dict, sort_keys=True)
    hash_val = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    return hash_val


def get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Get path for cached prediction."""
    # Use first 2 chars as subdirectory for better filesystem performance
    subdir = cache_key[:2]
    return cache_dir / subdir / f"{cache_key}.pkl"


class PredictionCache:
    """
    Cache for model predictions to avoid redundant inference.
    
    Stores:
      - pred_mask: Binary prediction mask
      - prob_map: Probability/logit map (optional)
      - extra: Additional model outputs
      - metadata: Cache metadata (timestamp, model info, etc.)
    """
    
    def __init__(self, cache_dir: Path, use_cache: bool = True, overwrite: bool = False):
        """
        Initialize prediction cache.
        
        Args:
            cache_dir: Directory for cache files
            use_cache: Whether to use caching
            overwrite: Whether to overwrite existing cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.overwrite = overwrite
        
        if use_cache:
            ensure_dir(self.cache_dir)
    
    def get_key(
        self,
        dataset: str,
        sample_id: str,
        model_name: str,
        weight_id: str,
        mode: str,
        noise_type: str,
        protocol: str,
        level: str,
        p: float,
        severity_params: Dict,
        noise_seed: int,
        prompt_cfg: Dict = None
    ) -> str:
        """Compute cache key."""
        return compute_cache_key(
            dataset, sample_id, model_name, weight_id, mode,
            noise_type, protocol, level, p, severity_params,
            noise_seed, prompt_cfg
        )
    
    def exists(self, cache_key: str) -> bool:
        """Check if cache entry exists."""
        if not self.use_cache:
            return False
        if self.overwrite:
            return False
        return get_cache_path(self.cache_dir, cache_key).exists()
    
    def load(self, cache_key: str) -> Optional[Dict]:
        """
        Load cached prediction.
        
        Returns:
            Dict with 'pred_mask', 'prob_map', 'extra', 'metadata'
            or None if not cached
        """
        if not self.use_cache:
            return None
        
        cache_path = get_cache_path(self.cache_dir, cache_key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"[WARN] Failed to load cache {cache_key}: {e}")
            return None
    
    def save(
        self,
        cache_key: str,
        pred_mask: np.ndarray,
        prob_map: Optional[np.ndarray] = None,
        extra: Dict = None,
        metadata: Dict = None
    ):
        """
        Save prediction to cache.
        
        Args:
            cache_key: Cache key
            pred_mask: Binary prediction mask
            prob_map: Probability map (optional)
            extra: Additional model outputs
            metadata: Cache metadata
        """
        if not self.use_cache:
            return
        
        cache_path = get_cache_path(self.cache_dir, cache_key)
        ensure_dir(cache_path.parent)
        
        data = {
            "pred_mask": pred_mask,
            "prob_map": prob_map,
            "extra": extra or {},
            "metadata": metadata or {}
        }
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"[WARN] Failed to save cache {cache_key}: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            ensure_dir(self.cache_dir)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"count": 0, "size_mb": 0}
        
        count = 0
        size = 0
        for p in self.cache_dir.rglob("*.pkl"):
            count += 1
            size += p.stat().st_size
        
        return {
            "count": count,
            "size_mb": round(size / (1024 * 1024), 2)
        }


def get_command_string(args) -> str:
    """Convert parsed arguments back to command string."""
    import sys
    return " ".join(sys.argv)
