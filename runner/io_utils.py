import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(path: Path, s: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def get_device(device: str) -> str:
    device = (device or "cpu").lower()
    if device.startswith("cuda"):
        try:
            import torch  # optional
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def env_seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
