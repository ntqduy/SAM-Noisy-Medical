#!/usr/bin/env python3
"""Run strict real-inference validation across models x prompt modes.

- Uses conda env `sam1` for UltraSAM, `sam` for all others.
- Uses tiny benchmark slice (BUSI, gaussian L0, max_samples=1) for fast real checks.
- Records per-run status, error tail, and basic metrics from raw CSV output.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


REPO = Path("/home/ubuntu/SAM-Noisy-Medical")
BASE_CFG = REPO / "configs" / "full_benchmark.yaml"
TMP_DIR = REPO / "outputs" / "real_mode_matrix" / "tmp_configs"
REPORT_DIR = REPO / "outputs" / "real_mode_matrix"

MODES = ["prompt_point", "prompt_bbox", "prompt_point_box"]
MODEL_ENVS = {
    "SAM": "sam",
    "SAM2": "sam",
    "SAM3": "sam",
    "MedSAM": "sam",
    "MedSAM2": "sam",
    "MedSAM3": "sam",
    "SAM-Med2D": "sam",
    "UltraSAM": "sam1",
}


def _tail(text: str, n: int = 25) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _load_base_cfg() -> dict:
    with BASE_CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_dataset(cfg: dict, name: str) -> dict:
    for ds in cfg.get("datasets", []):
        if str(ds.get("name")) == name:
            return copy.deepcopy(ds)
    raise RuntimeError(f"Dataset {name!r} not found in {BASE_CFG}")


def _pick_model(cfg: dict, name: str) -> dict:
    for m in cfg.get("models", []):
        if str(m.get("name")) == name:
            return copy.deepcopy(m)
    raise RuntimeError(f"Model {name!r} not found in {BASE_CFG}")


def _write_cfg(model_name: str, mode: str, stamp: str) -> Path:
    cfg = _load_base_cfg()

    run_name = f"real_{model_name}_{mode}_{stamp}".replace("-", "_")
    cfg["exp"] = {
        "name": run_name,
        "out_root": "outputs/real_mode_matrix/runs",
    }

    cfg["datasets"] = [_pick_dataset(cfg, "BUSI")]

    model_cfg = _pick_model(cfg, model_name)
    model_cfg["prompt_modes"] = [mode]
    cfg["models"] = [model_cfg]

    cfg["noises"] = ["gaussian"]
    cfg["levels"] = ["L0"]
    cfg.setdefault("noise_config", {})
    cfg["noise_config"]["n_noise_seeds"] = 1
    cfg.setdefault("stage1", {})
    cfg["stage1"]["save_artifacts"] = False
    cfg["stage1"]["artifact_samples_per_case"] = 0

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / f"{run_name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _extract_metrics(exp_dir: Path) -> Dict[str, float]:
    manifest = exp_dir / "raw_files_manifest.csv"
    if not manifest.exists():
        return {}

    with manifest.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    raw_csv = Path(rows[0]["raw_csv"])
    if not raw_csv.exists():
        return {}

    data = []
    with raw_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data.append(row)
    if not data:
        return {"n_rows": 0}

    def mean(col: str) -> float:
        vals = [float(r[col]) for r in data if r.get(col) not in (None, "")]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    return {
        "n_rows": len(data),
        "Dice_mean": mean("Dice"),
        "IoU_mean": mean("IoU"),
        "Recall_mean": mean("Recall"),
        "Precision_mean": mean("Precision"),
        "HD_mean": mean("HD"),
    }


def run_one(model: str, mode: str) -> dict:
    env = MODEL_ENVS[model]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cfg_path = _write_cfg(model, mode, stamp)

    cmd = (
        f"cd {REPO} && "
        f"PYTHONPATH={REPO} "
        f"conda run -n {env} python main.py "
        f"--config {cfg_path} --stage run --max_samples 1 --datasets BUSI --models '{model}'"
    )

    proc = subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        executable="/bin/bash",
    )

    run_name = cfg_path.stem
    exp_dir = REPO / "outputs" / "real_mode_matrix" / "runs" / run_name

    result = {
        "model": model,
        "mode": mode,
        "env": env,
        "config": str(cfg_path),
        "run_name": run_name,
        "exit_code": proc.returncode,
        "success": proc.returncode == 0,
        "stdout_tail": _tail(proc.stdout, 30),
        "stderr_tail": _tail(proc.stderr, 30),
        "exp_dir": str(exp_dir),
    }

    if proc.returncode == 0:
        result.update(_extract_metrics(exp_dir))

    return result


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    for model in MODEL_ENVS:
        for mode in MODES:
            print(f"[RUN] model={model} mode={mode} env={MODEL_ENVS[model]}", flush=True)
            res = run_one(model, mode)
            results.append(res)
            status = "OK" if res["success"] else "FAIL"
            print(f"[DONE] {status} model={model} mode={mode} exit={res['exit_code']}", flush=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"real_mode_matrix_report_{ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = REPORT_DIR / f"real_mode_matrix_report_{ts}.csv"
    fieldnames = sorted({k for r in results for k in r.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    n_ok = sum(1 for r in results if r["success"])
    print(f"[SUMMARY] success={n_ok}/{len(results)}")
    print(f"[REPORT] {json_path}")
    print(f"[REPORT] {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
