#!/usr/bin/env bash
set -euo pipefail

# Run Stage-1 in two conda envs to avoid dependency conflicts:
# - env sam: all non-UltraSAM models
# - env sam1: UltraSAM only
# Then aggregate all raw CSV files together.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-configs/full_benchmark.yaml}"
DATASETS="${2:-}"
MAX_SAMPLES="${3:-}"
ENV_MAIN="${4:-sam}"
ENV_ULTRA="${5:-sam1}"

COMMON_ARGS=(--config "$CFG" --stage run)
if [[ -n "$DATASETS" ]]; then
  COMMON_ARGS+=(--datasets "$DATASETS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

NON_ULTRA_MODELS="SAM,SAM2,SAM3,MedSAM,MedSAM2,MedSAM3,SAM-Med2D"
ULTRA_MODELS="UltraSAM"

echo "[check] Validate conda envs and interpreters"
conda run -n "$ENV_MAIN" python -c "import sys; print('ENV:', '$ENV_MAIN', 'PY:', sys.executable)"
conda run -n "$ENV_ULTRA" python -c "import sys; print('ENV:', '$ENV_ULTRA', 'PY:', sys.executable)"

echo "[1/3] Run non-UltraSAM models in env '$ENV_MAIN'"
PYTHONPATH="$ROOT_DIR" conda run -n "$ENV_MAIN" \
  python "$ROOT_DIR/main.py" "${COMMON_ARGS[@]}" --models "$NON_ULTRA_MODELS"

echo "[2/3] Run UltraSAM model in env '$ENV_ULTRA'"
PYTHONPATH="$ROOT_DIR" conda run -n "$ENV_ULTRA" \
  python "$ROOT_DIR/main.py" "${COMMON_ARGS[@]}" --models "$ULTRA_MODELS"

echo "[3/3] Aggregate all raw CSV files"
PYTHONPATH="$ROOT_DIR" conda run -n "$ENV_MAIN" \
  python "$ROOT_DIR/main.py" --config "$CFG" --stage aggregate

echo "Done. Combined stats are in outputs/<exp_name>/statistics_merged.csv"
