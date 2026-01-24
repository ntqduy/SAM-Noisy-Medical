"""Experiment runner and utilities.

Extended for AIO25 NoisySAM project with:
  - Extended stability metrics (compute_stability_extended)
  - Prediction caching (PredictionCache)
  - Protocol scheduling (get_schedule)
  - CLI override support
"""
from runner.experiment import run_experiment
from runner.io_utils import (
    load_yaml_config, ensure_dir, save_json, save_yaml_config,
    PredictionCache, compute_cache_key
)
from runner.protocols import build_protocol_cases, ProtocolCase, get_schedule
from runner.aggregate import aggregate_results, compute_stability, compute_stability_extended
from runner.config_schema import validate_config, get_default_config, apply_cli_overrides

__all__ = [
    "run_experiment",
    "load_yaml_config",
    "save_yaml_config",
    "ensure_dir",
    "save_json",
    "build_protocol_cases",
    "ProtocolCase",
    "get_schedule",
    "aggregate_results",
    "compute_stability",
    "compute_stability_extended",
    "validate_config",
    "get_default_config",
    "apply_cli_overrides",
    "PredictionCache",
    "compute_cache_key",
]
