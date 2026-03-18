"""Core package – experiment engine, model manager, and pipeline orchestration."""

from core.config_manager import ConfigManager
from core.dataset_manager import DatasetManager
from core.model_manager import ModelManager
from core.experiment_engine import ExperimentEngine

__all__ = ["ConfigManager", "DatasetManager", "ModelManager", "ExperimentEngine"]
