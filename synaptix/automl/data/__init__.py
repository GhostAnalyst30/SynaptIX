"""Capa de datos del motor AutoML: entrada, detección y perfilado."""

from .adapters import (
    LoadedData,
    input_kind,
    load_data,
    polars_available,
    resolve_engine,
    use_polars_profiling,
)
from .detection import TaskDetection, detect_task, detect_task_details, infer_feature_types
from .intelligence import DataProfile, DataWarning, leakage_checks, profile_data

__all__ = [
    "LoadedData",
    "input_kind",
    "load_data",
    "polars_available",
    "resolve_engine",
    "use_polars_profiling",
    "TaskDetection",
    "detect_task",
    "detect_task_details",
    "infer_feature_types",
    "DataProfile",
    "DataWarning",
    "leakage_checks",
    "profile_data",
]
