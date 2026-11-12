"""
synaptix.automl - Motor AutoML modular.

- ``AutoPipeline``  : AutoML completo con ``fit(df, target)``.
- ``data``          : entrada pandas/polars/NumPy, detección de tarea y perfilado.
- ``features``      : preprocesamiento sin fuga (ColumnTransformer por fold).
- ``evaluation``    : métricas y estrategias de validación cruzada.
- ``selection``     : registro de modelos, evaluación y ranking.
- ``tuning``        : grid, random, successive halving y Optuna.
- ``ensemble``      : votación, stacking y blending.
- ``explain``       : importancias y diagnóstico de errores.
- ``experiments``   : registro reproducible (JSON/CSV).
- ``profiling``     : tiempos y memoria por etapa.

Ejemplo
-------
>>> import synaptix as sx
>>> pipe = sx.AutoPipeline(scoring="f1_weighted", tuning="random", n_iter=30)
>>> pipe.fit(df, target="species")
>>> pipe.ranking_
>>> pipe.report(save="automl_report.html")
"""

from .data import DataProfile, detect_task_details, profile_data
from .exceptions import AllModelsFailedError, AutoMLError, MissingDependencyError
from .experiments import ExperimentRecord
from .pipeline import AutoPipeline, detect_task
from .profiling import Profiler
from .selection import list_models

__all__ = [
    "AutoPipeline",
    "detect_task",
    "detect_task_details",
    "profile_data",
    "DataProfile",
    "ExperimentRecord",
    "Profiler",
    "list_models",
    "AutoMLError",
    "AllModelsFailedError",
    "MissingDependencyError",
]
