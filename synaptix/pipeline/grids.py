"""
Registro de modelos y rejillas de hiperparámetros por defecto.

Usados por AutoPipeline y Pipeline para que el usuario no tenga que
elegir modelos ni parámetros; todo es sobreescribible. La fuente única es
:mod:`synaptix.automl.selection` y :mod:`synaptix.automl.tuning`.
"""

from __future__ import annotations

from ..automl.exceptions import AutoMLError
from ..automl.selection.registry import MODE_CANDIDATES, MODEL_REGISTRY, get_spec
from ..automl.tuning.spaces import DEFAULT_GRIDS

# Modelos que se comparan por defecto (modo "balanced").
DEFAULT_CANDIDATES = MODE_CANDIDATES["balanced"]


def resolve_model(task: str, name: str):
    """Devuelve la clase de modelo SynaptIX para un nombre amigable o alias.

    Parameters
    ----------
    task : str
        "classification" o "regression".
    name : str
        Nombre amigable, por ejemplo "random_forest".
    """
    try:
        return get_spec(task, name).wrapper
    except AutoMLError as exc:
        raise ValueError(str(exc)) from exc


__all__ = ["MODEL_REGISTRY", "DEFAULT_CANDIDATES", "DEFAULT_GRIDS", "resolve_model"]
