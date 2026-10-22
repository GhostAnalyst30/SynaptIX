"""
Objetos de resultado ligeros de SynaptIX.

Envuelven los resultados existentes (dicts de métricas, tablas de
comparación) sin romper el API actual:

- ``EvaluationResult``      : subclase de ``dict`` — se comporta igual que
  el dict de métricas de siempre, pero con métodos de exportación a ViewX.
- ``CrossValidationResult`` : subclase de ``dict`` para validación cruzada.
- ``ModelComparisonResult`` : subclase de ``pandas.DataFrame`` para la
  tabla de ``compare_models`` (conserva ``to_string``, ``sort_values``, etc.).

Todos incluyen ``to_report_data()`` (sin dependencias extra) y, con ViewX
instalado (``pip install synaptix[viewx]``), ``to_html()`` / ``to_dashboard()``,
``to_report()`` y ``to_presentation()``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .viewx.export import ViewXExportMixin

__all__ = [
    "EvaluationResult",
    "CrossValidationResult",
    "ModelComparisonResult",
]


class EvaluationResult(dict, ViewXExportMixin):
    """
    Métricas de evaluación de un modelo, con exportación a ViewX.

    Es una subclase de ``dict``: el acceso ``result["accuracy"]``,
    ``result.items()``, ``print(result)`` y las comparaciones con dicts
    siguen funcionando exactamente igual que antes.

    Attributes
    ----------
    model_name : str o None
        Nombre del modelo evaluado.
    task : str o None
        ``"classification"``, ``"regression"`` o ``"clustering"``.
    y_true, y_pred : ndarray o None
        Valores reales y predichos (permiten construir figuras).
    y_proba : ndarray o None
        Probabilidades de la clase positiva (clasificación binaria).
    """

    def __init__(
        self,
        metrics: dict,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ):
        super().__init__(metrics)
        self.model_name = model_name
        self.task = task
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba

    def to_dict(self) -> dict:
        """Devuelve las métricas como un ``dict`` plano."""
        return dict(self)


class CrossValidationResult(dict, ViewXExportMixin):
    """
    Resultado de validación cruzada, con exportación a ViewX.

    Subclase de ``dict`` con las claves originales
    (``"scores"``, ``"media"``, ``"desviacion"``).

    Attributes
    ----------
    model_name : str o None
        Nombre del modelo validado.
    scoring : str o None
        Métrica usada (por ejemplo ``"accuracy"`` o ``"r2"``).
    cv : int o None
        Número de folds.
    """

    def __init__(
        self,
        results: dict,
        model_name: Optional[str] = None,
        scoring: Optional[str] = None,
        cv: Optional[int] = None,
    ):
        super().__init__(results)
        self.model_name = model_name
        self.scoring = scoring
        self.cv = cv

    def to_dict(self) -> dict:
        """Devuelve los resultados como un ``dict`` plano."""
        return dict(self)


class ModelComparisonResult(pd.DataFrame, ViewXExportMixin):
    """
    Tabla comparativa de modelos (salida de ``compare_models``).

    Subclase de ``pandas.DataFrame``: conserva todo el API de pandas,
    incluido ``to_html()`` (que sigue devolviendo la tabla HTML de pandas).
    Para el dashboard de ViewX usa :meth:`to_dashboard`; también están
    disponibles ``to_report_data()``, ``to_report()`` y ``to_presentation()``.

    Attributes
    ----------
    task : str o None
        ``"classification"`` o ``"regression"``.
    scoring_name : str o None
        Nombre de la columna de score (``"accuracy"`` o ``"r2"``).
    cv : int o None
        Número de folds usados en la comparación.
    """

    _metadata = ["task", "scoring_name", "cv"]

    @property
    def _constructor(self):
        return ModelComparisonResult

    def __init__(self, *args: Any, **kwargs: Any):
        task = kwargs.pop("task", None)
        scoring_name = kwargs.pop("scoring_name", None)
        cv = kwargs.pop("cv", None)
        super().__init__(*args, **kwargs)
        if task is not None:
            self.task = task
        if scoring_name is not None:
            self.scoring_name = scoring_name
        if cv is not None:
            self.cv = cv
