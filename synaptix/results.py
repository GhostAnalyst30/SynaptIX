"""
Objetos de resultado ligeros de SynaptIX.

Envuelven los resultados existentes (dicts de métricas, tablas de
comparación) sin romper el API actual:

- ``EvaluationResult``      : subclase de ``dict`` — se comporta igual que
  el dict de métricas de siempre, pero con métodos de exportación a ViewX.
- ``CrossValidationResult`` : subclase de ``dict`` para validación cruzada.
- ``TrainTestResult``       : subclase de ``dict`` con las métricas de
  entrenamiento y prueba (``{"train": ..., "test": ...}``) de ``fit_evaluate``.
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
    "TrainTestResult",
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


class TrainTestResult(dict, ViewXExportMixin):
    """
    Métricas de entrenamiento y prueba de un modelo (salida de ``fit_evaluate``).

    Subclase de ``dict`` con las claves ``"train"`` y ``"test"``, cada una
    con un dict de métricas (``EvaluationResult`` en modelos supervisados
    y de clustering). ``"test"`` es ``None`` si no hubo conjunto de prueba.

    Attributes
    ----------
    model_name : str o None
        Nombre del modelo.
    task : str o None
        ``"classification"``, ``"regression"``, ``"clustering"``,
        ``"decomposition"`` o ``"reinforcement"``.
    test_size : float o None
        Proporción usada como prueba.
    n_train, n_test : int o None
        Muestras (o episodios en refuerzo) de cada conjunto.
    fit_time : float o None
        Segundos de entrenamiento.
    info : dict
        Información adicional del modelo (parámetros, nº de clusters, ...).
    """

    def __init__(
        self,
        train: dict,
        test: Optional[dict] = None,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        test_size: Optional[float] = None,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        fit_time: Optional[float] = None,
        info: Optional[dict] = None,
    ):
        super().__init__({"train": train, "test": test})
        self.model_name = model_name
        self.task = task
        self.test_size = test_size
        self.n_train = n_train
        self.n_test = n_test
        self.fit_time = fit_time
        self.info = info or {}

    @property
    def train(self) -> dict:
        return self["train"]

    @property
    def test(self) -> Optional[dict]:
        return self["test"]

    def to_frame(self) -> pd.DataFrame:
        """Tabla métrica × {train, test, diferencia} (diferencia = test - train)."""
        train = {k: v for k, v in self["train"].items() if np.isscalar(v)}
        frame = pd.DataFrame({"train": pd.Series(train, dtype=float)})
        if self["test"] is not None:
            test = {k: v for k, v in self["test"].items() if np.isscalar(v)}
            order = list(dict.fromkeys([*train, *test]))
            frame = frame.reindex(order)
            frame["test"] = pd.Series(test, dtype=float)
            frame["diferencia"] = frame["test"] - frame["train"]
        frame.index.name = "metrica"
        return frame

    def to_dict(self) -> dict:
        """Devuelve ``{"train": {...}, "test": {...}}`` con dicts planos."""
        return {
            key: (dict(value) if value is not None else None) for key, value in self.items()
        }

    def __repr__(self) -> str:
        header = f"TrainTestResult({self.model_name or 'modelo'}, task={self.task})"
        return f"{header}\n{self.to_frame().to_string()}"


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
        ``"classification"``, ``"regression"`` o ``"clustering"``.
    scoring_name : str o None
        Columna de la métrica de prueba usada para ordenar
        (por ejemplo ``"accuracy"``, ``"r2"`` o ``"silhouette"``).
    cv : int o None
        Número de folds (o de splits train/test si se usó ``test_size``).
    test_size : float o None
        Proporción de test usada, o ``None`` si se usó k-fold.
    """

    _metadata = ["task", "scoring_name", "cv", "test_size"]

    @property
    def _constructor(self):
        return ModelComparisonResult

    def __init__(self, *args: Any, **kwargs: Any):
        task = kwargs.pop("task", None)
        scoring_name = kwargs.pop("scoring_name", None)
        cv = kwargs.pop("cv", None)
        test_size = kwargs.pop("test_size", None)
        super().__init__(*args, **kwargs)
        self.test_size = test_size
        if task is not None:
            self.task = task
        if scoring_name is not None:
            self.scoring_name = scoring_name
        if cv is not None:
            self.cv = cv
