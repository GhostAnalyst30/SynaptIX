"""
synaptix.model_selection - Validación y selección de modelos.

- ``cross_validate``  : validación cruzada de cualquier modelo SynaptIX.
- ``GridSearch``      : búsqueda de hiperparámetros en rejilla.
- ``compare_models``  : entrena varios modelos y devuelve tabla comparativa.

Ejemplo
-------
>>> from synaptix.model_selection import compare_models
>>> tabla = compare_models(X, y, task="classification")
>>> print(tabla)  # modelos ordenados por desempeño
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

from ..base import ArrayLike, SynaptixModel, to_matrix, to_vector

__all__ = ["cross_validate", "GridSearch", "compare_models"]


def cross_validate(
    model: SynaptixModel,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    scoring: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Validación cruzada k-fold de un modelo SynaptIX.

    Parameters
    ----------
    model : SynaptixModel
        Modelo (no necesita estar entrenado).
    X : array-like
        Features.
    y : array-like
        Variable objetivo.
    cv : int, default=5
        Número de folds.
    scoring : str, optional
        Métrica de sklearn ("accuracy", "f1_weighted", "r2",
        "neg_mean_absolute_error", ...). Por defecto: accuracy en
        clasificación y R² en regresión.

    Returns
    -------
    dict
        ``{"scores", "media", "desviacion"}``.
    """
    if scoring is None:
        scoring = "accuracy" if model.task == "classification" else "r2"

    scores = cross_val_score(model.model, to_matrix(X), to_vector(y), cv=cv, scoring=scoring)
    results = {
        "scores": scores.tolist(),
        "media": float(scores.mean()),
        "desviacion": float(scores.std()),
    }

    if verbose:
        print(f"\n=== Validación cruzada ({cv} folds, {scoring}) ===")
        print(f"  Modelo    : {model.name}")
        print(f"  Media     : {results['media']:.4f}")
        print(f"  Desviación: {results['desviacion']:.4f}")

    return results


class GridSearch:
    """Búsqueda de hiperparámetros en rejilla con validación cruzada.

    Parameters
    ----------
    model : SynaptixModel
        Modelo base.
    param_grid : dict
        Rejilla de parámetros, por ejemplo
        ``{"n_estimators": [50, 100], "max_depth": [3, 5, None]}``.
    cv : int, default=5
        Número de folds.
    scoring : str, optional
        Métrica de sklearn (por defecto según la tarea del modelo).

    Ejemplo
    -------
    >>> from synaptix.supervised import RandomForestClassifier
    >>> from synaptix.model_selection import GridSearch
    >>> search = GridSearch(
    ...     RandomForestClassifier(),
    ...     {"n_estimators": [50, 100], "max_depth": [3, None]},
    ... )
    >>> mejor_modelo = search.fit(X, y)
    >>> search.best_params_
    """

    def __init__(
        self,
        model: SynaptixModel,
        param_grid: dict,
        cv: int = 5,
        scoring: Optional[str] = None,
    ):
        self.base_model = model
        if scoring is None:
            scoring = "accuracy" if model.task == "classification" else "r2"
        self.search = GridSearchCV(
            model.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        self.best_params_: Optional[dict] = None
        self.best_score_: Optional[float] = None

    def fit(self, X: ArrayLike, y: ArrayLike, verbose: bool = True) -> SynaptixModel:
        """Ejecuta la búsqueda y devuelve el modelo con los mejores parámetros.

        Returns
        -------
        SynaptixModel
            El modelo base re-entrenado con la mejor combinación.
        """
        self.search.fit(to_matrix(X), to_vector(y))
        self.best_params_ = self.search.best_params_
        self.best_score_ = float(self.search.best_score_)

        if verbose:
            print(f"\n=== GridSearch: {self.base_model.name} ===")
            print(f"  Mejor score : {self.best_score_:.4f}")
            print(f"  Parámetros  : {self.best_params_}")

        self.base_model.model = self.search.best_estimator_
        self.base_model.fitted = True
        return self.base_model


def compare_models(
    X: ArrayLike,
    y: ArrayLike,
    task: Literal["classification", "regression"] = "classification",
    cv: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entrena y compara varios modelos con validación cruzada (AutoML-lite).

    Parameters
    ----------
    X : array-like
        Features.
    y : array-like
        Variable objetivo.
    task : {"classification", "regression"}, default="classification"
        Tipo de problema.
    cv : int, default=5
        Número de folds de validación cruzada.

    Returns
    -------
    DataFrame
        Tabla con score medio, desviación y tiempo por modelo,
        ordenada de mejor a peor.

    Ejemplo
    -------
    >>> from synaptix.model_selection import compare_models
    >>> tabla = compare_models(X, y, task="regression")
    """
    from .. import supervised

    if task == "classification":
        candidates = [
            supervised.LogisticRegression(),
            supervised.DecisionTreeClassifier(),
            supervised.RandomForestClassifier(),
            supervised.GradientBoostingClassifier(),
            supervised.KNNClassifier(),
            supervised.NaiveBayes(),
            supervised.SVMClassifier(),
        ]
        scoring = "accuracy"
    else:
        candidates = [
            supervised.LinearRegression(),
            supervised.RidgeRegression(),
            supervised.LassoRegression(),
            supervised.DecisionTreeRegressor(),
            supervised.RandomForestRegressor(),
            supervised.GradientBoostingRegressor(),
            supervised.KNNRegressor(),
        ]
        scoring = "r2"

    X_matrix, y_vector = to_matrix(X), to_vector(y)
    rows = []

    for model in candidates:
        start = time.perf_counter()
        try:
            scores = cross_val_score(model.model, X_matrix, y_vector, cv=cv, scoring=scoring)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "modelo": model.name,
                    scoring: float(scores.mean()),
                    "desviacion": float(scores.std()),
                    "tiempo_s": round(elapsed, 3),
                }
            )
        except Exception as error:  # un modelo no debe frenar la comparación
            rows.append(
                {
                    "modelo": model.name,
                    scoring: np.nan,
                    "desviacion": np.nan,
                    "tiempo_s": np.nan,
                    "error": str(error),
                }
            )

    table = pd.DataFrame(rows).sort_values(scoring, ascending=False).reset_index(drop=True)

    if verbose:
        print(f"\n=== Comparación de modelos ({task}, {cv} folds) ===")
        print(table.to_string(index=False))

    return table
