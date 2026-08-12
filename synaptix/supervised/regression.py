"""
Modelos de regresión supervisada.

Todos los modelos heredan de :class:`synaptix.base.SynaptixModel` y
comparten la API ``fit / predict / evaluate / summary / save / load``.

Ejemplo
-------
>>> from synaptix.supervised import RandomForestRegressor
>>> model = RandomForestRegressor(n_estimators=200)
>>> model.fit(X_train, y_train)
>>> model.evaluate(X_test, y_test, plot=True)
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingRegressor as _SkGBR,
    RandomForestRegressor as _SkRFR,
)
from sklearn.linear_model import (
    Lasso as _SkLasso,
    LinearRegression as _SkLinear,
    Ridge as _SkRidge,
)
from sklearn.neighbors import KNeighborsRegressor as _SkKNR
from sklearn.svm import SVR as _SkSVR
from sklearn.tree import DecisionTreeRegressor as _SkDTR

from ..base import SynaptixModel


class _RegressionModel(SynaptixModel):
    task = "regression"


class LinearRegression(_RegressionModel):
    """Regresión lineal por mínimos cuadrados.

    Parameters
    ----------
    **kwargs
        Parámetros de ``sklearn.linear_model.LinearRegression``
        (por ejemplo ``fit_intercept=False``).
    """

    def __init__(self, **kwargs):
        super().__init__(_SkLinear(**kwargs))


class RidgeRegression(_RegressionModel):
    """Regresión lineal con regularización L2.

    Parameters
    ----------
    alpha : float, default=1.0
        Fuerza de la regularización. Valores mayores = más regularización.
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(_SkRidge(alpha=alpha, **kwargs))


class LassoRegression(_RegressionModel):
    """Regresión lineal con regularización L1 (selección de features).

    Parameters
    ----------
    alpha : float, default=1.0
        Fuerza de la regularización. Coeficientes poco relevantes se
        vuelven exactamente cero.
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(_SkLasso(alpha=alpha, **kwargs))


class DecisionTreeRegressor(_RegressionModel):
    """Árbol de decisión para regresión.

    Parameters
    ----------
    max_depth : int, optional
        Profundidad máxima del árbol (None = sin límite).
    """

    def __init__(self, max_depth=None, random_state: int = 42, **kwargs):
        super().__init__(_SkDTR(max_depth=max_depth, random_state=random_state, **kwargs))


class RandomForestRegressor(_RegressionModel):
    """Bosque aleatorio para regresión (ensamble de árboles).

    Parameters
    ----------
    n_estimators : int, default=100
        Número de árboles.
    max_depth : int, optional
        Profundidad máxima de cada árbol.
    """

    def __init__(
        self, n_estimators: int = 100, max_depth=None, random_state: int = 42, **kwargs
    ):
        super().__init__(
            _SkRFR(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs,
            )
        )


class GradientBoostingRegressor(_RegressionModel):
    """Gradient boosting para regresión.

    Parameters
    ----------
    n_estimators : int, default=100
        Número de etapas de boosting.
    learning_rate : float, default=0.1
        Tasa de aprendizaje de cada etapa.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            _SkGBR(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                **kwargs,
            )
        )


class SVR(_RegressionModel):
    """Máquina de vectores de soporte para regresión.

    Parameters
    ----------
    kernel : str, default="rbf"
        Kernel a usar: "linear", "poly", "rbf" o "sigmoid".
    C : float, default=1.0
        Parámetro de regularización.
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0, **kwargs):
        super().__init__(_SkSVR(kernel=kernel, C=C, **kwargs))


class KNNRegressor(_RegressionModel):
    """Regresión por k vecinos más cercanos.

    Parameters
    ----------
    n_neighbors : int, default=5
        Número de vecinos a considerar.
    """

    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__(_SkKNR(n_neighbors=n_neighbors, **kwargs))
