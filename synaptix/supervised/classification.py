"""
Modelos de clasificación supervisada.

Todos los modelos heredan de :class:`synaptix.base.SynaptixModel` y
comparten la API ``fit / predict / predict_proba / evaluate / save / load``.

Ejemplo
-------
>>> from synaptix.supervised import RandomForestClassifier
>>> model = RandomForestClassifier(n_estimators=100)
>>> model.fit(X_train, y_train)
>>> model.evaluate(X_test, y_test, plot=True)  # incluye matriz de confusión
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier as _SkGBC,
    RandomForestClassifier as _SkRFC,
)
from sklearn.linear_model import LogisticRegression as _SkLogistic
from sklearn.naive_bayes import GaussianNB as _SkGNB
from sklearn.neighbors import KNeighborsClassifier as _SkKNC
from sklearn.svm import SVC as _SkSVC
from sklearn.tree import DecisionTreeClassifier as _SkDTC

from ..base import SynaptixModel


class _ClassificationModel(SynaptixModel):
    task = "classification"


class LogisticRegression(_ClassificationModel):
    """Regresión logística (clasificación binaria o multiclase).

    Parameters
    ----------
    C : float, default=1.0
        Inverso de la fuerza de regularización.
    max_iter : int, default=1000
        Máximo de iteraciones del optimizador.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, **kwargs):
        super().__init__(_SkLogistic(C=C, max_iter=max_iter, **kwargs))


class DecisionTreeClassifier(_ClassificationModel):
    """Árbol de decisión para clasificación.

    Parameters
    ----------
    max_depth : int, optional
        Profundidad máxima del árbol (None = sin límite).
    """

    def __init__(self, max_depth=None, random_state: int = 42, **kwargs):
        super().__init__(_SkDTC(max_depth=max_depth, random_state=random_state, **kwargs))


class RandomForestClassifier(_ClassificationModel):
    """Bosque aleatorio para clasificación.

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
            _SkRFC(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs,
            )
        )


class GradientBoostingClassifier(_ClassificationModel):
    """Gradient boosting para clasificación.

    Parameters
    ----------
    n_estimators : int, default=100
        Número de etapas de boosting.
    learning_rate : float, default=0.1
        Tasa de aprendizaje.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            _SkGBC(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                **kwargs,
            )
        )


class SVMClassifier(_ClassificationModel):
    """Máquina de vectores de soporte para clasificación.

    Parameters
    ----------
    kernel : str, default="rbf"
        Kernel a usar: "linear", "poly", "rbf" o "sigmoid".
    C : float, default=1.0
        Parámetro de regularización.
    probability : bool, default=True
        Habilita ``predict_proba`` (algo más lento de entrenar).
    """

    def __init__(
        self, kernel: str = "rbf", C: float = 1.0, probability: bool = True, **kwargs
    ):
        super().__init__(_SkSVC(kernel=kernel, C=C, probability=probability, **kwargs))


class KNNClassifier(_ClassificationModel):
    """Clasificación por k vecinos más cercanos.

    Parameters
    ----------
    n_neighbors : int, default=5
        Número de vecinos a considerar.
    """

    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__(_SkKNC(n_neighbors=n_neighbors, **kwargs))


class NaiveBayes(_ClassificationModel):
    """Clasificador Naive Bayes gaussiano.

    Adecuado para features continuas; asume independencia condicional
    entre features y distribución normal por clase.
    """

    def __init__(self, **kwargs):
        super().__init__(_SkGNB(**kwargs))
