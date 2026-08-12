"""
synaptix.supervised - Aprendizaje supervisado.

Regresión:
    LinearRegression, RidgeRegression, LassoRegression,
    DecisionTreeRegressor, RandomForestRegressor,
    GradientBoostingRegressor, SVR, KNNRegressor

Clasificación:
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    GradientBoostingClassifier, SVMClassifier, KNNClassifier, NaiveBayes

Todos comparten la API: fit / predict / evaluate / summary / save / load.

Ejemplo
-------
>>> from synaptix.supervised import LogisticRegression
>>> from synaptix.preprocessing import train_test_split
>>> model = LogisticRegression()
>>> model.fit(X_train, y_train)
>>> resultados = model.evaluate(X_test, y_test, plot=True)
"""

from .classification import (
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    KNNClassifier,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    SVMClassifier,
)
from .regression import (
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    KNNRegressor,
    LassoRegression,
    LinearRegression,
    RandomForestRegressor,
    RidgeRegression,
    SVR,
)

__all__ = [
    # Regresión
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "SVR",
    "KNNRegressor",
    # Clasificación
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "SVMClassifier",
    "KNNClassifier",
    "NaiveBayes",
]
