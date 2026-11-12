"""
synaptix.supervised - Aprendizaje supervisado.

Regresión:
    LinearRegression, RidgeRegression, LassoRegression, ElasticNetRegression,
    DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor, SVR, KNNRegressor

Clasificación:
    LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, SVMClassifier, KNNClassifier, NaiveBayes

Bayesianos (con incertidumbre en las predicciones):
    BayesianRidgeRegression, ARDRegression, GaussianProcessRegressor,
    GaussianProcessClassifier, MultinomialNB, BernoulliNB, ComplementNB
    PyMCLinearRegression, PyMCLogisticRegression (requieren synaptix[bayes])

Todos comparten la API: fit / predict / evaluate / summary / save / load.

Ejemplo
-------
>>> from synaptix.supervised import BayesianRidgeRegression
>>> model = BayesianRidgeRegression()
>>> model.fit(X_train, y_train)
>>> media, inferior, superior = model.predict_interval(X_test, std=2)
"""

from .bayesian import (
    ARDRegression,
    BayesianRidgeRegression,
    BernoulliNB,
    ComplementNB,
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    MultinomialNB,
    PyMCLinearRegression,
    PyMCLogisticRegression,
)
from .classification import (
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    KNNClassifier,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    SVMClassifier,
)
from .regression import (
    DecisionTreeRegressor,
    ElasticNetRegression,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
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
    "ElasticNetRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "SVR",
    "KNNRegressor",
    # Clasificación
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "SVMClassifier",
    "KNNClassifier",
    "NaiveBayes",
    # Bayesianos
    "BayesianRidgeRegression",
    "ARDRegression",
    "GaussianProcessRegressor",
    "GaussianProcessClassifier",
    "MultinomialNB",
    "BernoulliNB",
    "ComplementNB",
    "PyMCLinearRegression",
    "PyMCLogisticRegression",
]
