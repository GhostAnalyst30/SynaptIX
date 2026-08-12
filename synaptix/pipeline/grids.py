"""
Registro de modelos y rejillas de hiperparámetros por defecto.

Usados por AutoPipeline y Pipeline para que el usuario no tenga que
elegir modelos ni parámetros; todo es sobreescribible.
"""

from __future__ import annotations

from .. import supervised

# Nombres amigables -> clase de modelo, por tarea
MODEL_REGISTRY = {
    "classification": {
        "logistic": supervised.LogisticRegression,
        "decision_tree": supervised.DecisionTreeClassifier,
        "random_forest": supervised.RandomForestClassifier,
        "gradient_boosting": supervised.GradientBoostingClassifier,
        "svm": supervised.SVMClassifier,
        "knn": supervised.KNNClassifier,
        "naive_bayes": supervised.NaiveBayes,
        "gaussian_process": supervised.GaussianProcessClassifier,
    },
    "regression": {
        "linear": supervised.LinearRegression,
        "ridge": supervised.RidgeRegression,
        "lasso": supervised.LassoRegression,
        "bayesian_ridge": supervised.BayesianRidgeRegression,
        "decision_tree": supervised.DecisionTreeRegressor,
        "random_forest": supervised.RandomForestRegressor,
        "gradient_boosting": supervised.GradientBoostingRegressor,
        "svm": supervised.SVR,
        "knn": supervised.KNNRegressor,
        "gaussian_process": supervised.GaussianProcessRegressor,
    },
}

# Modelos que se comparan por defecto (rápidos y robustos).
# gaussian_process se excluye por su costo O(n^3); se puede pedir explícito.
DEFAULT_CANDIDATES = {
    "classification": [
        "logistic",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svm",
        "knn",
        "naive_bayes",
    ],
    "regression": [
        "linear",
        "ridge",
        "lasso",
        "bayesian_ridge",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "knn",
    ],
}

# Rejillas de hiperparámetros por nombre amigable.
# Rejilla vacía = el modelo no se ajusta (sus defaults ya son razonables).
DEFAULT_GRIDS = {
    "logistic": {"C": [0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [3, 5, 10, None]},
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
    },
    "svm": {"C": [0.1, 1.0, 10.0]},
    "knn": {"n_neighbors": [3, 5, 7, 11]},
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "lasso": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "naive_bayes": {},
    "linear": {},
    "bayesian_ridge": {},
    "gaussian_process": {},
}


def resolve_model(task: str, name: str):
    """Devuelve la clase de modelo para un nombre amigable.

    Parameters
    ----------
    task : str
        "classification" o "regression".
    name : str
        Nombre amigable, por ejemplo "random_forest".
    """
    registry = MODEL_REGISTRY.get(task, {})
    if name not in registry:
        raise ValueError(
            f"Modelo '{name}' no disponible para {task}. "
            f"Opciones: {sorted(registry)}"
        )
    return registry[name]
