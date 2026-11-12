"""
Espacios de hiperparámetros por modelo.

Una sola definición por parámetro sirve para las tres familias de búsqueda:

- ``("float", low, high, log)`` / ``("int", low, high, log)``
- ``("cat", [opciones])``

``DEFAULT_GRIDS`` son rejillas pequeñas (compatibles con versiones
anteriores) para la búsqueda en rejilla.
"""

from __future__ import annotations

from typing import Any, Dict

from scipy import stats

_FOREST = {
    "n_estimators": ("int", 50, 400, False),
    "max_depth": ("cat", [None, 5, 10, 20]),
    "min_samples_leaf": ("int", 1, 10, False),
    "max_features": ("cat", ["sqrt", "log2", None]),
}

SPACES: Dict[str, Dict[str, tuple]] = {
    "logistic": {"C": ("float", 1e-3, 1e2, True)},
    "decision_tree": {
        "max_depth": ("int", 2, 30, False),
        "min_samples_leaf": ("int", 1, 20, False),
    },
    "random_forest": _FOREST,
    "extra_trees": _FOREST,
    "gradient_boosting": {
        "n_estimators": ("int", 50, 300, False),
        "learning_rate": ("float", 0.01, 0.3, True),
        "max_depth": ("int", 2, 6, False),
        "subsample": ("float", 0.6, 1.0, False),
    },
    "hist_gradient_boosting": {
        "learning_rate": ("float", 0.01, 0.3, True),
        "max_iter": ("int", 50, 500, False),
        "max_leaf_nodes": ("int", 15, 63, False),
        "l2_regularization": ("float", 1e-6, 10.0, True),
        "min_samples_leaf": ("int", 5, 50, False),
    },
    "svm": {"C": ("float", 1e-2, 1e2, True), "gamma": ("cat", ["scale", "auto"])},
    "knn": {"n_neighbors": ("int", 1, 50, False), "weights": ("cat", ["uniform", "distance"])},
    "naive_bayes": {"var_smoothing": ("float", 1e-12, 1e-6, True)},
    "ridge": {"alpha": ("float", 1e-3, 1e3, True)},
    "lasso": {"alpha": ("float", 1e-4, 1e1, True)},
    "elastic_net": {"alpha": ("float", 1e-4, 1e1, True), "l1_ratio": ("float", 0.05, 0.95, False)},
    "linear": {},
    "bayesian_ridge": {},
    "gaussian_process": {},
}

DEFAULT_GRIDS: Dict[str, Dict[str, list]] = {
    "logistic": {"C": [0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [3, 5, 10, None]},
    "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    "extra_trees": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    "hist_gradient_boosting": {"learning_rate": [0.05, 0.1], "max_iter": [100, 200]},
    "svm": {"C": [0.1, 1.0, 10.0]},
    "knn": {"n_neighbors": [3, 5, 7, 11]},
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "lasso": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "elastic_net": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "naive_bayes": {},
    "linear": {},
    "bayesian_ridge": {},
    "gaussian_process": {},
}


def to_distributions(space: Dict[str, tuple]) -> Dict[str, Any]:
    """Convierte un espacio a distribuciones de ``scipy`` para RandomizedSearch."""
    out: Dict[str, Any] = {}
    for param, spec in space.items():
        kind = spec[0]
        if kind == "cat":
            out[param] = list(spec[1])
        elif kind == "float":
            _, low, high, log = spec
            out[param] = stats.loguniform(low, high) if log else stats.uniform(low, high - low)
        elif kind == "int":
            _, low, high, _log = spec
            out[param] = stats.randint(low, high + 1)
        else:
            raise ValueError(f"Tipo de espacio desconocido: {kind!r}")
    return out


def suggest(trial: Any, space: Dict[str, tuple]) -> Dict[str, Any]:
    """Muestrea parámetros con un ``optuna.Trial``."""
    params: Dict[str, Any] = {}
    for param, spec in space.items():
        kind = spec[0]
        if kind == "cat":
            params[param] = trial.suggest_categorical(param, list(spec[1]))
        elif kind == "float":
            params[param] = trial.suggest_float(param, spec[1], spec[2], log=spec[3])
        else:
            params[param] = trial.suggest_int(param, spec[1], spec[2], log=spec[3])
    return params


def grid_size(grid: Dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= max(len(values), 1)
    return size
