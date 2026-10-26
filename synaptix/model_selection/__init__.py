"""
synaptix.model_selection - Validación y selección de modelos.

- ``cross_validate``  : validación cruzada de cualquier modelo SynaptIX.
- ``GridSearch``      : búsqueda de hiperparámetros en rejilla.
- ``compare_models``  : entrena varios modelos y devuelve tabla comparativa.
- ``list_models``     : nombres de los modelos que acepta ``compare_models``.

``compare_models``:

- Detecta automáticamente la tarea (clasificación, regresión o, sin ``y``,
  clustering).
- Selecciona los modelos candidatos según el nº de predictores y la
  cardinalidad de ``y``, o solo los indicados en ``models``.
- Permite fijar la proporción de test con ``test_size`` (split train/test).
- Reporta métricas de entrenamiento y de prueba por modelo, y cuánto
  se sobreajusta cada uno.
- Permite ordenar por cualquier métrica y en cualquier sentido.
- Valida que X e y sean numéricos y consistentes antes de entrenar.
- Un modelo que falla no aborta la comparación.

Ejemplo
-------
>>> from synaptix.model_selection import compare_models
>>> tabla = compare_models(X, y)  # tarea inferida, 5 folds
>>> tabla = compare_models(
...     X, y, test_size=0.25,
...     models=["RandomForestClassifier", "LogisticRegression"],
...     metrics=["accuracy", "f1", "mcc"],
... )
>>> tabla[["modelo", "accuracy_train", "accuracy", "sobreajuste"]]
>>> compare_models(X, task="clustering", test_size=0.2, n_clusters=4)
"""

from __future__ import annotations

import copy
import time
import warnings
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    ShuffleSplit,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.model_selection import cross_validate as sk_cross_validate

from ..base import ArrayLike, SynaptixModel, check_test_size, to_matrix, to_vector
from ..results import CrossValidationResult, ModelComparisonResult

__all__ = ["cross_validate", "GridSearch", "compare_models", "list_models"]

Task = Literal["classification", "regression", "clustering"]

# ---------------------------------------------------------------------------
# Métricas por tarea (nombres de scorer de sklearn; las de clustering son
# las de ``synaptix.metrics.clustering_metrics``). Las "extra" solo se
# calculan si se piden en ``metrics``.
# ---------------------------------------------------------------------------
_CLASSIFICATION_METRICS = [
    "accuracy",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "roc_auc_ovr_weighted",  # requiere predict_proba
]
_CLASSIFICATION_EXTRA = [
    "balanced_accuracy",
    "matthews_corrcoef",
    "neg_log_loss",
    "f1_macro",
    "precision_macro",
    "recall_macro",
]

_REGRESSION_METRICS = [
    "r2",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_root_mean_squared_error",
]
_REGRESSION_EXTRA = [
    "neg_median_absolute_error",
    "neg_mean_absolute_percentage_error",
    "explained_variance",
]

_CLUSTERING_INTERNAL = ["silhouette", "calinski_harabasz", "davies_bouldin"]
_CLUSTERING_EXTERNAL = ["ari", "nmi", "homogeneity", "completeness", "v_measure"]

# Nombre para mostrar en la tabla (las métricas neg_ se muestran en positivo).
_METRIC_DISPLAY = {
    "neg_mean_absolute_error": "mae",
    "neg_mean_squared_error": "mse",
    "neg_root_mean_squared_error": "rmse",
    "neg_median_absolute_error": "medae",
    "neg_mean_absolute_percentage_error": "mape",
    "roc_auc_ovr_weighted": "roc_auc",
    "matthews_corrcoef": "mcc",
    "neg_log_loss": "log_loss",
}
_METRIC_ALIASES = {display: metric for metric, display in _METRIC_DISPLAY.items()}
_METRIC_ALIASES.update(
    {"f1": "f1_weighted", "precision": "precision_weighted", "recall": "recall_weighted"}
)
# MAPE en % para que coincida con ``synaptix.metrics.regression_metrics``.
_METRIC_SCALE = {"neg_mean_absolute_percentage_error": 100.0}
_ERROR_METRICS = {"mae", "mse", "rmse", "medae", "mape", "log_loss", "davies_bouldin"}

# Máximo de valores distintos de un y entero para considerarlo clasificación.
_MAX_CLASSES_HEURISTIC = 20


def _display(metric: str) -> str:
    return _METRIC_DISPLAY.get(metric, metric)


# ---------------------------------------------------------------------------
# Validación previa al entrenamiento
# ---------------------------------------------------------------------------
def _ensure_numeric(
    X: ArrayLike, y: Optional[ArrayLike], *, allow_categorical_y: bool = True
):
    """Convierte X e y a arrays numéricos validando el contenido.

    Parameters
    ----------
    X : array-like
        Features. Debe ser convertible a float.
    y : array-like o None
        Objetivo. En clasificación puede ser texto/categórico (se codifica
        a enteros); en regresión debe ser numérico. ``None`` en clustering
        sin etiquetas.
    allow_categorical_y : bool
        Si True, permite y categórico. Si False, exige y numérico
        continuo (regresión).

    Returns
    -------
    (X_num, y_num) : tuple[np.ndarray, np.ndarray o None]
    """
    X_arr = np.asarray(to_matrix(X))

    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError(f"X está vacío o tiene una forma inválida: {X_arr.shape}.")

    # ---- X ----
    if not np.issubdtype(X_arr.dtype, np.number):
        try:
            X_arr = X_arr.astype(float)
        except (ValueError, TypeError) as err:
            raise ValueError(
                "X contiene columnas no numéricas. Codifícalas antes "
                "(OneHotEncoder, OrdinalEncoder, pd.get_dummies, etc.)."
            ) from err
    if not np.isfinite(X_arr).all():
        raise ValueError("X contiene NaN o infinitos. Imputa/limpia antes de entrenar.")

    if y is None:
        return X_arr, None

    y_arr = np.asarray(to_vector(y)).ravel()
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X e y tienen distinto número de muestras: "
            f"{X_arr.shape[0]} vs {y_arr.shape[0]}."
        )

    # ---- y ----
    if allow_categorical_y:
        if np.issubdtype(y_arr.dtype, np.number):
            if not np.isfinite(y_arr).all():
                raise ValueError("y contiene NaN o infinitos. Imputa/limpia antes.")
        else:
            codes, _ = pd.factorize(y_arr)
            if (codes < 0).any():
                raise ValueError("y contiene valores faltantes. Imputa/limpia antes.")
            y_arr = codes.astype(int)
    else:
        if not np.issubdtype(y_arr.dtype, np.number):
            try:
                y_arr = y_arr.astype(float)
            except (ValueError, TypeError) as err:
                raise ValueError("Para regresión, y debe ser numérico continuo.") from err
        if not np.isfinite(y_arr).all():
            raise ValueError("y contiene NaN o infinitos. Imputa/limpia antes.")

    return X_arr, y_arr


def _check_cv(y: np.ndarray, cv: int, task: str) -> None:
    """Valida que ``cv`` sea compatible con los datos y la tarea."""
    if not isinstance(cv, (int, np.integer)) or isinstance(cv, bool) or cv < 2:
        raise ValueError(f"cv debe ser un entero >= 2 (recibido: {cv!r}).")
    if cv > len(y):
        raise ValueError(f"cv={cv} es mayor que el número de muestras ({len(y)}).")

    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            raise ValueError("Clasificación requiere al menos 2 clases distintas en y.")
        if counts.min() < cv:
            raise ValueError(
                f"La clase menos frecuente tiene {counts.min()} muestras, menos que "
                f"cv={cv}. Reduce cv o agrupa clases poco frecuentes."
            )
    elif np.unique(y).size < 2:
        raise ValueError("Regresión requiere que y no sea constante.")


def _check_holdout(
    n_samples: int,
    y: Optional[np.ndarray],
    n_splits: int,
    test_size: Union[float, int],
    task: str,
) -> None:
    """Valida ``n_splits`` y ``test_size`` para particiones train/test aleatorias."""
    if not isinstance(n_splits, (int, np.integer)) or isinstance(n_splits, bool) or n_splits < 1:
        raise ValueError(f"cv debe ser un entero >= 1 con test_size (recibido: {n_splits!r}).")
    n_test = check_test_size(test_size, n_samples)

    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            raise ValueError("Clasificación requiere al menos 2 clases distintas en y.")
        if counts.min() < 2:
            raise ValueError(
                "La clase menos frecuente tiene menos de 2 muestras; no se puede "
                "estratificar el split. Agrupa clases poco frecuentes."
            )
        if n_test < len(counts) or n_samples - n_test < len(counts):
            raise ValueError(
                f"test_size={test_size} es incompatible con {len(counts)} clases: "
                f"train y test deben tener al menos una muestra por clase."
            )
    elif task == "regression" and np.unique(y).size < 2:
        raise ValueError("Regresión requiere que y no sea constante.")


def _infer_task(y: np.ndarray) -> str:
    """Heurística: y no numérico/booleano, o entero con pocos valores → clasificación."""
    if y.dtype == bool or not np.issubdtype(y.dtype, np.number):
        return "classification"
    if np.issubdtype(y.dtype, np.integer):
        if len(np.unique(y)) <= _MAX_CLASSES_HEURISTIC:
            return "classification"
    return "regression"


def _resolve_metric(name: str) -> str:
    """Acepta el nombre sklearn o el alias corto (mae, rmse, f1, mcc...), sin mayúsculas."""
    key = name.lower()
    return _METRIC_ALIASES.get(key, key)


def _subset_metrics(
    default: Sequence[str], extra: Sequence[str], names: Optional[Sequence[str]]
) -> List[str]:
    """Devuelve las métricas solicitadas (o las por defecto) validando que existan."""
    if names is None:
        return list(default)
    if isinstance(names, str):
        names = [names]
    pool = list(default) + list(extra)
    resolved = [_resolve_metric(n) for n in names]
    missing = [n for n, r in zip(names, resolved) if r not in pool]
    if missing:
        available = [_display(m) for m in pool]
        raise ValueError(f"Métricas desconocidas: {missing}. Disponibles: {available}")
    if not resolved:
        raise ValueError("metrics no puede estar vacío.")
    return list(dict.fromkeys(resolved))


def _safe_cross_validate(
    model, X, y, cv, metric_names: Sequence[str], return_train: bool
) -> dict:
    """cross_validate tolerante a fallos por métrica (p. ej. roc_auc sin probas).

    ``cv`` puede ser un entero (k-fold) o un splitter de sklearn.
    Devuelve un dict ``{"test_<metrica>": array, "train_<metrica>": array}``.
    """
    scoring = {name: name for name in metric_names}
    n_splits = cv if isinstance(cv, (int, np.integer)) else cv.get_n_splits()
    try:
        return sk_cross_validate(
            model, X, y, cv=cv, scoring=scoring, error_score=np.nan,
            return_train_score=return_train,
        )
    except Exception:
        results = {}
        for name in metric_names:
            try:
                r = sk_cross_validate(
                    model, X, y, cv=cv, scoring=name, error_score=np.nan,
                    return_train_score=return_train,
                )
                results[f"test_{name}"] = r["test_score"]
                if return_train:
                    results[f"train_{name}"] = r["train_score"]
            except Exception:
                results[f"test_{name}"] = np.full(n_splits, np.nan)
                if return_train:
                    results[f"train_{name}"] = np.full(n_splits, np.nan)
        return results


def _cluster_cross_validate(
    model, X, y, splitter, metric_names: Sequence[str], return_train: bool
) -> dict:
    """Ajusta un modelo de clustering por split y devuelve el mismo formato que sklearn.

    Sin ``splitter`` se ajusta y evalúa sobre todos los datos (columna ``test_``).
    """
    results: Dict[str, list] = {f"test_{n}": [] for n in metric_names}
    if splitter is not None and return_train:
        results.update({f"train_{n}": [] for n in metric_names})

    if splitter is None:
        splits = [(np.arange(len(X)), None)]
    else:
        splits = splitter.split(X)

    for train_idx, test_idx in splits:
        fitted = copy.deepcopy(model).fit(X[train_idx])
        if test_idx is None:
            parts = [("test", X, y)]
        else:
            parts = [("test", X[test_idx], None if y is None else y[test_idx])]
            if return_train:
                parts.append(("train", X[train_idx], None if y is None else y[train_idx]))
        for prefix, X_part, y_part in parts:
            scores = fitted.evaluate(X_part, y_part, metrics=list(metric_names), verbose=False)
            for name in metric_names:
                results[f"{prefix}_{name}"].append(scores.get(name, np.nan))
    return {key: np.asarray(values, dtype=float) for key, values in results.items()}


def _summarize(raw, metric: str):
    """Media y desviación de una serie de scores, en la escala que se muestra."""
    raw = np.asarray(raw if raw is not None else [], dtype=float)
    if raw.size == 0 or np.isnan(raw).all():
        return np.nan, np.nan
    mean, std = float(np.nanmean(raw)), float(np.nanstd(raw))
    if metric.startswith("neg_"):
        mean = -mean
    scale = _METRIC_SCALE.get(metric, 1.0)
    return mean * scale, std * scale


# ---------------------------------------------------------------------------
# Catálogo de modelos por escenario
# ---------------------------------------------------------------------------
def _catalog(task: str, n_features: int = 2, n_clusters: int = 3):
    """Devuelve ``(por_defecto, todos)``: candidatos del escenario y catálogo completo."""
    from .. import supervised, unsupervised

    if task == "regression":
        base = [
            supervised.LinearRegression(),
            supervised.RidgeRegression(),
            supervised.LassoRegression(),
            supervised.DecisionTreeRegressor(),
            supervised.RandomForestRegressor(),
            supervised.GradientBoostingRegressor(),
            supervised.KNNRegressor(),
        ]
        extra = [
            supervised.SVR(),
            supervised.BayesianRidgeRegression(),
            supervised.ARDRegression(),
            supervised.GaussianProcessRegressor(),
        ]
        default = base if n_features >= 2 else base[:1]
    elif task == "classification":
        base = [
            supervised.LogisticRegression(),
            supervised.DecisionTreeClassifier(),
            supervised.RandomForestClassifier(),
            supervised.GradientBoostingClassifier(),
            supervised.KNNClassifier(),
            supervised.NaiveBayes(),
            supervised.SVMClassifier(),
        ]
        extra = [
            supervised.GaussianProcessClassifier(),
            supervised.MultinomialNB(),
            supervised.BernoulliNB(),
            supervised.ComplementNB(),
        ]
        default = base
    else:
        base = [
            unsupervised.KMeans(n_clusters=n_clusters),
            unsupervised.HierarchicalClustering(n_clusters=n_clusters),
            unsupervised.GaussianMixture(n_components=n_clusters),
            unsupervised.DBSCAN(),
        ]
        extra = []
        default = base
    return default, base + extra


def list_models(task: Optional[str] = None) -> Union[Dict[str, List[str]], List[str]]:
    """Nombres de modelos que acepta ``compare_models(models=[...])``.

    Parameters
    ----------
    task : {"classification", "regression", "clustering"}, optional
        Si se omite, devuelve un dict con todas las tareas.

    Returns
    -------
    list de str o dict
    """
    tasks = ("classification", "regression", "clustering")
    if task is not None and task not in tasks:
        raise ValueError(f"task debe ser uno de {list(tasks)} (recibido: {task!r}).")
    names = {t: [m.name for m in _catalog(t)[1]] for t in tasks}
    return names[task] if task is not None else names


def _select_models(default: list, catalog: list, models, task: str) -> list:
    """Filtra el catálogo por nombre y/o admite instancias de modelos propias."""
    if models is None:
        return default
    if isinstance(models, str) or not isinstance(models, (list, tuple, set)):
        models = [models]

    by_name = {m.name.lower(): m for m in catalog}
    selected, unknown = [], []
    for m in models:
        if isinstance(m, str):
            if m.lower() in by_name:
                selected.append(by_name[m.lower()])
            else:
                unknown.append(m)
        elif hasattr(m, "fit") and hasattr(m, "name"):
            if getattr(m, "task", task) != task:
                warnings.warn(
                    f"El modelo '{m.name}' es de tarea '{m.task}' pero la comparación "
                    f"es de '{task}'.",
                    stacklevel=3,
                )
            selected.append(m)
        else:
            unknown.append(m)

    if unknown:
        raise ValueError(
            f"Modelos desconocidos: {unknown}. Disponibles: {[m.name for m in catalog]}"
        )
    if not selected:
        raise ValueError("models no puede estar vacío.")
    return selected


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def cross_validate(
    model: SynaptixModel,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    scoring: Optional[str] = None,
    verbose: bool = True,
) -> CrossValidationResult:
    """Validación cruzada k-fold de un modelo SynaptIX.

    Parameters
    ----------
    model : SynaptixModel
        Modelo (no necesita estar entrenado).
    X : array-like
        Features. Deben ser numéricas y sin NaN/infinitos.
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
    CrossValidationResult
        ``{"scores", "media", "desviacion"}``. Compatible con dict;
        con ViewX instalado permite exportar con ``.to_html()`` y afines.
    """
    if scoring is None:
        scoring = "accuracy" if model.task == "classification" else "r2"

    X_num, y_num = _ensure_numeric(
        X, y, allow_categorical_y=(model.task == "classification")
    )
    _check_cv(y_num, cv, model.task)

    scores = cross_val_score(model.model, X_num, y_num, cv=cv, scoring=scoring)
    results = CrossValidationResult(
        {
            "scores": scores.tolist(),
            "media": float(scores.mean()),
            "desviacion": float(scores.std()),
        },
        model_name=model.name,
        scoring=scoring,
        cv=cv,
    )

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
        if not isinstance(param_grid, (dict, list)) or not param_grid:
            raise ValueError("param_grid debe ser un dict (o lista de dicts) no vacío.")
        self.base_model = model
        self.cv = cv
        if scoring is None:
            scoring = "accuracy" if model.task == "classification" else "r2"
        self.search = GridSearchCV(
            model.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        self.best_params_: Optional[dict] = None
        self.best_score_: Optional[float] = None

    def fit(self, X: ArrayLike, y: ArrayLike, verbose: bool = True) -> SynaptixModel:
        """Valida los datos, ejecuta la búsqueda y devuelve el mejor modelo.

        Returns
        -------
        SynaptixModel
            El modelo base re-entrenado con la mejor combinación.
        """
        is_classification = self.base_model.task == "classification"
        X_num, y_num = _ensure_numeric(X, y, allow_categorical_y=is_classification)
        _check_cv(y_num, self.cv, self.base_model.task)

        # En clasificación se entrena con las etiquetas originales para que
        # predict() siga devolviendo las mismas clases que el usuario pasó.
        y_fit = to_vector(y) if is_classification else y_num
        self.search.fit(X_num, y_fit)
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
    y: Optional[ArrayLike] = None,
    task: Optional[Task] = None,
    cv: Optional[int] = None,
    test_size: Optional[Union[float, int]] = None,
    models: Optional[Sequence[Union[str, object]]] = None,
    metrics: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = None,
    ascending: Optional[bool] = None,
    return_train_score: bool = True,
    n_clusters: int = 3,
    random_state: Optional[int] = 42,
    verbose: bool = True,
) -> ModelComparisonResult:
    """Entrena y compara modelos con métricas de entrenamiento y de prueba.

    Parameters
    ----------
    X : array-like
        Features. Se valida que sean numéricas y sin NaN.
    y : array-like, optional
        Objetivo. Puede ser categórico en clasificación (se codifica) pero
        debe ser numérico en regresión. Si se omite, la tarea es
        clustering; en clustering, si se pasa, habilita métricas externas
        (ari, nmi, ...).
    task : {"classification", "regression", "clustering"}, optional
        Si se omite, se infiere a partir de y (o clustering si no hay y).
    cv : int, optional
        Sin ``test_size``: folds de validación cruzada (por defecto 5).
        Con ``test_size``: número de splits train/test aleatorios
        (por defecto 1, un único holdout).
    test_size : float o int, optional
        Proporción (float en (0, 1)) o cantidad (int) de muestras de
        prueba. Todos los modelos usan exactamente los mismos splits
        (estratificados en clasificación). En clustering, sin
        ``test_size`` se ajusta y evalúa sobre todos los datos.
    models : sequence of str or model, optional
        Modelos a comparar. Acepta nombres del catálogo (ver
        :func:`list_models`, sin distinguir mayúsculas) y/o instancias
        SynaptIX ya configuradas. Por defecto, los candidatos del
        escenario detectado.
    metrics : sequence of str, optional
        Métricas a calcular (nombre sklearn o alias corto). Por defecto:

        - Clasificación: accuracy, f1_weighted, precision_weighted,
          recall_weighted, roc_auc. Extra: balanced_accuracy, mcc,
          log_loss, f1_macro, precision_macro, recall_macro.
          Alias: f1, precision, recall (weighted).
        - Regresión: r2, mae, mse, rmse. Extra: medae, mape (%),
          explained_variance.
        - Clustering: silhouette, calinski_harabasz, davies_bouldin
          (+ ari, nmi, homogeneity, completeness, v_measure si hay y).
    sort_by : str, optional
        Métrica (de prueba) por la que ordenar. Por defecto la primera
        de la tarea: accuracy, r2 o silhouette.
    ascending : bool, optional
        Sentido del orden. Por defecto se deduce de la métrica: las de
        error (mae, mse, rmse, medae, mape, log_loss, davies_bouldin)
        ascendente; el resto descendente.
    return_train_score : bool, default=True
        Calcula también las métricas sobre el conjunto de entrenamiento
        (columnas ``<métrica>_train``) y la columna ``sobreajuste``.
    n_clusters : int, default=3
        Clusters de los candidatos por defecto en clustering.
    random_state : int, optional, default=42
        Semilla de los splits aleatorios (solo aplica con ``test_size``).
    verbose : bool, default=True
        Imprime la tabla resultante.

    Returns
    -------
    ModelComparisonResult
        Una fila por modelo con columnas ``modelo`` y, por métrica,
        ``<métrica>_train`` (entrenamiento), ``<métrica>`` (prueba) y
        ``<métrica>_std``. Además ``sobreajuste`` (cuánto mejor es train
        que test en la métrica de orden; positivo = sobreajuste),
        ``desviacion`` (std de la métrica de orden), ``tiempo_s`` y, si
        hubo fallos, ``error``.

    Ejemplo
    -------
    >>> from synaptix.model_selection import compare_models, list_models
    >>> list_models("regression")
    >>> tabla = compare_models(X, y, task="regression", sort_by="rmse")
    >>> tabla = compare_models(
    ...     X, y, test_size=0.3,
    ...     models=["RandomForestClassifier", "LogisticRegression"],
    ... )
    """
    # 1) Validación de argumentos e inferencia de la tarea
    valid_tasks = ("classification", "regression", "clustering")
    if task is not None and task not in valid_tasks:
        raise ValueError(f"task debe ser uno de {list(valid_tasks)} (recibido: {task!r}).")

    if y is None:
        if task not in (None, "clustering"):
            raise ValueError(f"task='{task}' requiere y. Sin y solo se admite 'clustering'.")
        task = "clustering"
        y_raw = None
    else:
        y_raw = np.asarray(to_vector(y)).ravel()
        if task is None:
            task = _infer_task(y_raw)
        elif task != "clustering":
            inferred = _infer_task(y_raw)
            if task != inferred:
                warnings.warn(
                    f"Se pasó task='{task}' pero la heurística sugiere '{inferred}'. "
                    f"Se respeta el valor explícito.",
                    stacklevel=2,
                )

    # 2) Validación numérica y de splits antes de entrenar
    X_num, y_num = _ensure_numeric(X, y_raw, allow_categorical_y=(task != "regression"))
    n_samples, n_features = X_num.shape

    if test_size is None:
        if task == "clustering":
            n_splits, splitter = 1, None
            return_train_score = False
        else:
            n_splits = 5 if cv is None else cv
            _check_cv(y_num, n_splits, task)
            splitter = n_splits
    else:
        n_splits = 1 if cv is None else cv
        _check_holdout(n_samples, y_num, n_splits, test_size, task)
        splitter_cls = StratifiedShuffleSplit if task == "classification" else ShuffleSplit
        splitter = splitter_cls(n_splits=n_splits, test_size=test_size, random_state=random_state)

    # 3) Selección de métricas
    if task == "classification":
        chosen = _subset_metrics(_CLASSIFICATION_METRICS, _CLASSIFICATION_EXTRA, metrics)
    elif task == "regression":
        chosen = _subset_metrics(_REGRESSION_METRICS, _REGRESSION_EXTRA, metrics)
    elif metrics is None:
        chosen = _CLUSTERING_INTERNAL + (_CLUSTERING_EXTERNAL if y_num is not None else [])
    else:
        chosen = _subset_metrics(_CLUSTERING_INTERNAL, _CLUSTERING_EXTERNAL, metrics)
    if task == "clustering":
        if y_num is None and any(m in _CLUSTERING_EXTERNAL for m in chosen):
            raise ValueError("Las métricas externas (ari, nmi, ...) requieren pasar y.")

    sort_by = _resolve_metric(sort_by) if sort_by else chosen[0]
    if sort_by not in chosen:
        raise ValueError(
            f"sort_by='{sort_by}' no está entre las métricas calculadas: "
            f"{[_display(m) for m in chosen]}"
        )
    sort_col = _display(sort_by)
    if ascending is None:
        ascending = sort_col in _ERROR_METRICS

    # 4) Selección de modelos según escenario
    default_models, catalog = _catalog(task, n_features=n_features, n_clusters=n_clusters)
    candidates = _select_models(default_models, catalog, models, task)
    if task == "classification":
        n_classes = len(np.unique(y_num))
        scenario = (
            "clasificación binaria" if n_classes == 2
            else f"clasificación multiclase ({n_classes} clases)"
        )
    elif task == "regression":
        scenario = (
            "regresión lineal simple" if n_features == 1
            else f"regresión múltiple ({n_features} predictores)"
        )
    else:
        scenario = f"clustering ({n_samples} muestras, {n_features} features)"

    if verbose:
        print("\n=== Comparación de modelos ===")
        print(f"  Escenario detectado : {scenario}")
        if test_size is None and task == "clustering":
            print("  Evaluación          : todos los datos (sin test_size)")
        elif test_size is None:
            print(f"  Folds               : {n_splits}")
        else:
            n_test = check_test_size(test_size, n_samples)
            print(
                f"  Train/test          : {n_samples - n_test}/{n_test} muestras "
                f"(test_size={test_size}, {n_splits} split(s))"
            )
        print(f"  Modelos             : {[m.name for m in candidates]}")
        print(f"  Métricas            : {[_display(m) for m in chosen]}")
        print(f"  Ordenando por       : {sort_col} ({'asc' if ascending else 'desc'})")

    # 5) Entrenamiento + evaluación
    rows = []
    seen: Dict[str, int] = {}
    for model in candidates:
        seen[model.name] = seen.get(model.name, 0) + 1
        label = model.name if seen[model.name] == 1 else f"{model.name} #{seen[model.name]}"
        row: dict = {"modelo": label}
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if task == "clustering":
                    results = _cluster_cross_validate(
                        model, X_num, y_num, splitter, chosen, return_train_score
                    )
                else:
                    results = _safe_cross_validate(
                        model.model, X_num, y_num, splitter, chosen, return_train_score
                    )
            elapsed = time.perf_counter() - start

            failed = []
            for metric in chosen:
                display = _display(metric)
                test_mean, test_std = _summarize(results.get(f"test_{metric}"), metric)
                if return_train_score:
                    row[f"{display}_train"], _ = _summarize(
                        results.get(f"train_{metric}"), metric
                    )
                row[display] = test_mean
                row[f"{display}_std"] = test_std
                if np.isnan(test_mean):
                    failed.append(display)

            if return_train_score:
                gap = row[f"{sort_col}_train"] - row[sort_col]
                row["sobreajuste"] = -gap if sort_col in _ERROR_METRICS else gap
            row["desviacion"] = row[f"{sort_col}_std"]
            row["tiempo_s"] = round(elapsed, 3)
            if failed:
                detail = str(caught[0].message).strip().splitlines()[0] if caught else ""
                row["error"] = f"Métricas fallidas: {failed}" + (f" ({detail})" if detail else "")
        except Exception as error:  # un modelo no debe frenar la comparación
            for metric in chosen:
                display = _display(metric)
                if return_train_score:
                    row[f"{display}_train"] = np.nan
                row[display] = np.nan
                row[f"{display}_std"] = np.nan
            if return_train_score:
                row["sobreajuste"] = np.nan
            row["desviacion"] = np.nan
            row["tiempo_s"] = np.nan
            row["error"] = str(error)
        rows.append(row)

    # 6) Ordenar y empaquetar
    table = pd.DataFrame(rows)
    if "error" in table.columns:
        table["error"] = table["error"].fillna("")
    table = table.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(
        drop=True
    )

    result = ModelComparisonResult(
        table, task=task, scoring_name=sort_col, cv=n_splits, test_size=test_size
    )

    if verbose:
        print(result.to_string(index=False))

    return result
