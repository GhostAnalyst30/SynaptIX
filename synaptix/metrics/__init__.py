"""
synaptix.metrics - Métricas de evaluación unificadas.

- ``regression_metrics``     : MAE, MSE, RMSE, R², MAPE (+ MedAE, explained_variance, max_error).
- ``classification_metrics`` : accuracy, precision, recall, F1 (+ balanced_accuracy,
  mcc, cohen_kappa, roc_auc, log_loss).
- ``clustering_metrics``     : silhouette, Calinski-Harabasz, Davies-Bouldin
  (+ ARI, NMI, homogeneity, completeness, v_measure si se pasa ``y_true``).
- ``classification_report``  : reporte completo + matriz de confusión visual.
- ``available_metrics``      : nombres de métricas disponibles por tarea.

Todas las funciones aceptan ``metrics=[...]`` para calcular solo las
métricas indicadas (sin distinguir mayúsculas).

Ejemplo
-------
>>> from synaptix.metrics import classification_report
>>> classification_report(y_test, y_pred, plot=True)
>>> regression_metrics(y_test, y_pred, metrics=["rmse", "r2", "medae"])
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn import metrics as sk_metrics

from ..base import ArrayLike, to_matrix, to_vector
from ..results import EvaluationResult

__all__ = [
    "regression_metrics",
    "classification_metrics",
    "clustering_metrics",
    "classification_report",
    "available_metrics",
]

_REGRESSION_DEFAULT = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
_REGRESSION_ALL = _REGRESSION_DEFAULT + ["MedAE", "explained_variance", "max_error"]

_CLASSIFICATION_DEFAULT = ["accuracy", "precision", "recall", "f1"]
_CLASSIFICATION_ALL = _CLASSIFICATION_DEFAULT + [
    "balanced_accuracy",
    "mcc",
    "cohen_kappa",
    "roc_auc",
    "log_loss",
]
_PROBA_METRICS = {"roc_auc", "log_loss"}

_CLUSTERING_INTERNAL = ["silhouette", "calinski_harabasz", "davies_bouldin"]
_CLUSTERING_EXTERNAL = ["ari", "nmi", "homogeneity", "completeness", "v_measure"]


def available_metrics(task: str) -> List[str]:
    """Nombres de las métricas disponibles para ``task``.

    Parameters
    ----------
    task : {"regression", "classification", "clustering"}

    Returns
    -------
    list de str
    """
    catalog = {
        "regression": _REGRESSION_ALL,
        "classification": _CLASSIFICATION_ALL,
        "clustering": _CLUSTERING_INTERNAL + _CLUSTERING_EXTERNAL,
    }
    if task not in catalog:
        raise ValueError(f"task debe ser uno de {list(catalog)} (recibido: {task!r}).")
    return list(catalog[task])


def _pick(requested: Optional[Sequence[str]], available: Sequence[str]) -> List[str]:
    """Resuelve nombres pedidos (sin distinguir mayúsculas) contra los disponibles."""
    if isinstance(requested, str):
        requested = [requested]
    lookup = {name.lower(): name for name in available}
    unknown = [name for name in requested if name.lower() not in lookup]
    if unknown:
        raise ValueError(f"Métricas desconocidas: {unknown}. Disponibles: {list(available)}")
    if not requested:
        raise ValueError("metrics no puede estar vacío.")
    return list(dict.fromkeys(lookup[name.lower()] for name in requested))


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metrics: Optional[Sequence[str]] = None,
) -> dict:
    """Calcula las métricas estándar de regresión.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    metrics : sequence of str, optional
        Métricas a calcular. Por defecto MAE, MSE, RMSE, R2 y MAPE.
        Disponibles además: MedAE, explained_variance, max_error.

    Returns
    -------
    EvaluationResult
        ``{"MAE", "MSE", "RMSE", "R2", "MAPE"}`` por defecto (MAPE en %;
        se omite si hay ceros en ``y_true`` y no se pidió
        explícitamente). Compatible con dict; con ViewX instalado
        permite exportar con ``.to_html()`` y afines.
    """
    y_true = to_vector(y_true).astype(float)
    y_pred = to_vector(y_pred).astype(float)
    has_zeros = bool(np.any(y_true == 0))

    if metrics is None:
        names = [m for m in _REGRESSION_DEFAULT if not (m == "MAPE" and has_zeros)]
    else:
        names = _pick(metrics, _REGRESSION_ALL)

    funcs: Dict[str, Callable[[], float]] = {
        "MAE": lambda: sk_metrics.mean_absolute_error(y_true, y_pred),
        "MSE": lambda: sk_metrics.mean_squared_error(y_true, y_pred),
        "RMSE": lambda: np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)),
        "R2": lambda: sk_metrics.r2_score(y_true, y_pred),
        "MAPE": lambda: (
            np.nan if has_zeros else 100 * np.mean(np.abs((y_true - y_pred) / y_true))
        ),
        "MedAE": lambda: sk_metrics.median_absolute_error(y_true, y_pred),
        "explained_variance": lambda: sk_metrics.explained_variance_score(y_true, y_pred),
        "max_error": lambda: sk_metrics.max_error(y_true, y_pred),
    }
    results = {name: float(funcs[name]()) for name in names}
    return EvaluationResult(results, task="regression", y_true=y_true, y_pred=y_pred)


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "weighted",
    metrics: Optional[Sequence[str]] = None,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
) -> dict:
    """Calcula las métricas estándar de clasificación.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    average : str, default="weighted"
        Promediado para problemas multiclase ("weighted", "macro", "micro").
    metrics : sequence of str, optional
        Métricas a calcular. Por defecto accuracy, precision, recall y f1.
        Disponibles además: balanced_accuracy, mcc, cohen_kappa, roc_auc,
        log_loss (estas dos requieren ``y_proba``).
    y_proba : ndarray, optional
        Probabilidades del modelo: ``(n,)`` con la clase positiva en
        binaria o ``(n, n_clases)``.
    labels : sequence, optional
        Clases en el orden de las columnas de ``y_proba``
        (``model.classes_``). Por defecto, las clases ordenadas de ``y_true``.

    Returns
    -------
    EvaluationResult
        ``{"accuracy", "precision", "recall", "f1"}`` por defecto.
        Compatible con dict; con ViewX instalado permite exportar con
        ``.to_html()`` y afines.
    """
    y_true = to_vector(y_true)
    y_pred = to_vector(y_pred)
    names = list(_CLASSIFICATION_DEFAULT) if metrics is None else _pick(
        metrics, _CLASSIFICATION_ALL
    )

    def _roc_auc() -> float:
        proba = np.asarray(y_proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba = proba[:, 1]
        if proba.ndim == 1:
            return sk_metrics.roc_auc_score(y_true, proba)
        return sk_metrics.roc_auc_score(
            y_true, proba, multi_class="ovr", average="weighted", labels=labels
        )

    def _log_loss() -> float:
        proba = np.asarray(y_proba)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return sk_metrics.log_loss(y_true, proba, labels=labels)

    funcs: Dict[str, Callable[[], float]] = {
        "accuracy": lambda: sk_metrics.accuracy_score(y_true, y_pred),
        "precision": lambda: sk_metrics.precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "recall": lambda: sk_metrics.recall_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "f1": lambda: sk_metrics.f1_score(y_true, y_pred, average=average, zero_division=0),
        "balanced_accuracy": lambda: sk_metrics.balanced_accuracy_score(y_true, y_pred),
        "mcc": lambda: sk_metrics.matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": lambda: sk_metrics.cohen_kappa_score(y_true, y_pred),
        "roc_auc": _roc_auc,
        "log_loss": _log_loss,
    }

    results = {}
    for name in names:
        if name in _PROBA_METRICS and y_proba is None:
            warnings.warn(
                f"'{name}' requiere probabilidades (y_proba); se devuelve NaN.",
                stacklevel=2,
            )
            results[name] = np.nan
            continue
        try:
            results[name] = float(funcs[name]())
        except ValueError as error:
            warnings.warn(f"No se pudo calcular '{name}': {error}", stacklevel=2)
            results[name] = np.nan
    return EvaluationResult(results, task="classification", y_true=y_true, y_pred=y_pred)


def clustering_metrics(
    X: ArrayLike,
    labels: ArrayLike,
    y_true: Optional[ArrayLike] = None,
    metrics: Optional[Sequence[str]] = None,
) -> dict:
    """Métricas de calidad de clustering.

    Parameters
    ----------
    X : array-like
        Datos usados en el clustering.
    labels : array-like
        Etiqueta de cluster de cada muestra.
    y_true : array-like, optional
        Etiquetas reales (si existen). Habilita las métricas externas
        ari, nmi, homogeneity, completeness y v_measure.
    metrics : sequence of str, optional
        Métricas a calcular. Por defecto las internas y, si se pasa
        ``y_true``, también las externas.

    Returns
    -------
    EvaluationResult
        Silhouette, Calinski y externas: mayor es mejor.
        Davies-Bouldin: menor es mejor.
    """
    X = to_matrix(X)
    labels = to_vector(labels)
    y_ref = None if y_true is None else to_vector(y_true)

    available = _CLUSTERING_INTERNAL + _CLUSTERING_EXTERNAL
    if metrics is None:
        names = _CLUSTERING_INTERNAL + (_CLUSTERING_EXTERNAL if y_ref is not None else [])
    else:
        names = _pick(metrics, available)
    if y_ref is None and any(name in _CLUSTERING_EXTERNAL for name in names):
        raise ValueError("Las métricas externas (ari, nmi, ...) requieren y_true.")

    unique = np.unique(labels[labels != -1])
    if any(name in _CLUSTERING_INTERNAL for name in names) and len(unique) < 2:
        raise ValueError("Se necesitan al menos 2 clusters para calcular métricas.")

    funcs: Dict[str, Callable[[], float]] = {
        "silhouette": lambda: sk_metrics.silhouette_score(X, labels),
        "calinski_harabasz": lambda: sk_metrics.calinski_harabasz_score(X, labels),
        "davies_bouldin": lambda: sk_metrics.davies_bouldin_score(X, labels),
        "ari": lambda: sk_metrics.adjusted_rand_score(y_ref, labels),
        "nmi": lambda: sk_metrics.normalized_mutual_info_score(y_ref, labels),
        "homogeneity": lambda: sk_metrics.homogeneity_score(y_ref, labels),
        "completeness": lambda: sk_metrics.completeness_score(y_ref, labels),
        "v_measure": lambda: sk_metrics.v_measure_score(y_ref, labels),
    }
    results = {name: float(funcs[name]()) for name in names}
    return EvaluationResult(results, task="clustering")


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    plot: bool = False,
    labels: Optional[list] = None,
) -> dict:
    """Reporte completo de clasificación con matriz de confusión.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    plot : bool, default=False
        Muestra la matriz de confusión con matplotlib.
    labels : list, optional
        Nombres de las clases para el gráfico.

    Returns
    -------
    EvaluationResult
        Métricas globales + reporte por clase de scikit-learn.
        Compatible con dict; exportable a ViewX.
    """
    y_true = to_vector(y_true)
    y_pred = to_vector(y_pred)

    results = classification_metrics(y_true, y_pred)

    print("\n=== Reporte de clasificación ===")
    for key, value in results.items():
        print(f"  {key:<10}: {value:.4f}")
    print()
    print(sk_metrics.classification_report(y_true, y_pred, zero_division=0))

    if plot:
        from ..visualization import plot_confusion_matrix

        plot_confusion_matrix(y_true, y_pred, labels=labels)

    results["por_clase"] = sk_metrics.classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    return results
