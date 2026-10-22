"""
synaptix.metrics - Métricas de evaluación unificadas.

- ``regression_metrics``     : MAE, MSE, RMSE, R², MAPE.
- ``classification_metrics`` : accuracy, precision, recall, F1.
- ``clustering_metrics``     : silhouette, Calinski-Harabasz, Davies-Bouldin.
- ``classification_report``  : reporte completo + matriz de confusión visual.

Ejemplo
-------
>>> from synaptix.metrics import classification_report
>>> classification_report(y_test, y_pred, plot=True)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn import metrics as sk_metrics

from ..base import ArrayLike, to_matrix, to_vector
from ..results import EvaluationResult

__all__ = [
    "regression_metrics",
    "classification_metrics",
    "clustering_metrics",
    "classification_report",
]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """Calcula las métricas estándar de regresión.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.

    Returns
    -------
    EvaluationResult
        ``{"MAE", "MSE", "RMSE", "R2", "MAPE"}`` (MAPE en %, se omite si
        hay ceros en ``y_true``). Compatible con dict; con ViewX
        instalado permite exportar con ``.to_html()`` y afines.
    """
    y_true = to_vector(y_true).astype(float)
    y_pred = to_vector(y_pred).astype(float)

    mse = sk_metrics.mean_squared_error(y_true, y_pred)
    results = {
        "MAE": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(sk_metrics.r2_score(y_true, y_pred)),
    }
    if not np.any(y_true == 0):
        results["MAPE"] = float(
            100 * np.mean(np.abs((y_true - y_pred) / y_true))
        )
    return EvaluationResult(results, task="regression", y_true=y_true, y_pred=y_pred)


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "weighted"
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

    Returns
    -------
    EvaluationResult
        ``{"accuracy", "precision", "recall", "f1"}``. Compatible con
        dict; con ViewX instalado permite exportar con ``.to_html()``
        y afines.
    """
    y_true = to_vector(y_true)
    y_pred = to_vector(y_pred)

    results = {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            sk_metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            sk_metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(
            sk_metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
    }
    return EvaluationResult(results, task="classification", y_true=y_true, y_pred=y_pred)


def clustering_metrics(X: ArrayLike, labels: ArrayLike) -> dict:
    """Métricas internas de calidad de clustering.

    Parameters
    ----------
    X : array-like
        Datos usados en el clustering.
    labels : array-like
        Etiqueta de cluster de cada muestra.

    Returns
    -------
    EvaluationResult
        ``{"silhouette", "calinski_harabasz", "davies_bouldin"}``.
        Silhouette y Calinski: mayor es mejor. Davies-Bouldin: menor es mejor.
    """
    X = to_matrix(X)
    labels = to_vector(labels)

    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        raise ValueError("Se necesitan al menos 2 clusters para calcular métricas.")

    results = {
        "silhouette": float(sk_metrics.silhouette_score(X, labels)),
        "calinski_harabasz": float(sk_metrics.calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(sk_metrics.davies_bouldin_score(X, labels)),
    }
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
