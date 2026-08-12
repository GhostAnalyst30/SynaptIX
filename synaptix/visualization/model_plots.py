"""
Gráficas de diagnóstico generadas a partir de modelos entrenados.

Ejemplo
-------
>>> from synaptix.visualization import plot_residuals, plot_decision_boundary
>>> plot_residuals(model, X_test, y_test)
>>> plot_decision_boundary(model, X, y, features=(0, 1))
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..base import ArrayLike, to_matrix, to_vector

__all__ = ["plot_residuals", "plot_decision_boundary", "model_report"]


def plot_residuals(model, X: ArrayLike, y: ArrayLike) -> None:
    """Análisis de residuos de un modelo de regresión entrenado.

    Muestra dos paneles: residuos vs. valores predichos (para detectar
    heterocedasticidad o patrones no capturados) y el histograma de
    residuos (idealmente centrado en cero y simétrico).

    Parameters
    ----------
    model : SynaptixModel
        Modelo de regresión ya entrenado.
    X : array-like
        Features de prueba.
    y : array-like
        Valores reales.

    Ejemplo
    -------
    >>> plot_residuals(modelo, X_test, y_test)
    """
    y_true = to_vector(y).astype(float)
    y_pred = np.asarray(model.predict(X)).ravel().astype(float)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#4c9f70")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicho")
    axes[0].set_ylabel("Residuo (real - predicho)")
    axes[0].set_title("Residuos vs. predicciones")
    axes[0].grid(alpha=0.25)

    axes[1].hist(residuals, bins=30, color="#4c9f70", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(
        residuals.mean(), color="crimson", linestyle=":",
        linewidth=1.4, label=f"media = {residuals.mean():.3f}",
    )
    axes[1].set_xlabel("Residuo")
    axes[1].set_title("Distribución de residuos")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model,
    X: ArrayLike,
    y: ArrayLike,
    features: Tuple[int, int] = (0, 1),
    resolution: int = 200,
) -> None:
    """Frontera de decisión de un clasificador en 2D.

    Reentrena internamente una copia del estimador usando solo las dos
    features indicadas y pinta las regiones de decisión sobre una malla.

    Parameters
    ----------
    model : SynaptixModel
        Clasificador de SynaptIX (entrenado o no; se usa una copia).
    X : DataFrame o ndarray
        Features completas.
    y : array-like
        Etiquetas.
    features : tuple, default=(0, 1)
        Índices de las dos features a usar.
    resolution : int, default=200
        Densidad de la malla (mayor = frontera más suave).

    Ejemplo
    -------
    >>> plot_decision_boundary(SVMClassifier(), X, y, features=(0, 2))
    """
    if isinstance(X, pd.DataFrame):
        names = [X.columns[features[0]], X.columns[features[1]]]
    else:
        names = [f"Feature {features[0]}", f"Feature {features[1]}"]

    X_full = to_matrix(X)
    X_2d = X_full[:, [features[0], features[1]]].astype(float)
    y_vec = to_vector(y)

    classes, y_codes = np.unique(y_vec, return_inverse=True)

    estimator = clone(getattr(model, "model", model))
    estimator.fit(X_2d, y_codes)

    margin_x = 0.08 * (X_2d[:, 0].max() - X_2d[:, 0].min() or 1.0)
    margin_y = 0.08 * (X_2d[:, 1].max() - X_2d[:, 1].min() or 1.0)
    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min() - margin_x, X_2d[:, 0].max() + margin_x, resolution),
        np.linspace(X_2d[:, 1].min() - margin_y, X_2d[:, 1].max() + margin_y, resolution),
    )
    grid_pred = estimator.predict(np.column_stack([xx.ravel(), yy.ravel()]))
    grid_pred = grid_pred.reshape(xx.shape).astype(float)

    plt.figure(figsize=(7.5, 6))
    plt.contourf(xx, yy, grid_pred, alpha=0.25, cmap="viridis", levels=len(classes))
    palette = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    for index, label in enumerate(classes):
        points = X_2d[y_codes == index]
        plt.scatter(
            points[:, 0], points[:, 1],
            color=palette[index], edgecolors="white",
            linewidths=0.5, s=42, label=str(label),
        )
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    model_name = getattr(model, "name", type(model).__name__)
    plt.title(f"Frontera de decisión: {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def model_report(model, X: ArrayLike, y: ArrayLike) -> dict:
    """Reporte visual completo de un modelo entrenado, en una llamada.

    Clasificación: métricas + matriz de confusión (+ ROC si es binaria).
    Regresión: métricas + real vs. predicho + análisis de residuos.

    Parameters
    ----------
    model : SynaptixModel
        Modelo entrenado.
    X : array-like
        Features de prueba.
    y : array-like
        Valores reales.

    Returns
    -------
    dict
        Métricas calculadas.

    Ejemplo
    -------
    >>> from synaptix.visualization import model_report
    >>> metricas = model_report(modelo, X_test, y_test)
    """
    from . import plot_confusion_matrix, plot_predictions, plot_roc_curve

    results = model.evaluate(X, y, verbose=True)
    y_true = to_vector(y)
    y_pred = model.predict(X)

    if getattr(model, "task", "regression") == "classification":
        plot_confusion_matrix(y_true, y_pred)
        classes = np.unique(y_true)
        if len(classes) == 2 and hasattr(model, "predict_proba"):
            positive = (y_true == classes[1]).astype(int)
            plot_roc_curve(positive, model.predict_proba(X)[:, 1])
    else:
        plot_predictions(y_true, y_pred)
        plot_residuals(model, X, y)

    return results
