"""
synaptix.visualization - Gráficos a partir de modelos y datos.

Evaluación de modelos:
- ``plot_confusion_matrix``  : matriz de confusión.
- ``plot_roc_curve``         : curva ROC con AUC (binaria).
- ``plot_learning_curve``    : curva de aprendizaje (train vs validación).
- ``plot_feature_importance``: importancia de features.
- ``plot_clusters``          : scatter 2D coloreado por cluster.
- ``plot_predictions``       : real vs predicho (regresión).
- ``plot_residuals``         : análisis de residuos (regresión).
- ``plot_decision_boundary`` : frontera de decisión 2D (clasificación).
- ``model_report``           : reporte visual completo en una llamada.

Análisis exploratorio (EDA):
- ``plot_distributions``     : histogramas de columnas numéricas.
- ``plot_correlation``       : heatmap de correlación.
- ``plot_boxplots``          : boxplots (opcionalmente segmentados).
- ``plot_scatter_matrix``    : matriz de dispersión.
- ``plot_missing``           : porcentaje de nulos por columna.

Ejemplo
-------
>>> from synaptix.visualization import plot_correlation, model_report
>>> plot_correlation(df)
>>> model_report(modelo, X_test, y_test)
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
from sklearn.model_selection import learning_curve as sk_learning_curve

from ..base import ArrayLike, to_matrix, to_vector
from .eda import (
    plot_boxplots,
    plot_correlation,
    plot_distributions,
    plot_missing,
    plot_scatter_matrix,
)
from .model_plots import model_report, plot_decision_boundary, plot_residuals

__all__ = [
    # Modelos
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_learning_curve",
    "plot_feature_importance",
    "plot_clusters",
    "plot_predictions",
    "plot_residuals",
    "plot_decision_boundary",
    "model_report",
    # EDA
    "plot_distributions",
    "plot_correlation",
    "plot_boxplots",
    "plot_scatter_matrix",
    "plot_missing",
]


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[list] = None,
    cmap: str = "Blues",
) -> None:
    """Grafica la matriz de confusión.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    labels : list, optional
        Nombres de las clases (por defecto, los valores únicos).
    """
    y_true, y_pred = to_vector(y_true), to_vector(y_pred)
    matrix = sk_metrics.confusion_matrix(y_true, y_pred)
    class_names = labels if labels is not None else np.unique(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap=cmap)
    fig.colorbar(image)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)

    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: ArrayLike, y_proba: ArrayLike) -> None:
    """Grafica la curva ROC de un clasificador binario.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales (binarias).
    y_proba : array-like
        Probabilidad de la clase positiva (por ejemplo,
        ``model.predict_proba(X)[:, 1]``).
    """
    y_true = to_vector(y_true)
    y_proba = to_vector(y_proba)

    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_proba)
    auc = sk_metrics.auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_learning_curve(
    model,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
    scoring: Optional[str] = None,
) -> None:
    """Curva de aprendizaje: desempeño vs tamaño del set de entrenamiento.

    Útil para diagnosticar overfitting (curvas separadas) o
    underfitting (ambas curvas bajas).

    Parameters
    ----------
    model : SynaptixModel o estimador sklearn
        Modelo a analizar.
    X, y : array-like
        Datos completos.
    cv : int, default=5
        Folds de validación cruzada.
    scoring : str, optional
        Métrica de sklearn.
    """
    estimator = getattr(model, "model", model)
    sizes, train_scores, val_scores = sk_learning_curve(
        estimator,
        to_matrix(X),
        to_vector(y),
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 8),
    )

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Entrenamiento")
    plt.plot(sizes, val_scores.mean(axis=1), "o-", label="Validación")
    plt.fill_between(
        sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15,
    )
    plt.xlabel("Muestras de entrenamiento")
    plt.ylabel("Score")
    plt.title("Curva de aprendizaje")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_feature_importance(model, top_n: int = 15) -> None:
    """Grafica la importancia de features de un modelo SynaptIX entrenado.

    Parameters
    ----------
    model : SynaptixModel
        Modelo entrenado con ``feature_importances_`` o ``coef_``.
    top_n : int, default=15
        Número de features a mostrar.
    """
    importances = model.feature_importances()
    if importances is None:
        raise ValueError(f"{model.name} no expone importancias de features.")

    top = importances.abs().sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(7, max(3, 0.4 * len(top))))
    plt.barh(top.index, top.values)
    plt.xlabel("Importancia")
    plt.title(f"Importancia de features: {model.name}")
    plt.tight_layout()
    plt.show()


def plot_clusters(
    X: ArrayLike,
    labels: ArrayLike,
    features: Tuple[int, int] = (0, 1),
    title: str = "Clusters",
) -> None:
    """Scatter 2D coloreado por etiqueta de cluster.

    Parameters
    ----------
    X : DataFrame o ndarray
        Datos originales.
    labels : array-like
        Etiqueta de cluster por muestra (-1 = ruido en DBSCAN).
    features : tuple, default=(0, 1)
        Índices de las dos columnas a graficar.
    """
    if isinstance(X, pd.DataFrame):
        names = [X.columns[features[0]], X.columns[features[1]]]
        X = X.values
    else:
        X = to_matrix(X)
        names = [f"Feature {features[0]}", f"Feature {features[1]}"]

    labels = to_vector(labels)

    plt.figure(figsize=(7, 6))
    for value in np.unique(labels):
        points = X[labels == value]
        label_name = "Ruido" if value == -1 else f"Cluster {value}"
        plt.scatter(points[:, features[0]], points[:, features[1]], label=label_name, alpha=0.7)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Gráfico real vs predicho para regresión.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    """
    y_true, y_pred = to_vector(y_true), to_vector(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], "--", color="black", label="Predicción perfecta")

    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title("Regresión: real vs predicho")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
