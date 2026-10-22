"""
Builders de figuras Plotly usados por los payloads de exportación a ViewX.

Requieren plotly (``pip install synaptix[viz]``). El resto de synaptix
funciona sin plotly instalado (la visualización del núcleo usa matplotlib
en ``synaptix.visualization``).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - depende del entorno
    raise ImportError(
        "Plotly no está instalado. Instala con: pip install synaptix[viz]"
    ) from exc

from ..base import ArrayLike, to_vector

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


def confusion_matrix_figure(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[list] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Matriz de confusión como heatmap Plotly.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas.
    labels : list, optional
        Nombres de las clases (por defecto, los valores únicos).
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura lista para ViewX ``add_chart(fig=...)``.
    """
    from sklearn import metrics as sk_metrics

    y_true, y_pred = to_vector(y_true), to_vector(y_pred)
    matrix = sk_metrics.confusion_matrix(y_true, y_pred)
    class_names = [str(c) for c in (labels if labels is not None else np.unique(y_true))]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=title or "Matriz de confusión",
        xaxis_title="Predicho",
        yaxis_title="Real",
        yaxis_autorange="reversed",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def roc_curve_figure(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Curva ROC con AUC para clasificación binaria.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales (binarias).
    y_proba : array-like
        Probabilidad de la clase positiva.
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Curva ROC.
    """
    from sklearn import metrics as sk_metrics

    y_true = to_vector(y_true)
    y_proba = to_vector(y_proba)

    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_proba)
    auc = sk_metrics.auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash"))
    )
    fig.update_layout(
        title=title or "Curva ROC",
        xaxis_title="Tasa de falsos positivos",
        yaxis_title="Tasa de verdaderos positivos",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def feature_importance_figure(
    importances: pd.Series,
    top_n: int = 15,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Barras horizontales de importancia de features.

    Parameters
    ----------
    importances : Series
        Salida de ``SynaptixModel.feature_importances()``.
    top_n : int, default=15
        Número de features a mostrar.
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura de importancias.
    """
    top = importances.abs().sort_values(ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(x=top.values, y=[str(i) for i in top.index], orientation="h"))
    fig.update_layout(
        title=title or "Importancia de features",
        xaxis_title="Importancia",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def predictions_figure(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Dispersión real vs. predicho para regresión.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura real vs. predicho con la diagonal de predicción perfecta.
    """
    y_true, y_pred = to_vector(y_true).astype(float), to_vector(y_pred).astype(float)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predicciones"))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Predicción perfecta",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=title or "Real vs. predicho",
        xaxis_title="Real",
        yaxis_title="Predicho",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def residuals_figure(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Residuos vs. predicho más histograma de residuos.

    Parameters
    ----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura de diagnóstico de residuos (dos paneles).
    """
    from plotly.subplots import make_subplots

    y_true, y_pred = to_vector(y_true).astype(float), to_vector(y_pred).astype(float)
    residuals = y_true - y_pred

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuos", "Histograma"))
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode="markers", name="Residuos"),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, name="Residuos"), row=1, col=2)
    fig.update_layout(
        title=title or "Diagnóstico de residuos",
        showlegend=False,
        **_LAYOUT_DEFAULTS,
    )
    return fig


def clusters_figure(
    X: ArrayLike,
    labels: ArrayLike,
    features: tuple = (0, 1),
    title: Optional[str] = None,
) -> go.Figure:
    """
    Dispersión 2D coloreada por etiqueta de cluster.

    Parameters
    ----------
    X : DataFrame o ndarray
        Datos originales.
    labels : array-like
        Etiqueta de cluster por muestra (-1 = ruido en DBSCAN).
    features : tuple, default=(0, 1)
        Índices de las dos columnas a graficar.
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura de clusters.
    """
    from ..base import to_matrix

    if isinstance(X, pd.DataFrame):
        names = [str(X.columns[features[0]]), str(X.columns[features[1]])]
        values = X.values
    else:
        values = to_matrix(X)
        names = [f"Feature {features[0]}", f"Feature {features[1]}"]

    labels = to_vector(labels)

    fig = go.Figure()
    for value in np.unique(labels):
        points = values[labels == value]
        label_name = "Ruido" if value == -1 else f"Cluster {value}"
        fig.add_trace(
            go.Scatter(
                x=points[:, features[0]],
                y=points[:, features[1]],
                mode="markers",
                name=label_name,
            )
        )
    fig.update_layout(
        title=title or "Clusters",
        xaxis_title=names[0],
        yaxis_title=names[1],
        **_LAYOUT_DEFAULTS,
    )
    return fig


def scores_figure(
    scores: Sequence[float],
    scoring: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Barras de scores por fold (validación cruzada) con línea de media.

    Parameters
    ----------
    scores : secuencia de float
        Scores por fold.
    scoring : str, optional
        Nombre de la métrica (para el eje Y).
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura de scores.
    """
    scores = [float(s) for s in scores]
    folds = [f"Fold {i + 1}" for i in range(len(scores))]

    fig = go.Figure(go.Bar(x=folds, y=scores, name=scoring or "score"))
    fig.add_hline(
        y=float(np.mean(scores)),
        line_dash="dash",
        line_color="red",
        annotation_text="media",
    )
    fig.update_layout(
        title=title or "Validación cruzada",
        yaxis_title=scoring or "score",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def model_comparison_figure(
    table: pd.DataFrame,
    score_col: str,
    model_col: str = "modelo",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Barras horizontales comparando modelos por score.

    Parameters
    ----------
    table : DataFrame
        Tabla de ``compare_models`` (una fila por modelo).
    score_col : str
        Columna de score a graficar (por ejemplo ``"accuracy"`` o ``"r2"``).
    model_col : str, default="modelo"
        Columna con el nombre del modelo.
    title : str, optional
        Título de la figura.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura comparativa.
    """
    data = table.dropna(subset=[score_col]).sort_values(score_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data[score_col].astype(float),
            y=data[model_col].astype(str),
            orientation="h",
        )
    )
    fig.update_layout(
        title=title or f"Comparación de modelos ({score_col})",
        xaxis_title=score_col,
        **_LAYOUT_DEFAULTS,
    )
    return fig
