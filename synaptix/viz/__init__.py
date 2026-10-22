"""
synaptix.viz - Figuras Plotly para payloads de exportación a ViewX.

Requiere plotly (``pip install synaptix[viz]``). Para la visualización
matplotlib del núcleo usa ``synaptix.visualization``.
"""

from .figures import (
    clusters_figure,
    confusion_matrix_figure,
    feature_importance_figure,
    model_comparison_figure,
    predictions_figure,
    residuals_figure,
    roc_curve_figure,
    scores_figure,
)

__all__ = [
    "confusion_matrix_figure",
    "roc_curve_figure",
    "feature_importance_figure",
    "predictions_figure",
    "residuals_figure",
    "clusters_figure",
    "scores_figure",
    "model_comparison_figure",
]
