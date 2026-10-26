"""
Capa de integración con ViewX — serializadores de resultados de SynaptIX.

Convierte los resultados de synaptix (evaluación de modelos, validación
cruzada, comparación de modelos) al payload dict que entiende
``viewx.from_report_payload()``:

``{"title", "sections", "tables", "valueboxes", "figures", "metadata"}``

Este módulo NO requiere ViewX instalado; solo lo reexporta si está
disponible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..results import (
    CrossValidationResult,
    EvaluationResult,
    ModelComparisonResult,
    TrainTestResult,
)

# Las clases de ViewX son opcionales; solo se importan si están disponibles.
try:
    from viewx import HTML, Report, DataMatrix, Presentation, Slide

    _VIEWX_CLASSES = True
except ImportError:
    HTML = Report = DataMatrix = Presentation = Slide = None  # type: ignore[misc, assignment]
    _VIEWX_CLASSES = False

__all__ = [
    "HTML",
    "Report",
    "DataMatrix",
    "Presentation",
    "Slide",
    "to_report_data",
    "VIEWX_CLASSES_AVAILABLE",
]

VIEWX_CLASSES_AVAILABLE: bool = _VIEWX_CLASSES


def _scalar_metrics(metrics: dict) -> Dict[str, float]:
    """Filtra las métricas escalares (excluye anidados como ``por_clase``)."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def _evaluation_payload(result: EvaluationResult) -> Dict[str, Any]:
    """Payload para métricas de evaluación de un modelo."""
    scalars = _scalar_metrics(result)
    model_name = result.model_name or "Modelo"
    task = result.task or "evaluación"

    valueboxes = [
        {"label": key, "value": f"{value:.4f}"}
        for key, value in list(scalars.items())[:4]
    ]

    tables: List[Dict[str, Any]] = [{"name": "metricas", "data": [scalars]}]

    per_class = result.get("por_clase")
    if isinstance(per_class, dict):
        class_rows = []
        for class_name, class_metrics in per_class.items():
            if isinstance(class_metrics, dict):
                class_rows.append({"clase": class_name, **class_metrics})
        if class_rows:
            tables.append({"name": "por_clase", "data": class_rows})

    n_samples = len(result.y_true) if result.y_true is not None else None
    body = f"Evaluación de {model_name} (tarea: {task})."
    if n_samples:
        body += f" Muestras evaluadas: {n_samples}."

    return {
        "title": f"Evaluación — {model_name}",
        "sections": [{"heading": "Resumen", "body": body}],
        "tables": tables,
        "valueboxes": valueboxes,
        "figures": [],
        "metadata": {"result_type": "EvaluationResult", "task": task},
    }


def _cross_validation_payload(result: CrossValidationResult) -> Dict[str, Any]:
    """Payload para resultados de validación cruzada."""
    scores = list(result.get("scores", []))
    model_name = result.model_name or "Modelo"
    scoring = result.scoring or "score"

    score_rows = [
        {"fold": i + 1, scoring: float(score)} for i, score in enumerate(scores)
    ]

    return {
        "title": f"Validación cruzada — {model_name}",
        "sections": [
            {
                "heading": "Resumen",
                "body": (
                    f"Validación cruzada de {model_name} "
                    f"({result.cv or len(scores)} folds, métrica: {scoring})."
                ),
            }
        ],
        "tables": [{"name": "scores_por_fold", "data": score_rows}],
        "valueboxes": [
            {"label": "Media", "value": f"{result.get('media', 0):.4f}"},
            {"label": "Desviación", "value": f"{result.get('desviacion', 0):.4f}"},
        ],
        "figures": [],
        "metadata": {"result_type": "CrossValidationResult", "scoring": scoring},
    }


def _train_test_payload(result: TrainTestResult) -> Dict[str, Any]:
    """Payload para métricas de entrenamiento vs. prueba."""
    model_name = result.model_name or "Modelo"
    frame = result.to_frame().reset_index()
    body = f"Evaluación train/test de {model_name} (tarea: {result.task})."
    if result.test_size is not None:
        body += f" test_size={result.test_size:.0%} ({result.n_train} train / {result.n_test} test)."

    source = result["test"] if result["test"] is not None else result["train"]
    valueboxes = [
        {"label": key, "value": f"{value:.4f}"}
        for key, value in list(_scalar_metrics(source).items())[:4]
    ]
    return {
        "title": f"Train vs. test — {model_name}",
        "sections": [{"heading": "Resumen", "body": body}],
        "tables": [{"name": "train_test", "data": frame.to_dict(orient="records")}],
        "valueboxes": valueboxes,
        "figures": [],
        "metadata": {"result_type": "TrainTestResult", "task": result.task},
    }


def _comparison_payload(result: pd.DataFrame) -> Dict[str, Any]:
    """Payload para la tabla comparativa de modelos."""
    task = getattr(result, "task", None) or "modelos"
    scoring = getattr(result, "scoring_name", None)
    if scoring is None:
        numeric_cols = result.select_dtypes(include="number").columns.tolist()
        scoring = numeric_cols[0] if numeric_cols else None

    valueboxes: List[Dict[str, Any]] = []
    if scoring is not None and len(result) > 0 and "modelo" in result.columns:
        best = result.iloc[0]
        valueboxes = [
            {"label": "Mejor modelo", "value": str(best["modelo"])},
            {"label": f"Mejor {scoring}", "value": f"{best[scoring]:.4f}"},
        ]

    return {
        "title": f"Comparación de modelos ({task})",
        "sections": [
            {
                "heading": "Resumen",
                "body": f"Comparación de {len(result)} modelos.",
            }
        ],
        "tables": [
            {"name": "comparacion", "data": result.to_dict(orient="records")}
        ],
        "valueboxes": valueboxes,
        "figures": [],
        "metadata": {"result_type": "ModelComparisonResult", "scoring": scoring},
    }


def _attach_figures(
    payload: Dict[str, Any],
    result: Any,
    include_figures: bool,
    data: Optional[Any],
) -> Dict[str, Any]:
    """Adjunta figuras Plotly al payload cuando se solicitan."""
    if not include_figures:
        return payload

    from ..viz.figures import (
        confusion_matrix_figure,
        model_comparison_figure,
        predictions_figure,
        residuals_figure,
        roc_curve_figure,
        scores_figure,
    )

    figures: List[Dict[str, Any]] = list(payload.get("figures", []))

    if isinstance(result, EvaluationResult):
        if result.y_true is not None and result.y_pred is not None:
            if result.task == "classification":
                figures.append(
                    {
                        "title": "Matriz de confusión",
                        "kind": "confusion_matrix",
                        "fig": confusion_matrix_figure(result.y_true, result.y_pred),
                    }
                )
                if result.y_proba is not None and len(np.unique(result.y_true)) == 2:
                    figures.append(
                        {
                            "title": "Curva ROC",
                            "kind": "roc_curve",
                            "fig": roc_curve_figure(result.y_true, result.y_proba),
                        }
                    )
            elif result.task == "regression":
                figures.append(
                    {
                        "title": "Real vs. predicho",
                        "kind": "predictions",
                        "fig": predictions_figure(result.y_true, result.y_pred),
                    }
                )
                figures.append(
                    {
                        "title": "Residuos",
                        "kind": "residuals",
                        "fig": residuals_figure(result.y_true, result.y_pred),
                    }
                )
    elif isinstance(result, CrossValidationResult):
        scores = list(result.get("scores", []))
        if scores:
            figures.append(
                {
                    "title": "Scores por fold",
                    "kind": "cv_scores",
                    "fig": scores_figure(scores, scoring=result.scoring),
                }
            )
    elif isinstance(result, pd.DataFrame):
        scoring = payload.get("metadata", {}).get("scoring")
        if scoring is not None and "modelo" in result.columns:
            figures.append(
                {
                    "title": "Comparación de modelos",
                    "kind": "model_comparison",
                    "fig": model_comparison_figure(result, score_col=scoring),
                }
            )

    payload["figures"] = figures
    return payload


def to_report_data(
    result: Any,
    include_figures: bool = False,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Serializa un resultado de synaptix en un payload compatible con ViewX.

    Esta función NO requiere ViewX instalado. Con ``include_figures=True``
    necesita plotly (``pip install synaptix[viz]``).

    Parameters
    ----------
    result : EvaluationResult, CrossValidationResult, ModelComparisonResult o dict
        Resultado a serializar. Un ``dict`` plano se asume que ya es un
        payload y se devuelve tal cual.
    include_figures : bool
        Adjuntar figuras Plotly al payload.
    data : DataFrame, optional
        Datos fuente para figuras adicionales (reservado).

    Returns
    -------
    dict
        Payload con claves ``title``, ``sections``, ``tables``,
        ``valueboxes``, ``figures`` y ``metadata``.
    """
    if isinstance(result, EvaluationResult):
        payload = _evaluation_payload(result)
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, CrossValidationResult):
        payload = _cross_validation_payload(result)
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, TrainTestResult):
        payload = _train_test_payload(result)
        test = result["test"]
        if include_figures and isinstance(test, EvaluationResult):
            payload["figures"] = _attach_figures({}, test, True, data)["figures"]
        return payload

    if isinstance(result, (ModelComparisonResult, pd.DataFrame)):
        payload = _comparison_payload(result)
        return _attach_figures(payload, result, include_figures, data)

    if isinstance(result, dict):
        return result

    raise TypeError(
        f"Tipo de resultado no soportado: {type(result).__name__}. "
        "Se espera EvaluationResult, CrossValidationResult, "
        "ModelComparisonResult, DataFrame o dict."
    )
