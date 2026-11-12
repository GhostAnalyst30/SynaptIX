"""
Explainability Engine: importancia de variables y diagnóstico de errores.

Todas las importancias miden contribución *predictiva* al modelo entrenado,
no efectos causales: una variable puede ser importante por estar
correlacionada con la causa real, o por fuga de información.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.inspection import permutation_importance

from .exceptions import MissingDependencyError
from .features import output_feature_names

CAUSAL_NOTE = (
    "Las importancias miden cuánto usa el modelo cada variable para predecir, "
    "no su efecto causal sobre el target."
)


def permutation_importances(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scorer: Any,
    n_repeats: int = 5,
    random_state: Optional[int] = None,
    max_rows: int = 2000,
    n_jobs: Optional[int] = 1,
) -> pd.DataFrame:
    """Importancia por permutación sobre las columnas originales.

    Se calcula con datos que el modelo no vio al entrenar (el holdout), de
    modo que refleja capacidad de generalización y no memorización.
    """
    if len(X) > max_rows:
        idx = np.random.default_rng(random_state).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    result = permutation_importance(pipeline, X, y, scoring=scorer, n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=n_jobs)
    table = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False, kind="mergesort").reset_index(
        drop=True)


def model_importances(pipeline: Any) -> Optional[pd.DataFrame]:
    """Importancia nativa del modelo (árboles) o magnitud de coeficientes.

    Los coeficientes solo son comparables entre sí si las variables están
    en la misma escala (por eso el escalado por defecto).
    """
    model = pipeline.steps[-1][1]
    if hasattr(model, "feature_importances_"):
        values, kind = np.asarray(model.feature_importances_, dtype=float), "impurity"
    elif hasattr(model, "coef_"):
        coef = np.atleast_2d(np.asarray(model.coef_, dtype=float))
        values, kind = np.abs(coef).mean(axis=0), "abs_coef"
    else:
        return None
    try:
        names = output_feature_names(pipeline)
    except Exception:
        names = [f"x{i}" for i in range(len(values))]
    if len(names) != len(values):
        names = [f"x{i}" for i in range(len(values))]
    table = pd.DataFrame({"feature": names, "importance": values, "kind": kind})
    return table.sort_values("importance", ascending=False, kind="mergesort").reset_index(
        drop=True)


def coefficients(pipeline: Any, class_labels: Optional[Sequence] = None) -> Optional[pd.DataFrame]:
    """Coeficientes de modelos lineales (una columna por clase en multiclase)."""
    model = pipeline.steps[-1][1]
    if not hasattr(model, "coef_"):
        return None
    coef = np.atleast_2d(np.asarray(model.coef_, dtype=float))
    names = output_feature_names(pipeline)
    if coef.shape[0] == 1:
        columns = ["coef"]
    elif class_labels is not None and len(class_labels) == coef.shape[0]:
        columns = [str(c) for c in class_labels]
    else:
        columns = [f"clase_{i}" for i in range(coef.shape[0])]
    return pd.DataFrame(coef.T, index=names, columns=columns)


def diagnostics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence] = None,
    index: Optional[Sequence] = None,
    top: int = 10,
) -> Dict[str, Any]:
    """Diagnóstico de errores sobre el conjunto de prueba.

    Clasificación: matriz de confusión, métricas por clase y pares de
    clases más confundidos. Regresión: estadísticos de residuos, mayores
    errores y relación entre |residuo| y predicción (heterocedasticidad).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task == "classification":
        labels = list(np.unique(np.concatenate([y_true, y_pred]))) if labels is None else list(labels)
        matrix = skm.confusion_matrix(y_true, y_pred, labels=labels)
        names = [str(label) for label in labels]
        confusion = pd.DataFrame(matrix, index=names, columns=names)
        confusion.index.name, confusion.columns.name = "real", "predicho"
        report = skm.classification_report(y_true, y_pred, labels=labels, target_names=names,
                                           output_dict=True, zero_division=0)
        per_class = pd.DataFrame({name: report[name] for name in names}).T
        off = matrix.copy()
        np.fill_diagonal(off, 0)
        pairs = [(names[i], names[j], int(off[i, j]))
                 for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))
                 if off[i, j] > 0][:5]
        return {
            "confusion_matrix": confusion,
            "per_class": per_class,
            "most_confused": pd.DataFrame(pairs, columns=["real", "predicho", "n"]),
        }

    residuals = y_true.astype(float) - y_pred.astype(float)
    order = np.argsort(-np.abs(residuals))[:top]
    idx = np.asarray(index) if index is not None else np.arange(len(y_true))
    largest = pd.DataFrame({"index": idx[order], "real": y_true[order],
                            "predicho": y_pred[order], "residuo": residuals[order]})
    spread = pd.Series(np.abs(residuals)).corr(pd.Series(y_pred.astype(float)),
                                               method="spearman")
    return {
        "residuals": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "skew": float(pd.Series(residuals).skew()) if len(residuals) > 2 else 0.0,
            "max_abs": float(np.abs(residuals).max()),
            "spearman_abs_residual_vs_pred": float(spread) if pd.notna(spread) else float("nan"),
        },
        "largest_errors": largest,
    }


def shap_values(
    pipeline: Any,
    X: pd.DataFrame,
    max_samples: int = 200,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Valores SHAP del modelo sobre las features transformadas.

    Requiere ``shap`` (``pip install synaptix[explain]``). Usa
    ``shap.Explainer`` sobre la función de predicción del modelo, con una
    muestra de fondo de como máximo ``max_samples`` filas.
    """
    try:
        import shap
    except ImportError as exc:
        raise MissingDependencyError("shap", "explain", "explain_shap()") from exc

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)
    transform = pipeline[:-1]
    model = pipeline.steps[-1][1]
    data = transform.transform(X)
    if hasattr(data, "toarray"):
        data = data.toarray()
    names = output_feature_names(pipeline)
    predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    explainer = shap.Explainer(predict, data, feature_names=names)
    explanation = explainer(data)
    return {"values": explanation.values, "base_values": explanation.base_values,
            "data": data, "feature_names": names, "explanation": explanation}
