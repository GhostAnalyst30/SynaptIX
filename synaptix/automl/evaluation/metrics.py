"""
Evaluation Engine: métricas principales y secundarias por tarea.

Los nombres aceptan los de scikit-learn y alias cortos (``f1``, ``rmse``,
``mcc``, ``pr_auc``...). Las métricas de error se muestran en positivo
aunque scikit-learn las maximice con signo negativo.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn import metrics as skm
from sklearn.metrics import get_scorer, make_scorer
from sklearn.preprocessing import label_binarize

from ..exceptions import AutoMLError

# Proporción máxima de valores |y| ~ 0 que MAPE puede excluir antes de
# declararse no fiable.
MAPE_MAX_EXCLUDED = 0.10
MAPE_NEAR_ZERO_REL = 1e-6


def safe_mape(y_true, y_pred) -> float:
    """MAPE en % que excluye objetivos ~0 y avisa si son demasiados.

    Un objetivo es "cercano a cero" si ``|y| <= 1e-6 * max|y|``. Si más
    del 10 % lo es, devuelve NaN: el MAPE no es interpretable y conviene
    usar MAE o RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scale = np.max(np.abs(y_true)) if y_true.size else 0.0
    near_zero = np.abs(y_true) <= max(MAPE_NEAR_ZERO_REL * scale, 1e-12)
    excluded = float(near_zero.mean()) if y_true.size else 1.0
    if excluded > MAPE_MAX_EXCLUDED or near_zero.all():
        warnings.warn(
            f"MAPE no fiable: {excluded:.0%} de los valores reales son ~0. "
            "Usa MAE o RMSE.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    keep = ~near_zero
    return float(100 * np.mean(np.abs((y_true[keep] - y_pred[keep]) / y_true[keep])))


def pr_auc_score(y_true, y_proba, labels: Optional[Sequence] = None) -> float:
    """Área bajo la curva precisión-recall (average precision).

    Binaria: sobre la clase positiva. Multiclase: one-vs-rest ponderado
    por soporte.
    """
    proba = np.asarray(y_proba)
    if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 2):
        positive = proba if proba.ndim == 1 else proba[:, 1]
        classes = np.unique(y_true) if labels is None else np.asarray(labels)
        return float(skm.average_precision_score(y_true == classes[-1], positive))
    classes = np.arange(proba.shape[1]) if labels is None else np.asarray(labels)
    y_bin = label_binarize(y_true, classes=classes)
    return float(skm.average_precision_score(y_bin, proba, average="weighted"))


@dataclass
class MetricSpec:
    """Métrica resuelta.

    Attributes
    ----------
    key : str
        Nombre canónico (clave en los resultados de validación cruzada).
    display : str
        Nombre que se muestra (positivo para las de error).
    scorer : callable
        Scorer de scikit-learn ``scorer(estimator, X, y)``.
    greater_is_better : bool
        Sentido de optimización de la métrica mostrada.
    """

    key: str
    display: str
    scorer: Any
    greater_is_better: bool

    def to_display(self, raw: Union[float, np.ndarray]):
        """Convierte el valor de scikit-learn a la escala mostrada."""
        return raw if self.greater_is_better else -np.asarray(raw)


_CLASSIFICATION_ALIASES = {
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "mcc": "matthews_corrcoef",
    "log_loss": "neg_log_loss",
    "average_precision": "pr_auc",
}
_REGRESSION_ALIASES = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "medae": "neg_median_absolute_error",
    "mape": "mape",
    "neg_mean_absolute_percentage_error": "mape",
}
_DISPLAY = {
    "neg_mean_absolute_error": "mae",
    "neg_mean_squared_error": "mse",
    "neg_root_mean_squared_error": "rmse",
    "neg_median_absolute_error": "medae",
    "neg_log_loss": "log_loss",
    "matthews_corrcoef": "mcc",
    "roc_auc_ovr_weighted": "roc_auc",
}

DEFAULT_SCORING = {"classification": "accuracy", "regression": "r2"}
DEFAULT_SECONDARY = {
    "classification": ["balanced_accuracy", "f1_weighted", "mcc", "roc_auc", "pr_auc", "log_loss"],
    "regression": ["rmse", "mae", "medae"],
}


def resolve_metric(
    name: Union[str, Callable],
    task: str,
    labels: Optional[Sequence] = None,
) -> MetricSpec:
    """Resuelve un nombre de métrica (o un scorer propio) para la tarea.

    ``labels`` son las clases ordenadas (el orden de ``predict_proba``).
    """
    if callable(name) and not isinstance(name, str):
        label = getattr(name, "__name__", "custom")
        return MetricSpec(label, label, name, True)
    key = str(name).lower()
    aliases = _CLASSIFICATION_ALIASES if task == "classification" else _REGRESSION_ALIASES
    key = aliases.get(key, key)

    if task == "classification":
        n_classes = 2 if labels is None else len(labels)
        if key == "roc_auc":
            key = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
        if key == "pr_auc":
            scorer = make_scorer(
                pr_auc_score, response_method="predict_proba",
                labels=None if labels is None else list(labels),
            )
            return MetricSpec("pr_auc", "pr_auc", scorer, True)
    elif key == "mape":
        scorer = make_scorer(safe_mape, greater_is_better=False)
        return MetricSpec("mape", "mape", scorer, False)

    try:
        scorer = get_scorer(key)
    except (ValueError, KeyError) as exc:
        raise AutoMLError(f"Métrica desconocida para {task}: {name!r}") from exc
    greater = not key.startswith("neg_")
    display = _DISPLAY.get(key, key[4:] if key.startswith("neg_") else key)
    return MetricSpec(key, display, scorer, greater)


def resolve_metrics(
    scoring: Optional[Union[str, Callable]],
    metrics: Optional[Sequence[str]],
    task: str,
    labels: Optional[Sequence] = None,
) -> List[MetricSpec]:
    """Lista de métricas: la principal primero y luego las secundarias."""
    primary = resolve_metric(scoring or DEFAULT_SCORING[task], task, labels)
    names = DEFAULT_SECONDARY[task] if metrics is None else list(metrics)
    specs = [primary]
    for name in names:
        spec = resolve_metric(name, task, labels)
        if spec.key not in {s.key for s in specs}:
            specs.append(spec)
    return specs


def needs_proba(spec: MetricSpec) -> bool:
    """True si la métrica necesita ``predict_proba`` o ``decision_function``."""
    response = getattr(spec.scorer, "_response_method", None)
    if response is None:
        return False
    if isinstance(response, str):
        response = (response,)
    return "predict" not in response


# ---------------------------------------------------------------------------
# Métricas finales sobre el conjunto de prueba
# ---------------------------------------------------------------------------
def holdout_metrics(
    task: str,
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
):
    """Métricas del conjunto de prueba (evaluado una sola vez).

    Conserva las claves de ``SynaptixModel.evaluate`` (``accuracy``,
    ``precision``, ``recall``, ``f1`` / ``MAE``, ``MSE``, ``RMSE``, ``R2``,
    ``MAPE``) y agrega las demás métricas del motor.
    """
    from ... import metrics as sx_metrics

    if task == "classification":
        names = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"]
        if y_proba is not None:
            names += ["roc_auc", "log_loss"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = sx_metrics.classification_metrics(
                y_true, y_pred, metrics=names, y_proba=y_proba, labels=labels
            )
        if y_proba is not None:
            try:
                result["pr_auc"] = pr_auc_score(y_true, y_proba, labels=labels)
            except ValueError:
                result["pr_auc"] = float("nan")
        return result

    result = sx_metrics.regression_metrics(
        y_true, y_pred, metrics=["MAE", "MSE", "RMSE", "R2", "MedAE"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result["MAPE"] = safe_mape(y_true, y_pred)
    return result


def summarize_scores(values: np.ndarray) -> Dict[str, float]:
    """Media, desviación, mejor y peor valor de una serie de folds."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan}
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }
