"""Evaluación: métricas y estrategias de validación cruzada."""

from .cross_validation import STRATEGIES, holdout_split, is_splitter, make_cv, resolve_strategy
from .metrics import (
    DEFAULT_SCORING,
    DEFAULT_SECONDARY,
    MetricSpec,
    pr_auc_score,
    resolve_metric,
    resolve_metrics,
    safe_mape,
    summarize_scores,
    holdout_metrics,
)

__all__ = [
    "STRATEGIES",
    "holdout_split",
    "is_splitter",
    "make_cv",
    "resolve_strategy",
    "DEFAULT_SCORING",
    "DEFAULT_SECONDARY",
    "MetricSpec",
    "pr_auc_score",
    "resolve_metric",
    "resolve_metrics",
    "safe_mape",
    "summarize_scores",
    "holdout_metrics",
]
