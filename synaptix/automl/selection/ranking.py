"""
Evaluación de candidatos con validación cruzada y ranking.

Cada candidato se evalúa como un ``Pipeline`` completo (preprocesamiento +
modelo), así el preprocesamiento se ajusta dentro de cada fold. Los errores
se registran por modelo; un modelo fallido nunca puede ganar.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from ..budget import TimeBudget
from ..evaluation.metrics import MetricSpec, needs_proba, summarize_scores
from ..exceptions import AllModelsFailedError

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class CandidateResult:
    """Resultado de validación cruzada de un candidato.

    Attributes
    ----------
    name : str
        Nombre del modelo.
    status : str
        ``ok``, ``partial`` (fallaron métricas secundarias), ``failed`` o
        ``skipped`` (sin presupuesto de tiempo).
    scores : dict
        Puntajes por fold en escala de scikit-learn, por clave de métrica.
    fit_time, score_time : float
        Tiempo medio de ajuste y de predicción por fold (segundos).
    params : dict
        Hiperparámetros del modelo.
    error : str
        Mensaje del error, si lo hubo.
    warnings : list de str
        Advertencias emitidas durante el entrenamiento (sin duplicados).
    """

    name: str
    status: str
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    fit_time: float = float("nan")
    score_time: float = float("nan")
    total_time: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status in (STATUS_OK, STATUS_PARTIAL)


def _supports(model: Any, spec: MetricSpec) -> bool:
    if not needs_proba(spec):
        return True
    response = getattr(spec.scorer, "_response_method", ())
    if isinstance(response, str):
        response = (response,)
    return any(hasattr(model, method) for method in response)


def _short_params(estimator: Any) -> Dict[str, Any]:
    """Parámetros que difieren de los defaults de la clase (para mostrar)."""
    try:
        defaults = type(estimator)().get_params(deep=False)
    except Exception:
        defaults = {}
    params = estimator.get_params(deep=False)
    return {k: v for k, v in params.items()
            if k not in defaults or repr(defaults[k]) != repr(v)}


def evaluate_candidate(
    name: str,
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Any,
    specs: Sequence[MetricSpec],
    groups: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
    pre_dispatch: str = "2*n_jobs",
) -> CandidateResult:
    """Valida un pipeline con todas las métricas y aísla sus errores."""
    model = pipeline.steps[-1][1]
    primary = specs[0]
    usable = [primary] + [s for s in specs[1:] if _supports(model, s)]
    start = time.perf_counter()
    kwargs = dict(cv=cv, groups=groups, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                  error_score="raise")
    status, error, output = STATUS_OK, "", None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            output = cross_validate(pipeline, X, y,
                                    scoring={s.key: s.scorer for s in usable}, **kwargs)
        except Exception as exc:
            if len(usable) > 1:
                try:
                    output = cross_validate(pipeline, X, y,
                                            scoring={primary.key: primary.scorer}, **kwargs)
                    status = STATUS_PARTIAL
                    error = f"Métricas secundarias fallaron: {type(exc).__name__}: {exc}"
                except Exception as exc2:
                    status, error = STATUS_FAILED, f"{type(exc2).__name__}: {exc2}"
            else:
                status, error = STATUS_FAILED, f"{type(exc).__name__}: {exc}"

    messages = list(dict.fromkeys(
        f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught
    ))[:5]
    result = CandidateResult(
        name=name,
        status=status,
        total_time=time.perf_counter() - start,
        params=_short_params(model),
        error=error,
        warnings=messages,
    )
    if output is None:
        return result

    result.scores = {
        key[5:]: np.asarray(values, dtype=float)
        for key, values in output.items() if key.startswith("test_")
    }
    result.fit_time = float(np.mean(output["fit_time"]))
    result.score_time = float(np.mean(output["score_time"]))
    primary_scores = result.scores.get(primary.key)
    if primary_scores is None or np.isnan(primary_scores).any():
        result.status = STATUS_FAILED
        result.error = (result.error + "; " if result.error else "") + (
            f"La métrica principal '{primary.display}' dio NaN en algún fold."
        )
    return result


def evaluate_candidates(
    candidates: Sequence[Any],
    build_pipeline: Callable[[Any], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Any,
    specs: Sequence[MetricSpec],
    groups: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
    pre_dispatch: str = "2*n_jobs",
    budget: Optional[TimeBudget] = None,
    log: Optional[Callable[[str], None]] = None,
) -> List[CandidateResult]:
    """Evalúa todos los candidatos, respetando el presupuesto de tiempo."""
    results = []
    for candidate in candidates:
        if budget is not None and budget.exhausted and any(r.usable for r in results):
            results.append(CandidateResult(candidate.name, STATUS_SKIPPED,
                                           error="Presupuesto de tiempo agotado."))
            continue
        try:
            pipeline = build_pipeline(candidate)
        except Exception as exc:
            results.append(CandidateResult(candidate.name, STATUS_FAILED,
                                           error=f"{type(exc).__name__}: {exc}"))
            continue
        result = evaluate_candidate(candidate.name, pipeline, X, y, cv, specs,
                                    groups=groups, n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        results.append(result)
        if log is not None:
            if result.usable:
                mean = specs[0].to_display(np.mean(result.scores[specs[0].key]))
                log(f"      {candidate.name:<24} {specs[0].display}={float(mean):.4f} "
                    f"({result.total_time:.2f}s)")
            else:
                log(f"      {candidate.name:<24} FALLÓ: {result.error[:80]}")
    return results


def build_ranking(results: Sequence[CandidateResult], specs: Sequence[MetricSpec]) -> pd.DataFrame:
    """Tabla de ranking ordenada por la métrica principal (mejor primero).

    Columnas: ``modelo``, ``estado``, la métrica principal (media de los
    folds), ``desviacion``, ``mejor``, ``peor``, las métricas secundarias,
    tiempos, ``parametros`` y ``error``.
    """
    primary = specs[0]
    rows = []
    for r in results:
        row: Dict[str, Any] = {"modelo": r.name, "estado": r.status}
        raw = r.scores.get(primary.key, np.array([]))
        stats = summarize_scores(primary.to_display(raw) if raw.size else raw)
        best, worst = (stats["max"], stats["min"]) if primary.greater_is_better else (
            stats["min"], stats["max"])
        row.update({primary.display: stats["mean"], "desviacion": stats["std"],
                    "mejor": best, "peor": worst})
        for spec in specs[1:]:
            values = r.scores.get(spec.key)
            row[spec.display] = (summarize_scores(spec.to_display(values))["mean"]
                                 if values is not None else np.nan)
        row.update({
            "tiempo_ajuste_s": round(r.fit_time, 4) if not np.isnan(r.fit_time) else np.nan,
            "tiempo_prediccion_s": (round(r.score_time, 4)
                                    if not np.isnan(r.score_time) else np.nan),
            "tiempo_s": round(r.total_time, 3),
            "parametros": r.params,
            "error": r.error,
        })
        rows.append(row)
    table = pd.DataFrame(rows)
    usable = table["estado"].isin([STATUS_OK, STATUS_PARTIAL])
    table["_order"] = np.where(usable, 0, 1)
    table = table.sort_values(
        ["_order", primary.display],
        ascending=[True, not primary.greater_is_better],
        na_position="last",
        kind="mergesort",
    ).drop(columns="_order").reset_index(drop=True)
    return table


def pick_best(results: Sequence[CandidateResult], specs: Sequence[MetricSpec]) -> CandidateResult:
    """Mejor candidato válido por media de la métrica principal.

    Raises
    ------
    AllModelsFailedError
        Si ningún candidato terminó con estado ``ok`` o ``partial``.
    """
    usable = [r for r in results if r.usable]
    if not usable:
        raise AllModelsFailedError({r.name: r.error or r.status for r in results})
    key = specs[0].key
    return max(usable, key=lambda r: float(np.mean(r.scores[key])))
