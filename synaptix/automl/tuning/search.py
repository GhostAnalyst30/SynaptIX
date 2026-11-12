"""
Hyperparameter Optimization Engine.

Estrategias intercambiables sobre el ``Pipeline`` completo (el
preprocesamiento se reajusta en cada fold):

- ``grid``    : ``GridSearchCV`` sobre una rejilla pequeña.
- ``random``  : ``RandomizedSearchCV`` sobre distribuciones.
- ``halving`` : ``HalvingRandomSearchCV`` (successive halving); con menos
  de 1000 filas no compensa y se usa ``random``.
- ``optuna`` / ``bayesian``: TPE de Optuna con poda por fold (early
  stopping de combinaciones malas). Requiere ``synaptix[automl]``.
- ``auto``    : ``grid`` si la rejilla cabe en ``n_iter``; si no, ``random``.

Con ``time_budget`` la búsqueda en rejilla o aleatoria se ejecuta
candidato a candidato y se detiene al agotar el presupuesto.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    cross_validate,
)

from ..budget import TimeBudget
from ..exceptions import AutoMLError, MissingDependencyError
from .spaces import grid_size, suggest, to_distributions

STRATEGIES = ("auto", "grid", "random", "halving", "optuna", "bayesian")
HALVING_MIN_SAMPLES = 1000


@dataclass
class TuningResult:
    """Resultado de una búsqueda de hiperparámetros.

    Attributes
    ----------
    strategy : str
        Estrategia ejecutada (puede diferir de la pedida, ver ``note``).
    best_params : dict
        Mejores hiperparámetros del modelo (sin prefijo ``model__``).
    best_score : float
        Media de validación cruzada en escala de scikit-learn.
    best_std : float
        Desviación estándar entre folds del mejor candidato.
    estimator : Pipeline
        Pipeline sin entrenar con los mejores parámetros.
    n_evaluated : int
        Combinaciones evaluadas.
    n_failed : int
        Combinaciones cuyo entrenamiento falló.
    trials : DataFrame
        Historial: parámetros, media y desviación.
    seconds : float
        Duración de la búsqueda.
    note : str
        Avisos (cambio de estrategia, presupuesto agotado...).
    """

    strategy: str
    best_params: Dict[str, Any]
    best_score: float
    best_std: float
    estimator: Any
    n_evaluated: int
    n_failed: int = 0
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)
    seconds: float = 0.0
    note: str = ""


def _prefixed(params: Dict[str, Any]) -> Dict[str, Any]:
    return {f"model__{k}": v for k, v in params.items()}


def _strip(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k.split("__", 1)[1] if k.startswith("model__") else k: v for k, v in params.items()}


def tune_pipeline(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Any,
    scorer: Any,
    strategy: str = "auto",
    grid: Optional[Dict[str, list]] = None,
    space: Optional[Dict[str, tuple]] = None,
    distributions: Optional[Dict[str, Any]] = None,
    n_iter: int = 20,
    groups: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
    pre_dispatch: str = "2*n_jobs",
    random_state: Optional[int] = None,
    budget: Optional[TimeBudget] = None,
) -> Optional[TuningResult]:
    """Busca los mejores hiperparámetros del paso ``model`` del pipeline.

    Returns
    -------
    TuningResult o None
        ``None`` si el modelo no tiene espacio de búsqueda.
    """
    if strategy not in STRATEGIES:
        raise AutoMLError(f"tuning debe ser uno de {STRATEGIES} o None (recibido: {strategy!r}).")
    grid = grid or {}
    space = space or {}
    if distributions is None and space:
        distributions = to_distributions(space)
    distributions = distributions or {}
    if not grid and not distributions:
        return None

    note = ""
    if strategy == "auto":
        strategy = "grid" if grid and grid_size(grid) <= n_iter else "random"
    if strategy == "grid" and not grid:
        strategy, note = "random", "Sin rejilla para el modelo: se usó búsqueda aleatoria."
    if strategy in ("random", "halving") and not distributions:
        strategy, note = "grid", "Sin distribuciones para el modelo: se usó la rejilla."
    if strategy == "halving" and len(X) < HALVING_MIN_SAMPLES:
        strategy = "random"
        note = (f"Successive halving no compensa con {len(X)} filas "
                f"(< {HALVING_MIN_SAMPLES}): se usó búsqueda aleatoria.")
    if strategy in ("optuna", "bayesian"):
        if not space:
            if not grid:
                return None
            space = {k: ("cat", list(v)) for k, v in grid.items()}
        return _tune_optuna(pipeline, X, y, cv, scorer, space, n_iter, groups,
                            random_state, budget, strategy)

    start = time.perf_counter()
    if budget is not None and budget.seconds is not None:
        if strategy == "grid":
            candidates = list(ParameterGrid(grid))
        else:
            candidates = list(ParameterSampler(distributions, n_iter=n_iter,
                                               random_state=random_state))
        result = _budgeted_search(pipeline, X, y, cv, scorer, candidates, groups,
                                  n_jobs, pre_dispatch, budget, strategy)
        result.note = "; ".join(filter(None, [note, result.note]))
        result.seconds = time.perf_counter() - start
        return result

    common = dict(scoring=scorer, cv=cv, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                  error_score=np.nan, refit=False)
    if strategy == "grid":
        search = GridSearchCV(pipeline, _prefixed(grid), **common)
    elif strategy == "random":
        search = RandomizedSearchCV(pipeline, _prefixed(distributions), n_iter=n_iter,
                                    random_state=random_state, **common)
    else:
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        common.pop("pre_dispatch")  # HalvingRandomSearchCV no lo acepta
        search = HalvingRandomSearchCV(
            pipeline, _prefixed(distributions), n_candidates=n_iter, factor=3,
            random_state=random_state, **common)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        search.fit(X, y, groups=groups)
    result = _from_cv_results(pipeline, search.cv_results_, strategy)
    result.note = note
    result.seconds = time.perf_counter() - start
    return result


def _from_cv_results(pipeline: Any, cv_results: Dict[str, Any], strategy: str) -> TuningResult:
    means = np.asarray(cv_results["mean_test_score"], dtype=float)
    stds = np.asarray(cv_results["std_test_score"], dtype=float)
    params = list(cv_results["params"])
    if strategy == "halving":
        last = np.asarray(cv_results["iter"]) == np.max(cv_results["iter"])
        valid = last & ~np.isnan(means)
    else:
        valid = ~np.isnan(means)
    if not valid.any():
        raise AutoMLError("Todas las combinaciones de hiperparámetros fallaron.")
    best = int(np.flatnonzero(valid)[np.argmax(means[valid])])
    trials = pd.DataFrame({"params": [_strip(p) for p in params],
                           "mean": means, "std": stds})
    best_params = _strip(params[best])
    return TuningResult(
        strategy=strategy,
        best_params=best_params,
        best_score=float(means[best]),
        best_std=float(stds[best]),
        estimator=clone(pipeline).set_params(**_prefixed(best_params)),
        n_evaluated=len(params),
        n_failed=int(np.isnan(means).sum()),
        trials=trials,
    )


def _budgeted_search(pipeline, X, y, cv, scorer, candidates, groups, n_jobs,
                     pre_dispatch, budget: TimeBudget, strategy: str) -> TuningResult:
    rows: List[Dict[str, Any]] = []
    stopped = False
    for params in candidates:
        if budget.exhausted and rows:
            stopped = True
            break
        estimator = clone(pipeline).set_params(**_prefixed(params))
        try:
            out = cross_validate(estimator, X, y, cv=cv, scoring=scorer, groups=groups,
                                 n_jobs=n_jobs, pre_dispatch=pre_dispatch, error_score="raise")
            scores = out["test_score"]
            rows.append({"params": params, "mean": float(np.mean(scores)),
                         "std": float(np.std(scores))})
        except Exception:
            rows.append({"params": params, "mean": np.nan, "std": np.nan})
    trials = pd.DataFrame(rows)
    valid = trials["mean"].notna()
    if not valid.any():
        raise AutoMLError("Todas las combinaciones de hiperparámetros fallaron.")
    best = trials.loc[valid, "mean"].idxmax()
    best_params = dict(trials.at[best, "params"])
    note = (f"Presupuesto agotado tras {len(rows)} de {len(candidates)} combinaciones."
            if stopped else "")
    return TuningResult(
        strategy=strategy,
        best_params=best_params,
        best_score=float(trials.at[best, "mean"]),
        best_std=float(trials.at[best, "std"]),
        estimator=clone(pipeline).set_params(**_prefixed(best_params)),
        n_evaluated=len(rows),
        n_failed=int((~valid).sum()),
        trials=trials,
        note=note,
    )


def _tune_optuna(pipeline, X, y, cv, scorer, space, n_iter, groups, random_state,
                 budget: Optional[TimeBudget], strategy: str) -> TuningResult:
    try:
        import optuna
    except ImportError as exc:
        raise MissingDependencyError("optuna", "automl", f"tuning='{strategy}'") from exc

    start = time.perf_counter()
    splits = list(cv.split(X, y, groups))
    X_values = X

    def objective(trial):
        params = suggest(trial, space)
        estimator = clone(pipeline).set_params(**_prefixed(params))
        scores = []
        for step, (train_idx, valid_idx) in enumerate(splits):
            fold = clone(estimator).fit(X_values.iloc[train_idx], y[train_idx])
            scores.append(float(scorer(fold, X_values.iloc[valid_idx], y[valid_idx])))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("std", float(np.std(scores)))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    timeout = budget.remaining() if budget is not None else None
    study.optimize(objective, n_trials=n_iter, timeout=timeout, catch=(Exception,))

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise AutoMLError("Optuna no completó ninguna combinación de hiperparámetros.")
    best = study.best_trial
    trials = pd.DataFrame({
        "params": [t.params for t in study.trials],
        "mean": [t.value if t.value is not None else np.nan for t in study.trials],
        "std": [t.user_attrs.get("std", np.nan) for t in study.trials],
        "state": [t.state.name for t in study.trials],
    })
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    n_failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    return TuningResult(
        strategy="optuna",
        best_params=dict(best.params),
        best_score=float(best.value),
        best_std=float(best.user_attrs.get("std", np.nan)),
        estimator=clone(pipeline).set_params(**_prefixed(best.params)),
        n_evaluated=len(study.trials),
        n_failed=n_failed,
        trials=trials,
        seconds=time.perf_counter() - start,
        note=f"TPE (seed={random_state}); {n_pruned} combinaciones podadas por early stopping.",
    )
