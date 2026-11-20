"""
Feature Subset Search para SynaptIX AutoML.

Búsqueda y evaluación sistemática de subconjuntos de variables por modelo de
Machine Learning, garantizando validación cruzada estricta, prevención total
de data leakage, compatibilidad con pandas/polars y preservación de columnas
en inferencia.
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from ..budget import TimeBudget
from ..evaluation.metrics import MetricSpec, needs_proba
from ..exceptions import AutoMLError
from ..features import build_preprocessor, make_model_pipeline
from ..parallel import native_thread_limits, resolve_n_jobs
from .ranking import (
    CandidateResult,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
    _short_params,
    _supports,
)

SearchStrategy = Literal["none", "exhaustive", "random", "forward", "backward", "auto"]


@dataclass
class SubsetEvaluationRecord:
    """Registro de la evaluación de un subconjunto para un modelo."""

    model: str
    features: List[str]
    n_features: int
    cv_mean: float
    cv_std: float
    fit_time: float
    status: str
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    error: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


def count_combinations(n_total: int, min_k: int, max_k: int) -> int:
    """Calcula la cantidad total teórica de combinaciones posibles."""
    if n_total < 1 or min_k > max_k or min_k > n_total:
        return 0
    actual_max = min(max_k, n_total)
    actual_min = max(min_k, 1)
    total = 0
    for k in range(actual_min, actual_max + 1):
        total += math.comb(n_total, k)
    return total


class FeatureSubsetSearch:
    """Motor de búsqueda de subconjuntos de variables por algoritmo."""

    def __init__(
        self,
        strategy: SearchStrategy = "auto",
        min_features: int = 1,
        max_features: Optional[int] = None,
        max_combinations: Optional[int] = None,
        metric: Union[str, MetricSpec] = "auto",
        n_jobs: Optional[int] = None,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.min_features = max(1, int(min_features))
        self.max_features = max_features
        self.max_combinations = max_combinations
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.results_records_: List[SubsetEvaluationRecord] = []
        self.best_by_model_: Dict[str, Dict[str, Any]] = {}
        self.best_overall_: Optional[Dict[str, Any]] = None
        self.summary_: Dict[str, Any] = {}
        self.results_df_: Optional[pd.DataFrame] = None
        self.ranking_df_: Optional[pd.DataFrame] = None

    def _resolve_effective_strategy(self, n_features: int) -> str:
        """Determina la estrategia efectiva considerando 'auto' y número de variables."""
        strat = self.strategy if self.strategy not in ("auto", None) else "auto"
        if strat != "auto":
            return strat
        if n_features <= 5:
            return "exhaustive"
        if n_features <= 12:
            return "forward"
        return "forward"

    def _generate_exhaustive_stream(
        self, features: Sequence[str], min_k: int, max_k: int, limit: Optional[int] = None
    ) -> Iterator[Tuple[str, ...]]:
        """Generador iterativo (streaming) de combinaciones con itertools."""
        count = 0
        for k in range(min_k, max_k + 1):
            for combo in itertools.combinations(features, k):
                yield combo
                count += 1
                if limit is not None and count >= limit:
                    return

    def _generate_random_subsets(
        self, features: Sequence[str], min_k: int, max_k: int, limit: int, rng: np.random.Generator
    ) -> List[Tuple[str, ...]]:
        """Generador determinista de subconjuntos aleatorios sin duplicados."""
        total_possible = count_combinations(len(features), min_k, max_k)
        if total_possible <= limit:
            # Si el total cabe en el límite, generar todas y mezclar
            all_combos = list(self._generate_exhaustive_stream(features, min_k, max_k))
            rng.shuffle(all_combos)
            return all_combos[:limit]

        subsets: List[Tuple[str, ...]] = []
        seen: Set[Tuple[str, ...]] = set()
        feature_arr = np.asarray(features)
        p = len(features)

        # Probabilidades ponderadas por la cantidad de combinaciones de cada tamaño
        k_values = list(range(min_k, max_k + 1))
        k_weights = [math.comb(p, k) for k in k_values]
        total_weight = sum(k_weights)
        k_probs = [w / total_weight for w in k_weights]

        max_attempts = limit * 50
        attempts = 0
        while len(subsets) < limit and attempts < max_attempts:
            attempts += 1
            k = int(rng.choice(k_values, p=k_probs))
            chosen = tuple(sorted(rng.choice(feature_arr, size=k, replace=False)))
            if chosen not in seen:
                seen.add(chosen)
                subsets.append(chosen)

        return subsets

    def search_for_model(
        self,
        candidate_name: str,
        estimator: Any,
        pipeline_builder: Callable[[Any, Sequence[str]], Any],
        X: pd.DataFrame,
        y: np.ndarray,
        cv: Any,
        specs: Sequence[MetricSpec],
        features: Sequence[str],
        groups: Optional[np.ndarray] = None,
        budget: Optional[TimeBudget] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> Tuple[CandidateResult, List[str]]:
        """Evalúa las combinaciones para un único modelo candidato."""
        primary_spec = specs[0]
        n_features_total = len(features)
        min_k = max(1, min(self.min_features, n_features_total))
        max_k = n_features_total if self.max_features is None else min(self.max_features, n_features_total)
        max_k = max(min_k, max_k)

        effective_strategy = self._resolve_effective_strategy(n_features_total)
        effective_limit = self.max_combinations
        rng = np.random.default_rng(self.random_state)

        model_records: List[SubsetEvaluationRecord] = []
        best_record: Optional[SubsetEvaluationRecord] = None

        def evaluate_one_subset(combo: Sequence[str]) -> SubsetEvaluationRecord:
            nonlocal best_record
            subset_cols = list(combo)
            start_eval = time.perf_counter()

            if budget is not None and budget.exhausted and model_records:
                rec = SubsetEvaluationRecord(
                    model=candidate_name,
                    features=subset_cols,
                    n_features=len(subset_cols),
                    cv_mean=float("nan"),
                    cv_std=float("nan"),
                    fit_time=0.0,
                    status=STATUS_SKIPPED,
                    error="Presupuesto de tiempo agotado.",
                )
                model_records.append(rec)
                self.results_records_.append(rec)
                return rec

            try:
                sub_pipeline = pipeline_builder(clone(estimator), subset_cols)
            except Exception as exc:
                rec = SubsetEvaluationRecord(
                    model=candidate_name,
                    features=subset_cols,
                    n_features=len(subset_cols),
                    cv_mean=float("nan"),
                    cv_std=float("nan"),
                    fit_time=0.0,
                    status=STATUS_FAILED,
                    error=f"Error creando pipeline: {type(exc).__name__}: {exc}",
                )
                model_records.append(rec)
                self.results_records_.append(rec)
                return rec

            usable_specs = [primary_spec] + [s for s in specs[1:] if _supports(estimator, s)]
            scoring_dict = {s.key: s.scorer for s in usable_specs}
            cv_kwargs = dict(
                cv=cv,
                groups=groups,
                n_jobs=self.n_jobs,
                error_score="raise",
            )

            status = STATUS_OK
            error_msg = ""
            output = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    output = cross_validate(
                        sub_pipeline, X[subset_cols], y, scoring=scoring_dict, **cv_kwargs
                    )
                except Exception as exc:
                    if len(usable_specs) > 1:
                        try:
                            output = cross_validate(
                                sub_pipeline,
                                X[subset_cols],
                                y,
                                scoring={primary_spec.key: primary_spec.scorer},
                                **cv_kwargs,
                            )
                            status = STATUS_PARTIAL
                            error_msg = f"Métricas secundarias fallaron: {exc}"
                        except Exception as exc2:
                            status = STATUS_FAILED
                            error_msg = f"{type(exc2).__name__}: {exc2}"
                    else:
                        status = STATUS_FAILED
                        error_msg = f"{type(exc).__name__}: {exc}"

            elapsed = time.perf_counter() - start_eval
            fit_time = float(np.mean(output["fit_time"])) if output is not None else elapsed

            scores: Dict[str, np.ndarray] = {}
            if output is not None:
                for k, v in output.items():
                    if k.startswith("test_"):
                        scores[k[5:]] = np.asarray(v, dtype=float)

            primary_scores = scores.get(primary_spec.key)
            if primary_scores is None or np.isnan(primary_scores).any():
                status = STATUS_FAILED
                error_msg = (error_msg + "; " if error_msg else "") + "La métrica dio NaN."
                cv_mean = float("nan")
                cv_std = float("nan")
            else:
                cv_mean = float(primary_spec.to_display(np.mean(primary_scores)))
                cv_std = float(np.std(primary_scores))

            rec = SubsetEvaluationRecord(
                model=candidate_name,
                features=subset_cols,
                n_features=len(subset_cols),
                cv_mean=cv_mean,
                cv_std=cv_std,
                fit_time=fit_time,
                status=status,
                scores=scores,
                error=error_msg,
                params=_short_params(estimator),
            )
            model_records.append(rec)
            self.results_records_.append(rec)

            if status in (STATUS_OK, STATUS_PARTIAL):
                if best_record is None:
                    best_record = rec
                else:
                    is_better = (
                        cv_mean > best_record.cv_mean
                        if primary_spec.greater_is_better
                        else cv_mean < best_record.cv_mean
                    )
                    if is_better:
                        best_record = rec

            return rec

        # Ejecución según estrategia
        if effective_strategy == "forward":
            # Sequential Forward Selection (SFS)
            current_subset: List[str] = []
            remaining_features = list(features)
            eval_count = 0

            for step in range(1, max_k + 1):
                if not remaining_features:
                    break
                best_feature_this_step = None
                best_record_this_step: Optional[SubsetEvaluationRecord] = None

                for candidate_f in remaining_features:
                    if budget is not None and budget.exhausted and model_records:
                        break
                    if effective_limit is not None and eval_count >= effective_limit:
                        break

                    candidate_combo = current_subset + [candidate_f]
                    rec = evaluate_one_subset(candidate_combo)
                    eval_count += 1

                    if rec.status in (STATUS_OK, STATUS_PARTIAL):
                        if best_record_this_step is None:
                            best_record_this_step = rec
                            best_feature_this_step = candidate_f
                        else:
                            is_step_better = (
                                rec.cv_mean > best_record_this_step.cv_mean
                                if primary_spec.greater_is_better
                                else rec.cv_mean < best_record_this_step.cv_mean
                            )
                            if is_step_better:
                                best_record_this_step = rec
                                best_feature_this_step = candidate_f

                if best_feature_this_step is not None:
                    current_subset.append(best_feature_this_step)
                    remaining_features.remove(best_feature_this_step)
                else:
                    break

                if budget is not None and budget.exhausted:
                    break
                if effective_limit is not None and eval_count >= effective_limit:
                    break

        elif effective_strategy == "backward":
            # Sequential Backward Selection (SBS)
            current_subset = list(features)
            eval_count = 0

            # Si la cantidad de features supera max_k, reducir inicialmente
            if len(current_subset) <= max_k:
                evaluate_one_subset(current_subset)
                eval_count += 1

            while len(current_subset) > min_k:
                if budget is not None and budget.exhausted and model_records:
                    break
                if effective_limit is not None and eval_count >= effective_limit:
                    break

                worst_feature_this_step = None
                best_record_this_step: Optional[SubsetEvaluationRecord] = None

                for feature_to_remove in current_subset:
                    if budget is not None and budget.exhausted:
                        break
                    if effective_limit is not None and eval_count >= effective_limit:
                        break

                    candidate_combo = [f for f in current_subset if f != feature_to_remove]
                    rec = evaluate_one_subset(candidate_combo)
                    eval_count += 1

                    if rec.status in (STATUS_OK, STATUS_PARTIAL):
                        if best_record_this_step is None:
                            best_record_this_step = rec
                            worst_feature_this_step = feature_to_remove
                        else:
                            is_step_better = (
                                rec.cv_mean > best_record_this_step.cv_mean
                                if primary_spec.greater_is_better
                                else rec.cv_mean < best_record_this_step.cv_mean
                            )
                            if is_step_better:
                                best_record_this_step = rec
                                worst_feature_this_step = feature_to_remove

                if worst_feature_this_step is not None:
                    current_subset.remove(worst_feature_this_step)
                else:
                    break

        elif effective_strategy == "random":
            limit = effective_limit or min(50, count_combinations(n_features_total, min_k, max_k))
            combos = self._generate_random_subsets(features, min_k, max_k, limit=limit, rng=rng)
            for combo in combos:
                if budget is not None and budget.exhausted and model_records:
                    break
                evaluate_one_subset(combo)

        else:  # "exhaustive"
            combos_stream = self._generate_exhaustive_stream(
                features, min_k, max_k, limit=effective_limit
            )
            for combo in combos_stream:
                if budget is not None and budget.exhausted and model_records:
                    break
                evaluate_one_subset(combo)

        # Construir CandidateResult para el mejor subconjunto de este modelo
        if best_record is not None:
            chosen_features = list(best_record.features)
            status = best_record.status
            scores = best_record.scores
            fit_time = best_record.fit_time
            error = best_record.error
            self.best_by_model_[candidate_name] = {
                "features": chosen_features,
                "n_features": len(chosen_features),
                "cv_mean": best_record.cv_mean,
                "cv_std": best_record.cv_std,
                "fit_time": fit_time,
            }
        else:
            # Si todas las combinaciones fallaron para este modelo
            chosen_features = list(features[:min_k])
            status = STATUS_FAILED
            scores = {}
            fit_time = float("nan")
            error = "Todas las combinaciones fallaron para este modelo."
            self.best_by_model_[candidate_name] = {
                "features": chosen_features,
                "n_features": len(chosen_features),
                "cv_mean": float("nan"),
                "cv_std": float("nan"),
                "fit_time": fit_time,
                "error": error,
            }

        candidate_result = CandidateResult(
            name=candidate_name,
            status=status,
            scores=scores,
            fit_time=fit_time,
            score_time=float("nan"),
            total_time=sum(r.fit_time for r in model_records),
            params=_short_params(estimator),
            error=error,
            warnings=[],
        )
        return candidate_result, chosen_features

    def finalize(
        self,
        primary_spec: MetricSpec,
        features_all: Sequence[str],
        total_time: float,
        budget: Optional[TimeBudget] = None,
    ) -> None:
        """Consolida resultados, genera rankings y resúmenes."""
        rows = []
        for r in self.results_records_:
            rows.append({
                "model": r.model,
                "features": r.features,
                "n_features": r.n_features,
                "cv_mean": r.cv_mean,
                "cv_std": r.cv_std,
                "fit_time": r.fit_time,
                "status": r.status,
                "error": r.error,
            })

        df_results = pd.DataFrame(rows)
        self.results_df_ = df_results

        # Ranking ordenado
        if not df_results.empty:
            usable = df_results["status"].isin([STATUS_OK, STATUS_PARTIAL])
            df_results_copy = df_results.copy()
            df_results_copy["_order"] = np.where(usable, 0, 1)
            ranking = df_results_copy.sort_values(
                ["_order", "cv_mean"],
                ascending=[True, not primary_spec.greater_is_better],
                na_position="last",
                kind="mergesort",
            ).drop(columns="_order").reset_index(drop=True)
            ranking.insert(0, "rank", range(1, len(ranking) + 1))
            self.ranking_df_ = ranking
        else:
            self.ranking_df_ = pd.DataFrame(
                columns=["rank", "model", "features", "n_features", "cv_mean", "cv_std", "fit_time", "status"]
            )

        # Determinar el mejor global
        n_features_total = len(features_all)
        min_k = max(1, min(self.min_features, n_features_total))
        max_k = n_features_total if self.max_features is None else min(self.max_features, n_features_total)
        total_possible = count_combinations(n_features_total, min_k, max_k)

        usable_records = [r for r in self.results_records_ if r.status in (STATUS_OK, STATUS_PARTIAL)]
        if usable_records:
            if primary_spec.greater_is_better:
                best_rec = max(usable_records, key=lambda r: r.cv_mean)
            else:
                best_rec = min(usable_records, key=lambda r: r.cv_mean)
            self.best_overall_ = {
                "model": best_rec.model,
                "features": list(best_rec.features),
                "n_features": best_rec.n_features,
                "cv_mean": best_rec.cv_mean,
                "cv_std": best_rec.cv_std,
            }
        else:
            self.best_overall_ = None

        term_reason = "completed"
        if budget is not None and budget.exhausted:
            term_reason = "time_budget_exhausted"
        elif any(r.status == STATUS_SKIPPED for r in self.results_records_):
            term_reason = "time_budget_exhausted"
        elif self.max_combinations is not None and len(self.results_records_) >= self.max_combinations:
            term_reason = "max_combinations_reached"

        self.summary_ = {
            "n_original_features": n_features_total,
            "n_possible_combinations": total_possible,
            "n_evaluated_combinations": len(self.results_records_),
            "n_failed_combinations": sum(1 for r in self.results_records_ if r.status == STATUS_FAILED),
            "models_evaluated": list(dict.fromkeys(r.model for r in self.results_records_)),
            "best_by_model": dict(self.best_by_model_),
            "best_overall": self.best_overall_,
            "total_time_seconds": round(total_time, 3),
            "strategy_used": self._resolve_effective_strategy(n_features_total),
            "termination_reason": term_reason,
        }
