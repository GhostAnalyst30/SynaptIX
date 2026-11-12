"""
AutoPipeline 2.0: machine learning automático con metodología verificable.

Flujo de ``fit``:

1. Carga y normalización de la entrada (pandas, polars, NumPy o archivo).
2. Detección de la tarea y perfilado de los datos (advertencias, sin
   eliminar columnas).
3. Separación del conjunto de prueba sobre los datos crudos, antes de
   ajustar ninguna transformación.
4. Comparación de modelos con validación cruzada; cada candidato es un
   ``Pipeline`` (preprocesamiento + modelo) que se reajusta en cada fold.
5. Optimización de hiperparámetros con la misma validación.
6. Ensemble opcional, evaluado con los mismos splits.
7. Entrenamiento final en todo el conjunto de entrenamiento.
8. Evaluación en el conjunto de prueba, una sola vez.
9. Explicabilidad, registro del experimento y reporte.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..base import SynaptixModel
from .budget import TimeBudget
from .config import resolve_preset
from .data.adapters import (
    input_kind,
    load_data,
    pandas_to_polars,
    polars_to_pandas,
    use_polars_profiling,
)
from .data.detection import detect_task_details, infer_feature_types
from .data.intelligence import DataProfile, leakage_checks, profile_data, train_test_overlap
from .ensemble import build_ensemble
from .evaluation.cross_validation import holdout_split, make_cv
from .evaluation.metrics import MetricSpec, holdout_metrics, resolve_metrics
from .exceptions import AutoMLError
from .experiments import ExperimentRecord, to_jsonable
from .explain import (
    CAUSAL_NOTE,
    coefficients,
    diagnostics,
    model_importances,
    permutation_importances,
    shap_values,
)
from .features import (
    build_feature_selector,
    build_preprocessor,
    make_model_pipeline,
    output_feature_names,
)
from .parallel import native_thread_limits, resolve_n_jobs
from .profiling import Profiler
from .selection.ranking import (
    CandidateResult,
    build_ranking,
    evaluate_candidate,
    evaluate_candidates,
    pick_best,
)
from .selection.registry import Candidate, select_candidates
from .tuning.search import STRATEGIES as TUNING_STRATEGIES
from .tuning.search import tune_pipeline
from .tuning.spaces import DEFAULT_GRIDS, SPACES

Task = Literal["classification", "regression"]


def detect_task(y: pd.Series) -> str:
    """Detecta si el target es de clasificación o de regresión.

    Ver :func:`synaptix.automl.data.detect_task_details` para los criterios.
    Emite un ``UserWarning`` si la decisión es ambigua.
    """
    from .data.detection import detect_task as _detect

    return _detect(y)


class _CleanerAdapter:
    """Compatibilidad con ``AutoPipeline.cleaner_`` de versiones anteriores.

    ``transform(df)`` aplica el preprocesamiento aprendido en el
    entrenamiento y devuelve un DataFrame con las features del modelo.
    """

    def __init__(self, transformer: Any, feature_columns: List[str], names: List[str]):
        self.transformer = transformer
        self.feature_columns = feature_columns
        self.names = names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        values = self.transformer.transform(df[self.feature_columns])
        if hasattr(values, "toarray"):
            values = values.toarray()
        return pd.DataFrame(values, columns=self.names, index=df.index)


class AutoPipeline:
    """AutoML de principio a fin: ``fit(df, target)`` y el sistema hace el resto.

    Para principiantes basta con los valores por defecto; cada etapa es
    configurable para usuarios avanzados.

    Parameters
    ----------
    task : {"classification", "regression"}, optional
        Se detecta con varios criterios si no se indica.
    models : list, optional
        Candidatos: nombres (``"random_forest"``, ``"svm"``...), tuplas
        ``(nombre, {parámetros})``, estimadores de scikit-learn o modelos
        SynaptIX. Por defecto, los del ``mode``.
    exclude_models : list de str, optional
        Modelos a excluir de los candidatos.
    mode : {"fast", "balanced", "exhaustive"}, default="balanced"
        Preset de candidatos, folds e iteraciones de búsqueda.
    engine : {"auto", "pandas", "polars"}, default="auto"
        Motor de lectura y perfilado. ``auto`` lee con polars solo si la
        entrada ya es polars, y perfila con polars cuando los datos tienen
        al menos 1 millón de celdas (2-4x más rápido en los benchmarks, con
        el doble de memoria pico; se desactiva con ``memory_optimization``).
        El entrenamiento siempre usa pandas/NumPy (scikit-learn).
    cv : int o splitter, optional
        Folds de validación cruzada o un splitter propio. Por defecto el
        del preset (5 en ``balanced``).
    cv_strategy : str, default="auto"
        ``"kfold"``, ``"stratified"``, ``"repeated"``,
        ``"repeated_stratified"``, ``"group"``, ``"stratified_group"``,
        ``"timeseries"`` o ``"auto"`` (según tarea, grupos y tiempo).
    cv_repeats : int, default=3
        Repeticiones de las estrategias ``repeated*``.
    scoring : str o callable, optional
        Métrica principal (``"accuracy"``, ``"f1_weighted"``, ``"rmse"``,
        ``"neg_root_mean_squared_error"``...). Por defecto accuracy / r2.
    metrics : list de str, optional
        Métricas secundarias del ranking.
    tune : bool, default=True
        Activa la búsqueda de hiperparámetros (equivale a ``tuning=None``
        si es False).
    tuning : {"auto", "grid", "random", "halving", "optuna", "bayesian"}, default="auto"
        Estrategia de búsqueda. ``bayesian`` usa Optuna (TPE).
    n_iter : int, optional
        Combinaciones a evaluar en las búsquedas aleatorias/bayesianas.
    tune_top_k : int, optional
        Cuántos modelos del ranking se optimizan.
    time_budget : float, optional
        Segundos máximos para la selección y la búsqueda.
    param_grids : dict, optional
        Rejillas propias por modelo, p. ej. ``{"random_forest": {"n_estimators": [300]}}``.
    param_distributions : dict, optional
        Distribuciones por modelo: listas, distribuciones de scipy o tuplas
        ``("float", low, high, log)`` / ``("int", low, high, log)``.
    test_size : float, int o None, default=0.2
        Fracción (o filas) del conjunto de prueba. ``None`` para no reservar
        prueba (solo validación cruzada).
    impute_strategy : {"mean", "median", "most_frequent", "constant", "knn"}, default="median"
    encode_method : {"onehot", "ordinal"}, default="onehot"
        (``"label"`` se acepta como alias de ``"ordinal"``).
    scale_method : {"standard", "minmax", "robust"} o None, default="standard"
    max_onehot_categories : int, default=20
        Categorías máximas por variable en el one-hot (el resto: "infrequent").
    drop_constant : bool, default=False
        Elimina features constantes dentro de cada fold.
    feature_selection : {"variance", "kbest", "model"}, optional
        Selección de features dentro de cada fold.
    feature_selection_k : int o float, optional
        Número (o proporción) de features para ``kbest``.
    ensemble : {"voting", "stacking", "blending"}, optional
        Construye un ensemble con los mejores modelos; solo se elige si
        mejora la validación cruzada.
    ensemble_top_k : int, default=3
        Modelos que entran al ensemble.
    explain : bool, default=True
        Calcula importancias (permutación en el conjunto de prueba,
        nativas o coeficientes) y diagnósticos de errores.
    ignore_columns : list de str, optional
        Columnas que no se usan como features (no se leen con polars lazy).
    n_jobs : int, default=1
        Procesos para la validación cruzada y la búsqueda (un solo nivel de
        paralelismo; los modelos internos usan 1).
    pre_dispatch : str, default="2*n_jobs"
        Tareas despachadas por adelantado (controla la memoria).
    memory_optimization : bool, default=False
        One-hot disperso para los modelos que lo admiten.
    cache : bool, default=False
        Cachea el preprocesamiento entre combinaciones de hiperparámetros
        (``joblib.Memory`` en un directorio temporal que se borra al final).
    random_state : int, default=42
        Semilla de splits, modelos y búsquedas.
    verbose : bool, default=True
        Imprime el progreso de cada etapa.

    Attributes
    ----------
    task_ : str
    target_ : str
    profile_ : DataProfile
        Perfil y advertencias de los datos.
    ranking_ : DataFrame
        Comparación de modelos (validación cruzada en train).
    best_name_ : str
    best_model_ : SynaptixModel
        Modelo final; recibe las features ya transformadas.
    best_params_ : dict o None
        Hiperparámetros elegidos por la búsqueda (None si no se optimizó).
    pipeline_ : sklearn.pipeline.Pipeline
        Preprocesamiento + modelo, entrenado; recibe datos crudos.
    cleaner_ : objeto con ``transform(df)``
        Preprocesamiento ajustado (compatibilidad).
    cv_score_ : dict
        Media y desviación de la métrica principal del modelo final en CV.
    results_ : dict
        Métricas del conjunto de prueba.
    feature_importances_ : DataFrame o None
    diagnostics_ : dict o None
    timings_ : dict
        Segundos por etapa.
    experiment_ : ExperimentRecord
    warnings_ : list de str

    Ejemplo
    -------
    >>> import synaptix as sx
    >>> pipe = sx.AutoPipeline()
    >>> pipe.fit(df, target="species")
    >>> pipe.report()
    >>> predicciones = pipe.predict(df_nuevo)
    """

    def __init__(
        self,
        task: Optional[Task] = None,
        models: Optional[Sequence[Any]] = None,
        exclude_models: Optional[Sequence[str]] = None,
        mode: Literal["fast", "balanced", "exhaustive"] = "balanced",
        engine: Literal["auto", "pandas", "polars"] = "auto",
        cv: Optional[Any] = None,
        cv_strategy: str = "auto",
        cv_repeats: int = 3,
        scoring: Optional[Union[str, Callable]] = None,
        metrics: Optional[Sequence[str]] = None,
        tune: bool = True,
        tuning: Optional[str] = "auto",
        n_iter: Optional[int] = None,
        tune_top_k: Optional[int] = None,
        time_budget: Optional[float] = None,
        param_grids: Optional[Dict[str, dict]] = None,
        param_distributions: Optional[Dict[str, dict]] = None,
        test_size: Optional[Union[float, int]] = 0.2,
        impute_strategy: str = "median",
        encode_method: str = "onehot",
        scale_method: Optional[str] = "standard",
        max_onehot_categories: Optional[int] = 20,
        drop_constant: bool = False,
        feature_selection: Optional[str] = None,
        feature_selection_k: Optional[Union[int, float]] = None,
        ensemble: Optional[str] = None,
        ensemble_top_k: int = 3,
        explain: bool = True,
        ignore_columns: Optional[Sequence[str]] = None,
        n_jobs: Optional[int] = 1,
        pre_dispatch: str = "2*n_jobs",
        memory_optimization: bool = False,
        cache: bool = False,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.task = task
        self.models = models
        self.exclude_models = exclude_models
        self.mode = mode
        self.engine = engine
        self.cv = cv
        self.cv_strategy = cv_strategy
        self.cv_repeats = cv_repeats
        self.scoring = scoring
        self.metrics = metrics
        self.tune = tune
        self.tuning = tuning
        self.n_iter = n_iter
        self.tune_top_k = tune_top_k
        self.time_budget = time_budget
        self.param_grids = param_grids or {}
        self.param_distributions = param_distributions or {}
        self.test_size = test_size
        self.impute_strategy = impute_strategy
        self.encode_method = encode_method
        self.scale_method = scale_method
        self.max_onehot_categories = max_onehot_categories
        self.drop_constant = drop_constant
        self.feature_selection = feature_selection
        self.feature_selection_k = feature_selection_k
        self.ensemble = ensemble
        self.ensemble_top_k = ensemble_top_k
        self.explain = explain
        self.ignore_columns = list(ignore_columns or [])
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.memory_optimization = memory_optimization
        self.cache = cache
        self.random_state = random_state
        self.verbose = verbose
        self._validate_params()
        self._reset()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------

    _PUBLIC_PARAMS = (
        "task", "models", "exclude_models", "mode", "engine", "cv", "cv_strategy",
        "cv_repeats", "scoring", "metrics", "tune", "tuning", "n_iter", "tune_top_k",
        "time_budget", "param_grids", "param_distributions", "test_size",
        "impute_strategy", "encode_method", "scale_method", "max_onehot_categories",
        "drop_constant", "feature_selection", "feature_selection_k", "ensemble",
        "ensemble_top_k", "explain", "ignore_columns", "n_jobs", "pre_dispatch",
        "memory_optimization", "cache", "random_state", "verbose",
    )

    def get_params(self) -> Dict[str, Any]:
        """Parámetros de configuración del pipeline."""
        return {name: getattr(self, name) for name in self._PUBLIC_PARAMS}

    def _validate_params(self) -> None:
        if self.task not in (None, "classification", "regression"):
            raise AutoMLError(f"task debe ser 'classification', 'regression' o None "
                              f"(recibido: {self.task!r}).")
        if self.tuning is not None and self.tuning not in TUNING_STRATEGIES:
            raise AutoMLError(f"tuning debe ser uno de {TUNING_STRATEGIES} o None "
                              f"(recibido: {self.tuning!r}).")
        if self.ensemble not in (None, "voting", "stacking", "blending"):
            raise AutoMLError(f"ensemble debe ser 'voting', 'stacking', 'blending' o None "
                              f"(recibido: {self.ensemble!r}).")
        resolve_preset(self.mode)
        resolve_n_jobs(self.n_jobs)

    def _reset(self) -> None:
        self.task_: Optional[str] = None
        self.target_: Optional[str] = None
        self.task_detection_ = None
        self.profile_: Optional[DataProfile] = None
        self.feature_columns_: List[str] = []
        self.feature_types_: Dict[str, List[str]] = {}
        self.classes_: Optional[np.ndarray] = None
        self.cv_strategy_: Optional[str] = None
        self.holdout_: Optional[str] = None
        self.metric_specs_: List[MetricSpec] = []
        self.candidate_results_: List[CandidateResult] = []
        self.ranking_: Optional[pd.DataFrame] = None
        self.tuning_results_: Dict[str, Any] = {}
        self.ensemble_result_: Optional[Dict[str, Any]] = None
        self.best_name_: Optional[str] = None
        self.best_params_: Optional[dict] = None
        self.best_model_ = None
        self.pipeline_ = None
        self.cleaner_ = None
        self.cv_score_: Dict[str, float] = {}
        self.results_: Optional[dict] = None
        self.test_score_: Optional[float] = None
        self.feature_importances_: Optional[pd.DataFrame] = None
        self.model_importances_: Optional[pd.DataFrame] = None
        self.coefficients_: Optional[pd.DataFrame] = None
        self.diagnostics_: Optional[dict] = None
        self.timings_: Dict[str, float] = {}
        self.profiler_: Optional[Profiler] = None
        self.experiment_: Optional[ExperimentRecord] = None
        self.warnings_: List[str] = []
        self.notes_: List[str] = []
        self.n_train_: int = 0
        self.n_test_: int = 0
        self._eval: Dict[str, Any] = {}
        self._input_kind: Optional[str] = None
        self._engine_used: Optional[str] = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _warn(self, message: str) -> None:
        self.warnings_.append(message)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Any,
        target: Union[str, int, Sequence],
        groups: Optional[Union[str, Sequence]] = None,
        time_column: Optional[str] = None,
    ) -> "AutoPipeline":
        """Ejecuta el AutoML completo.

        Parameters
        ----------
        data : DataFrame, polars.DataFrame, polars.LazyFrame, ndarray o ruta
            Datos crudos (pueden tener nulos, categóricas y fechas).
        target : str
            Columna objetivo. Con NumPy: índice de columna o vector ``y``.
        groups : str o array-like, optional
            Columna (o vector) de grupos: las filas de un grupo nunca se
            reparten entre entrenamiento y validación/prueba. Útil para
            pacientes, clientes o zonas geográficas.
        time_column : str, optional
            Columna temporal: los datos se ordenan por ella, la prueba son
            las últimas observaciones y la validación es ``TimeSeriesSplit``.

        Returns
        -------
        AutoPipeline
        """
        self._validate_params()
        self._reset()
        profiler = Profiler()
        self.profiler_ = profiler
        budget = TimeBudget(self.time_budget)
        cfg = resolve_preset(self.mode, cv=self.cv, n_iter=self.n_iter,
                             tune_top_k=self.tune_top_k)

        # 1. Carga ------------------------------------------------------
        drop_cols = [c for c in self.ignore_columns if c != target]
        loaded, target = load_data(data, target, self.engine, drop_columns=drop_cols)
        profiler.add("carga", loaded.load_seconds)
        if loaded.convert_seconds:
            profiler.add("conversion_a_pandas", loaded.convert_seconds)
        self._input_kind, self._engine_used = loaded.kind, loaded.engine
        df = loaded.frame
        self.target_ = target

        group_col = groups if isinstance(groups, str) else None
        for col in filter(None, [group_col, time_column]):
            if col not in df.columns:
                raise AutoMLError(f"La columna '{col}' no existe en los datos.")
        if group_col is not None:
            group_values = df[group_col].to_numpy()
        elif groups is not None:
            group_values = np.asarray(groups)
            if len(group_values) != len(df):
                raise AutoMLError("groups debe tener una entrada por fila.")
        else:
            group_values = None

        with profiler.stage("analisis_datos"):
            if time_column is not None:
                order = np.argsort(df[time_column].to_numpy(), kind="stable")
                if not np.array_equal(order, np.arange(len(df))):
                    df = df.iloc[order]
                    if group_values is not None:
                        group_values = group_values[order]

            y_all = df[target]
            if y_all.notna().sum() == 0:
                raise AutoMLError(f"El target '{target}' no tiene valores.")
            detection = detect_task_details(y_all.dropna())
            self.task_detection_ = detection
            if self.task is not None:
                self.task_ = self.task
                if detection.confidence == "high" and detection.task != self.task:
                    self._warn(f"task='{self.task}' indicado, pero el target parece "
                               f"'{detection.task}' ({'; '.join(detection.reasons)}).")
            else:
                self.task_ = detection.task
                if detection.confidence == "low":
                    self._warn(f"Detección de tarea ambigua: se eligió '{detection.task}' "
                               f"({'; '.join(detection.reasons)}). Fíjala con task=.")

            polars_frame = loaded.polars_frame
            if polars_frame is None and use_polars_profiling(
                    self.engine, df.size, self.memory_optimization):
                polars_frame = pandas_to_polars(df)
                if polars_frame is None:
                    self.notes_.append("No se pudo convertir a polars para el perfilado; "
                                       "se usó pandas.")
            profile = profile_data(df, target, self.task_, polars_frame=polars_frame)
            del polars_frame
            self.profile_ = profile

            mask = y_all.notna().to_numpy()
            excluded = {target, time_column, group_col} - {None}
            self.feature_columns_ = [c for c in df.columns if c not in excluded]
            if time_column is not None:
                self.notes_.append(f"'{time_column}' se usa para ordenar y validar; "
                                   "no entra como feature.")
            X = df.loc[mask, self.feature_columns_]
            y = self._prepare_target(y_all[mask])
            if group_values is not None:
                group_values = group_values[mask]
            self.feature_types_ = infer_feature_types(X)

        self._log(f"[1/6] Datos: {len(X)} filas, {X.shape[1]} features "
                  f"(engine={loaded.engine}); tarea: {self.task_}")

        # 2. Holdout ----------------------------------------------------
        is_temporal = time_column is not None
        if self.test_size is None:
            train_idx, test_idx = np.arange(len(X)), np.array([], dtype=int)
            self.holdout_ = "sin conjunto de prueba"
            self._warn("test_size=None: no hay evaluación final independiente; "
                       "solo se reporta validación cruzada.")
        else:
            train_idx, test_idx, self.holdout_ = holdout_split(
                len(X), self.test_size, self.task_, y, group_values, is_temporal,
                self.random_state)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        groups_train = None if group_values is None else group_values[train_idx]
        self.n_train_, self.n_test_ = len(X_train), len(X_test)
        if self.task_ == "classification":
            self.classes_ = np.unique(y_train)
            if len(self.classes_) < 2:
                raise AutoMLError("El conjunto de entrenamiento tiene una sola clase.")
            missing = set(np.unique(y_test)) - set(self.classes_)
            if missing:
                raise AutoMLError(f"Clases del conjunto de prueba ausentes en train: {missing}.")

        with profiler.stage("chequeos_fuga"):
            for warning in leakage_checks(X_train, pd.Series(y_train), self.task_, target,
                                          self.random_state):
                profile.warnings.append(warning)
            if len(X_test) and len(X) <= 1_000_000:
                overlap = train_test_overlap(X_train, X_test)
                if overlap:
                    profile.add("train_test_overlap",
                                f"{overlap} filas de prueba son idénticas a filas de "
                                "entrenamiento: la evaluación puede ser optimista.")
        self.warnings_.extend(f"[{w.code}] {w.column + ': ' if w.column else ''}{w.message}"
                              for w in profile.warnings if w.severity == "warning")

        # 3. Métricas y validación -------------------------------------
        specs = resolve_metrics(self.scoring, self.metrics, self.task_,
                                labels=self.classes_)
        self.metric_specs_ = specs
        if (self.scoring is None and self.task_ == "classification"
                and any(w.code == "class_imbalance" for w in profile.warnings)):
            self._warn("La métrica principal es accuracy con clases desbalanceadas; "
                       "considera scoring='balanced_accuracy' o 'f1_weighted'.")
        cv_splitter, self.cv_strategy_ = make_cv(
            cfg["cv"], self.cv_strategy, self.task_, y=y_train, groups=groups_train,
            is_temporal=is_temporal, n_repeats=self.cv_repeats,
            random_state=self.random_state)
        self._log(f"[2/6] Holdout {self.holdout_}: {self.n_train_} train / "
                  f"{self.n_test_} test. Validación: {self.cv_strategy_}. "
                  f"Métrica: {specs[0].display}")

        # 4. Selección de modelos --------------------------------------
        candidates, notes = select_candidates(
            self.task_, self.mode, self.models, self.exclude_models,
            n_samples=self.n_train_, random_state=self.random_state)
        self.notes_.extend(notes)
        cache_dir = tempfile.mkdtemp(prefix="synaptix_cache_") if self.cache else None
        memory = None
        if cache_dir is not None:
            from joblib import Memory

            memory = Memory(location=cache_dir, verbose=0)

        def build(candidate: Candidate, estimator: Any = None) -> Any:
            sparse = self.memory_optimization and candidate.supports_sparse
            return self._make_pipeline(estimator if estimator is not None
                                       else candidate.estimator, sparse, memory)

        try:
            with native_thread_limits(self.n_jobs), profiler.stage("seleccion_modelos"):
                results = evaluate_candidates(
                    candidates, build, X_train, y_train, cv_splitter, specs,
                    groups=groups_train, n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch,
                    budget=budget, log=self._log if self.verbose else None)
            self.candidate_results_ = results
            best = pick_best(results, specs)
            n_ok = sum(r.usable for r in results)
            self._log(f"[3/6] {n_ok}/{len(results)} modelos válidos. Mejor en CV: "
                      f"{best.name} ({specs[0].display}="
                      f"{float(specs[0].to_display(np.mean(best.scores[specs[0].key]))):.4f})")
            for r in results:
                if r.status == "failed":
                    self._warn(f"Modelo '{r.name}' falló y quedó fuera: {r.error}")

            by_name = {c.name: c for c in candidates}
            chosen = self._tune(results, by_name, build, X_train, y_train, cv_splitter,
                                specs, groups_train, cfg, budget, profiler)

            if self.ensemble:
                chosen = self._try_ensemble(results, by_name, chosen, X_train, y_train,
                                            cv_splitter, specs, groups_train, memory,
                                            is_temporal, profiler)

            # 7. Entrenamiento final -----------------------------------
            with native_thread_limits(self.n_jobs), profiler.stage("entrenamiento_final"):
                final = clone(chosen["pipeline"])
                final.fit(X_train, y_train)
                final.memory = None
        finally:
            if cache_dir is not None:
                shutil.rmtree(cache_dir, ignore_errors=True)

        self.pipeline_ = final
        self.best_name_ = chosen["name"]
        self.best_params_ = chosen.get("tuned_params")
        self.cv_score_ = {"mean": chosen["cv_mean"], "std": chosen["cv_std"],
                          "metric": specs[0].display, "source": chosen["source"]}
        self.ranking_ = build_ranking(self.candidate_results_, specs)
        self._set_compat_attributes(chosen)
        self._log(f"[5/6] Modelo final: {self.best_name_} "
                  f"(CV {specs[0].display}={self.cv_score_['mean']:.4f} "
                  f"± {self.cv_score_['std']:.4f})")

        # 8. Evaluación en prueba --------------------------------------
        if len(X_test):
            with profiler.stage("inferencia_test"):
                y_pred = final.predict(X_test)
                proba = final.predict_proba(X_test) if (
                    self.task_ == "classification" and hasattr(final, "predict_proba")) else None
            self.results_ = holdout_metrics(self.task_, y_test, y_pred, proba,
                                            labels=self.classes_)
            raw = specs[0].scorer(final, X_test, y_test)
            self.test_score_ = float(specs[0].to_display(raw))
            self._eval = {"y_true": np.asarray(y_test), "y_pred": np.asarray(y_pred),
                          "proba": proba, "index": np.asarray(X_test.index)}
            self._log(f"[6/6] Prueba (evaluada una sola vez): {specs[0].display}="
                      f"{self.test_score_:.4f}")
        else:
            self.results_ = {}

        # 9. Explicabilidad --------------------------------------------
        if self.explain:
            with profiler.stage("explicabilidad"):
                self._explain(final, X_test, y_test, specs)

        self.timings_ = profiler.to_dict()
        self.experiment_ = self._build_experiment(X, profile)
        return self

    # ------------------------------------------------------------------

    def _prepare_target(self, y: pd.Series) -> np.ndarray:
        if self.task_ == "classification":
            values = y.to_numpy()
            if values.dtype == object or str(values.dtype).startswith(("str", "string")):
                values = values.astype(str).astype(object)
            n_classes = len(pd.unique(values))
            if n_classes < 2:
                raise AutoMLError("El target tiene una sola clase: no se puede clasificar.")
            return values
        numeric = pd.to_numeric(y, errors="coerce")
        if numeric.isna().any():
            raise AutoMLError("Para regresión el target debe ser numérico.")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AutoMLError("El target contiene valores infinitos.")
        if np.unique(values).size < 2:
            raise AutoMLError("El target es constante: no hay nada que predecir.")
        return values

    def _make_pipeline(self, estimator: Any, sparse: bool, memory: Any = None) -> Any:
        preprocessor = build_preprocessor(
            self.feature_types_,
            impute_strategy=self.impute_strategy,
            scale_method=self.scale_method,
            encode_method=self.encode_method,
            max_onehot_categories=self.max_onehot_categories,
            sparse=sparse,
        )
        selector = build_feature_selector(self.feature_selection, self.task_,
                                          self.feature_selection_k, self.random_state)
        return make_model_pipeline(preprocessor, clone(estimator),
                                   drop_constant=self.drop_constant,
                                   selector=selector, memory=memory)

    def _tune(self, results, by_name, build, X, y, cv, specs, groups, cfg, budget,
              profiler) -> Dict[str, Any]:
        """Optimiza los mejores modelos y devuelve el elegido por CV."""
        key = specs[0].key
        usable = sorted((r for r in results if r.usable),
                        key=lambda r: float(np.mean(r.scores[key])), reverse=True)
        best = usable[0]

        def entry(result: CandidateResult) -> Dict[str, Any]:
            scores = result.scores[key]
            return {"name": result.name, "pipeline": build(by_name[result.name]),
                    "cv_raw": float(np.mean(scores)),
                    "cv_mean": float(specs[0].to_display(np.mean(scores))),
                    "cv_std": float(np.std(scores)), "source": "default",
                    "tuned_params": None, "estimator": by_name[result.name].estimator}

        chosen = entry(best)
        strategy = self.tuning if self.tune else None
        if strategy is None:
            self._log("[4/6] Sin búsqueda de hiperparámetros (tune=False o tuning=None).")
            return chosen

        with native_thread_limits(self.n_jobs), profiler.stage("optimizacion"):
            for result in usable[: cfg["tune_top_k"]]:
                if budget.exhausted:
                    self.notes_.append("Presupuesto de tiempo agotado antes de optimizar "
                                       f"'{result.name}'.")
                    break
                candidate = by_name[result.name]
                spec_name = candidate.spec.name if candidate.spec else None
                user_grid = self.param_grids.get(candidate.name) or self.param_grids.get(
                    spec_name or "")
                user_dist = self.param_distributions.get(candidate.name) or \
                    self.param_distributions.get(spec_name or "")
                grid = user_grid or DEFAULT_GRIDS.get(spec_name or "", {})
                space, distributions = _split_distributions(user_dist)
                if space is None and distributions is None:
                    space = SPACES.get(spec_name or "", {})
                local_strategy = strategy
                if user_grid and strategy == "auto":
                    local_strategy = "grid"
                try:
                    outcome = tune_pipeline(
                        build(candidate), X, y, cv, specs[0].scorer, local_strategy,
                        grid=grid, space=space, distributions=distributions,
                        n_iter=cfg["n_iter"], groups=groups, n_jobs=self.n_jobs,
                        pre_dispatch=self.pre_dispatch, random_state=self.random_state,
                        budget=budget)
                except AutoMLError as exc:
                    self._warn(f"Búsqueda de '{result.name}' falló: {exc}")
                    continue
                if outcome is None:
                    self.notes_.append(f"'{result.name}' no tiene hiperparámetros que "
                                       "optimizar; se usan sus valores por defecto.")
                    continue
                self.tuning_results_[result.name] = outcome
                if outcome.note:
                    self.notes_.append(f"{result.name}: {outcome.note}")
                default_raw = float(np.mean(result.scores[key]))
                self._log(f"[4/6] {result.name}: {outcome.strategy} "
                          f"({outcome.n_evaluated} combinaciones, {outcome.seconds:.1f}s) "
                          f"-> {specs[0].display}="
                          f"{float(specs[0].to_display(outcome.best_score)):.4f} "
                          f"(defaults: {float(specs[0].to_display(default_raw)):.4f})")
                if outcome.best_score > chosen["cv_raw"] or (
                        result.name == chosen["name"] and outcome.best_score >= default_raw):
                    chosen = {
                        "name": result.name, "pipeline": outcome.estimator,
                        "cv_raw": outcome.best_score,
                        "cv_mean": float(specs[0].to_display(outcome.best_score)),
                        "cv_std": outcome.best_std, "source": f"tuning:{outcome.strategy}",
                        "tuned_params": outcome.best_params,
                        "estimator": outcome.estimator.steps[-1][1],
                    }
        return chosen

    def _try_ensemble(self, results, by_name, chosen, X, y, cv, specs, groups, memory,
                      is_temporal, profiler) -> Dict[str, Any]:
        if self.ensemble == "stacking" and is_temporal:
            self._warn("Stacking omitido: su validación interna mezclaría pasado y futuro.")
            return chosen
        key = specs[0].key
        usable = sorted((r for r in results if r.usable),
                        key=lambda r: float(np.mean(r.scores[key])), reverse=True)
        members = []
        for r in usable[: self.ensemble_top_k]:
            outcome = self.tuning_results_.get(r.name)
            estimator = (outcome.estimator.steps[-1][1] if outcome is not None
                         else by_name[r.name].estimator)
            members.append((r.name, estimator))
        if len(members) < 2:
            self._warn("Ensemble omitido: se necesitan al menos 2 modelos válidos.")
            return chosen
        name = f"ensemble_{self.ensemble}"
        with native_thread_limits(self.n_jobs), profiler.stage("ensemble"):
            estimator = build_ensemble(self.ensemble, self.task_, members, self.random_state,
                                       shuffle=not is_temporal)
            pipeline = self._make_pipeline(estimator, sparse=False, memory=memory)
            result = evaluate_candidate(name, pipeline, X, y, cv, specs, groups=groups,
                                        n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        self.candidate_results_.append(result)
        info = {"name": name, "members": [m for m, _ in members], "status": result.status}
        if not result.usable:
            info["error"] = result.error
            self._warn(f"El ensemble falló: {result.error}")
            self.ensemble_result_ = info
            return chosen
        raw = float(np.mean(result.scores[key]))
        info.update({"cv_mean": float(specs[0].to_display(raw)),
                     "cv_std": float(np.std(result.scores[key])),
                     "best_single": chosen["name"], "best_single_cv": chosen["cv_mean"]})
        info["selected"] = raw > chosen["cv_raw"]
        self.ensemble_result_ = info
        self._log(f"      {name}: {specs[0].display}={info['cv_mean']:.4f} "
                  f"({'elegido' if info['selected'] else 'no supera al mejor individual'})")
        if not info["selected"]:
            return chosen
        return {"name": name, "pipeline": pipeline, "cv_raw": raw,
                "cv_mean": info["cv_mean"], "cv_std": info["cv_std"],
                "source": "ensemble", "tuned_params": chosen.get("tuned_params"),
                "estimator": estimator}

    def _set_compat_attributes(self, chosen: Dict[str, Any]) -> None:
        final = self.pipeline_
        model = final.steps[-1][1]
        try:
            names = output_feature_names(final)
        except Exception:
            names = None
        candidate = next((r for r in self.candidate_results_ if r.name == chosen["name"]), None)
        spec = None
        if candidate is not None and not chosen["name"].startswith("ensemble_"):
            from .selection.registry import SPECS

            spec = SPECS[self.task_].get(chosen["name"])
        if spec is not None:
            wrapper = spec.wrapper()
            wrapper.model = model
        else:
            wrapper = SynaptixModel(model, name=chosen["name"])
            wrapper.task = self.task_
        wrapper.fitted = True
        wrapper.feature_names_ = names
        self.best_model_ = wrapper
        transformer = final[:-1]
        self.cleaner_ = _CleanerAdapter(transformer, self.feature_columns_,
                                        names or [f"x{i}" for i in range(
                                            getattr(model, "n_features_in_", 0))])

    def _explain(self, final, X_test, y_test, specs) -> None:
        self.model_importances_ = model_importances(final)
        try:
            self.coefficients_ = coefficients(final, self.classes_)
        except Exception:
            self.coefficients_ = None
        if len(X_test):
            try:
                self.feature_importances_ = permutation_importances(
                    final, X_test, y_test, specs[0].scorer, n_repeats=5,
                    random_state=self.random_state, n_jobs=self.n_jobs)
            except Exception as exc:
                self._warn(f"No se pudo calcular la importancia por permutación: {exc}")
            self.diagnostics_ = diagnostics(self.task_, self._eval["y_true"],
                                            self._eval["y_pred"], labels=self.classes_,
                                            index=self._eval["index"])
        else:
            self.notes_.append("Sin conjunto de prueba no se calcula la importancia por "
                               "permutación (sobre train sería optimista).")

    def _build_experiment(self, X: pd.DataFrame, profile: DataProfile) -> ExperimentRecord:
        config = to_jsonable({k: v for k, v in self.get_params().items() if k != "models"})
        config["models"] = None if self.models is None else [
            m if isinstance(m, (str, tuple)) else type(getattr(m, "model", m)).__name__
            for m in self.models]
        tuning = {name: {"strategy": r.strategy, "best_params": r.best_params,
                         "best_score": r.best_score, "n_evaluated": r.n_evaluated,
                         "n_failed": r.n_failed, "seconds": r.seconds, "note": r.note}
                  for name, r in self.tuning_results_.items()}
        record = ExperimentRecord(
            config=config,
            environment=ExperimentRecord.environment_info(),
            random_state=self.random_state,
            data={
                "input": self._input_kind, "engine": self._engine_used,
                "n_rows": int(len(X)), "n_features": int(X.shape[1]),
                "target": self.target_, "task": self.task_,
                "task_detection": None if self.task_detection_ is None else {
                    "task": self.task_detection_.task,
                    "confidence": self.task_detection_.confidence,
                    "reasons": self.task_detection_.reasons},
                "feature_types": self.feature_types_,
                "classes": None if self.classes_ is None else list(map(str, self.classes_)),
                "holdout": self.holdout_, "n_train": self.n_train_, "n_test": self.n_test_,
                "cv_strategy": self.cv_strategy_,
                "metric": self.metric_specs_[0].display,
                "profile_warnings": [w.__dict__ for w in profile.warnings],
            },
            ranking=to_jsonable(self.ranking_.to_dict(orient="records")),
            tuning=tuning,
            ensemble=self.ensemble_result_ or {},
            selected_model=self.best_name_ or "",
            validation=dict(self.cv_score_),
            test_metrics=dict(self.results_ or {}),
            timings=dict(self.timings_),
            warnings=list(self.warnings_) + [f"[nota] {n}" for n in self.notes_],
        )
        return record

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def _new_frame(self, data: Any) -> pd.DataFrame:
        kind = input_kind(data)
        if kind == "pandas":
            frame = data
        elif kind == "polars":
            frame = polars_to_pandas(data)
        elif kind == "polars_lazy":
            frame = polars_to_pandas(data.collect())
        elif kind == "numpy":
            arr = np.asarray(data)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_columns_):
                raise AutoMLError(f"Se esperaban {len(self.feature_columns_)} columnas.")
            frame = pd.DataFrame(arr, columns=self.feature_columns_)
        else:
            raise AutoMLError("predict acepta DataFrames (pandas/polars) o arrays NumPy.")
        missing = [c for c in self.feature_columns_ if c not in frame.columns]
        if missing:
            raise AutoMLError(f"Faltan columnas usadas en el entrenamiento: {missing}")
        return frame[self.feature_columns_]

    def predict(self, df: Any) -> np.ndarray:
        """Predice sobre datos nuevos crudos con las reglas aprendidas en train.

        Parameters
        ----------
        df : DataFrame (pandas o polars) o ndarray
            Datos con las columnas originales; el target, si aparece, se ignora.
            Las categorías no vistas en entrenamiento no generan error.
        """
        self._check_fitted()
        return self.pipeline_.predict(self._new_frame(df))

    def predict_proba(self, df: Any) -> np.ndarray:
        """Probabilidades por clase (columnas en el orden de ``classes_``)."""
        self._check_fitted()
        if not hasattr(self.pipeline_, "predict_proba"):
            raise AttributeError(f"{self.best_name_} no expone predict_proba.")
        return self.pipeline_.predict_proba(self._new_frame(df))

    # ------------------------------------------------------------------
    # Reportes, explicabilidad y persistencia
    # ------------------------------------------------------------------

    def report(self, plot: bool = True, save: Optional[str] = None) -> dict:
        """Imprime el reporte completo y devuelve las métricas de prueba.

        Parameters
        ----------
        plot : bool, default=True
            Muestra gráficos (requiere matplotlib): matriz de confusión,
            curvas ROC/PR o residuos, e importancias.
        save : str, optional
            Ruta ``.html`` (reporte autocontenido) o ``.json`` (datos del
            reporte y registro del experimento).
        """
        from .reporting import render_report

        self._check_fitted()
        render_report(self, plot=plot, save=save)
        return self.results_

    def report_dict(self) -> dict:
        """Contenido del reporte como diccionario serializable."""
        from .reporting import build_report

        self._check_fitted()
        return build_report(self)

    def explain_shap(self, df: Any, max_samples: int = 200) -> dict:
        """Valores SHAP del modelo final (requiere ``synaptix[explain]``)."""
        self._check_fitted()
        return shap_values(self.pipeline_, self._new_frame(df), max_samples=max_samples,
                           random_state=self.random_state)

    @property
    def causal_note(self) -> str:
        return CAUSAL_NOTE

    def save(self, path: str) -> None:
        """Guarda el pipeline completo (preprocesamiento + modelo + metadatos).

        No se guardan los datos de entrenamiento ni las features de prueba.
        """
        self._check_fitted()
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> "AutoPipeline":
        """Carga un pipeline guardado con :meth:`save`."""
        with open(path, "rb") as file:
            return pickle.load(file)

    def _check_fitted(self) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("El pipeline no está entrenado. Llama a fit(df, target).")

    def __repr__(self) -> str:
        if self.pipeline_ is None:
            return "<AutoPipeline (sin entrenar)>"
        return f"<AutoPipeline ({self.task_}, mejor: {self.best_name_})>"


def _split_distributions(user: Optional[dict]):
    """Separa las distribuciones del usuario en espacio (tuplas/listas) o scipy."""
    if not user:
        return None, None
    if all(isinstance(v, list) or (isinstance(v, tuple) and v and v[0] in ("float", "int", "cat"))
           for v in user.values()):
        space = {k: ("cat", v) if isinstance(v, list) else v for k, v in user.items()}
        return space, None
    return None, dict(user)
