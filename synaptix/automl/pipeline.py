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
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Union

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
from .exceptions import AutoMLError, MissingDependencyError
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
from .feature_engineering import FeatureEngineeringTransformer
from .data_quality import (
    ROW_LEVEL_CODES,
    analyze_and_solve_data_quality,
    SolverStrategy,
    DataQualityReport,
)
from .parallel import native_thread_limits, resolve_n_jobs
from .profiling import Profiler
from .selection.feature_subset_search import FeatureSubsetSearch
from .selection.ranking import (
    CandidateResult,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
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
    feature_search : {"none", "exhaustive", "random", "forward", "backward"}, default="none"
        Búsqueda de subconjuntos de variables independiente por algoritmo.
        - "none": desactiva la búsqueda (evalúa todas las features).
        - "exhaustive": evalúa todas las combinaciones posibles en [min_features, max_features].
        - "random": evalúa combinaciones aleatorias no duplicadas.
        - "forward": selección secuencial progresiva hacia adelante (SFS).
        - "backward": eliminación secuencial progresiva hacia atrás (SBS).
    feature_search_min_features : int, default=1
        Cantidad mínima de variables por subconjunto evaluado.
    feature_search_max_features : int o None, default=None
        Cantidad máxima de variables por subconjunto. Si es None, usa todas las disponibles.
    feature_search_max_combinations : int o None, default=None
        Límite máximo de combinaciones evaluadas por modelo.
    feature_search_strategy : {"auto", "exhaustive", "random", "forward", "backward"}, default="auto"
        Estrategia heurística cuando feature_search='auto' según el número de variables.
    feature_search_metric : str o MetricSpec, default="auto"
        Métrica utilizada para seleccionar la mejor combinación. "auto" usa la métrica principal.
    feature_search_n_jobs : int o None, default=None
        Paralelización de la búsqueda de variables. Si es None, usa n_jobs.
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
    associations : dict, optional
        Variables derivadas a crear: ``{"nombre_nueva": "expresion"}``.
        Las expresiones usan sintaxis Python segura (aritmética, funciones numpy).
        Se aplican en fit() y se replican automáticamente en predict().
    fe_handle_div_zero : {"error", "inf", "nan", "zero"}, default="nan"
        Qué hacer ante división por cero en variables derivadas.
    fe_handle_invalid : {"error", "nan"}, default="nan"
        Qué hacer ante resultados inválidos (inf/nan) en variables derivadas.
    warning_strategy : {"none", "safe", "aggressive"}, default="none"
        Estrategia de corrección de calidad de datos:
        - "none": solo analiza, no modifica datos
        - "safe": imputación, infinitos, columnas constantes, filas duplicadas
        - "aggressive": incluye safe + outliers, categorías raras, conversión tipos

        La detección y las correcciones que estiman parámetros (medianas,
        cuantiles, frecuencias) se calculan **solo con las filas de
        entrenamiento**, después del holdout: el test nunca influye en cómo se
        imputan los datos. Solo las acciones que eliminan filas (duplicados)
        se aplican antes de separar, porque son deterministas.
    warning_report : bool, default=False
        Si True, incluye reporte detallado de calidad en ``report()["data_quality"]``.
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
    X_train_ : pandas.DataFrame
        Features de entrenamiento con el índice original: los datos crudos
        (más ``associations`` y las correcciones de ``warning_strategy``),
        no la matriz transformada por one-hot, escalado o imputación.
    X_test_ : pandas.DataFrame
        Features de prueba, con el índice original. Vacío (mismas columnas)
        si ``test_size=None``.
    y_train_ : pandas.Series
        Target de entrenamiento alineado con ``X_train_``.
    y_test_ : pandas.Series
        Target de prueba alineado con ``X_test_``. Vacío si
        ``test_size=None``.
    data_train_ : pandas.DataFrame
        ``X_train_`` más la columna target, conservando el índice original.
    data_test_ : pandas.DataFrame
        ``X_test_`` más la columna target, conservando el índice original.
        Vacío (con las mismas columnas) si ``test_size=None``.
    best_data_train_ : pandas.DataFrame
        Solo las variables de ``best_features_`` más el target, para train.
    best_data_test_ : pandas.DataFrame
        Solo las variables de ``best_features_`` más el target, para test.
        Vacío (con las mismas columnas) si ``test_size=None``.
    data_ : tuple de (DataFrame, DataFrame)
        Atajo ``(data_train_, data_test_)``. Requiere haber llamado a ``fit``.
    best_data_ : tuple de (DataFrame, DataFrame)
        Atajo ``(best_data_train_, best_data_test_)``. Requiere ``fit``.
    data : tuple de (DataFrame, DataFrame)
        Alias de ``data_``.
    best_features_ : list de str o None
        Columnas que componen el subconjunto de variables óptimo para el modelo ganador.
    best_score_ : float o None
        Puntuación media de validación cruzada del modelo ganador en su subconjunto óptimo.
    feature_search_results_ : DataFrame o None
        Registro exhaustivo de combinaciones evaluadas por modelo con sus métricas.
    feature_search_ranking_ : DataFrame o None
        Ranking ordenado de las combinaciones evaluadas por rendimiento en CV.
    feature_search_summary_ : dict o None
        Resumen cuantitativo de la búsqueda de subconjuntos de variables.
    results_ : dict
        Métricas del conjunto de prueba.
    selected_features_ : list de str
        Columnas que recibe el modelo tras el preprocesamiento (one-hot
        incluido).
    removed_features_ : list de str
        Columnas preparadas que ningún selector conservó.
    feature_selection_info_ : dict
        Método de selección, número de columnas y justificación de los descartes.
    feature_importance_ : DataFrame o None
        Importancia preferida: permutación en el holdout si se calculó, si no
        la nativa del modelo.
    feature_importance_source_ : {"permutation_test", "model"} o None
    data_quality_report_ : DataQualityReport o None
        Diagnóstico y acciones de calidad (con ``warning_strategy``).
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

    Recuperar exactamente los datos usados en el entrenamiento (índices
    originales, target incluido y sin one-hot ni escalado):

    >>> train, test = pipe.data_          # columnas de feature_columns_ + target
    >>> train, test = pipe.data           # alias
    >>> best_train, best_test = pipe.best_data_   # solo best_features_ + target
    >>> list(best_train.columns) == list(pipe.best_features_) + [pipe.target_]
    True
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
        feature_search: Literal["none", "exhaustive", "random", "forward", "backward"] = "none",
        feature_search_min_features: int = 1,
        feature_search_max_features: Optional[int] = None,
        feature_search_max_combinations: Optional[int] = None,
        feature_search_strategy: str = "auto",
        feature_search_metric: str = "auto",
        feature_search_n_jobs: Optional[int] = None,
        ensemble: Optional[str] = None,
        ensemble_top_k: int = 3,
        explain: bool = True,
        ignore_columns: Optional[Sequence[str]] = None,
        n_jobs: Optional[int] = 1,
        pre_dispatch: str = "2*n_jobs",
        memory_optimization: bool = False,
        cache: bool = False,
        associations: Optional[Dict[str, str]] = None,
        fe_handle_div_zero: str = "nan",
        fe_handle_invalid: str = "nan",
        warning_strategy: Literal["none", "safe", "aggressive"] = "none",
        warning_report: bool = False,
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
        self.feature_search = feature_search
        self.feature_search_min_features = feature_search_min_features
        self.feature_search_max_features = feature_search_max_features
        self.feature_search_max_combinations = feature_search_max_combinations
        self.feature_search_strategy = feature_search_strategy
        self.feature_search_metric = feature_search_metric
        self.feature_search_n_jobs = feature_search_n_jobs
        self.ensemble = ensemble
        self.ensemble_top_k = ensemble_top_k
        # El parámetro público sigue llamándose ``explain`` (get_params), pero
        # no puede ocupar el atributo: ``pipe.explain()`` es el método de
        # diagnóstico de explicabilidad.
        self._explain_enabled = bool(explain)
        self.ignore_columns = list(ignore_columns or [])
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.memory_optimization = memory_optimization
        self.cache = cache
        self.associations = associations or {}
        self.fe_handle_div_zero = fe_handle_div_zero
        self.fe_handle_invalid = fe_handle_invalid
        self.warning_strategy = warning_strategy
        self.warning_report = warning_report
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
        "drop_constant", "feature_selection", "feature_selection_k",
        "feature_search", "feature_search_min_features", "feature_search_max_features",
        "feature_search_max_combinations", "feature_search_strategy",
        "feature_search_metric", "feature_search_n_jobs", "ensemble",
        "ensemble_top_k", "explain", "ignore_columns", "n_jobs", "pre_dispatch",
        "memory_optimization", "cache", "associations", "fe_handle_div_zero",
        "fe_handle_invalid", "warning_strategy", "warning_report", "random_state", "verbose",
    )

    def get_params(self) -> Dict[str, Any]:
        """Parámetros de configuración del pipeline."""
        params = {name: getattr(self, name) for name in self._PUBLIC_PARAMS
                  if name != "explain"}
        params["explain"] = self._explain_enabled
        return params

    @property
    def explain_enabled(self) -> bool:
        """Si ``fit`` calcula explicabilidad automáticamente (parámetro ``explain``)."""
        return self._explain_enabled

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
        if self.fe_handle_div_zero not in ("error", "inf", "nan", "zero"):
            raise AutoMLError(
                f"fe_handle_div_zero debe ser uno de ('error', 'inf', 'nan', 'zero'), "
                f"recibido: {self.fe_handle_div_zero!r}"
            )
        if self.fe_handle_invalid not in ("error", "nan"):
            raise AutoMLError(
                f"fe_handle_invalid debe ser 'error' o 'nan', "
                f"recibido: {self.fe_handle_invalid!r}"
            )
        if self.warning_strategy not in ("none", "safe", "aggressive"):
            raise AutoMLError(
                f"warning_strategy debe ser 'none', 'safe' o 'aggressive' "
                f"(recibido: {self.warning_strategy!r})"
            )
        if self.feature_search not in ("none", "exhaustive", "random", "forward", "backward", "auto", None, False, True):
            raise AutoMLError(
                f"feature_search debe ser uno de ('none', 'exhaustive', 'random', 'forward', 'backward') o None "
                f"(recibido: {self.feature_search!r})."
            )
        if self.feature_search_min_features < 1:
            raise AutoMLError("feature_search_min_features debe ser >= 1.")
        if (
            self.feature_search_max_features is not None
            and self.feature_search_max_features < self.feature_search_min_features
        ):
            raise AutoMLError(
                f"feature_search_max_features ({self.feature_search_max_features}) "
                f"no puede ser menor que feature_search_min_features ({self.feature_search_min_features})."
            )
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
        self.fe_transformer_: Optional[FeatureEngineeringTransformer] = None
        self.data_quality_report_: Optional[DataQualityReport] = None
        self.best_features_: Optional[List[str]] = None
        self.best_score_: Optional[float] = None
        self.feature_search_results_: Optional[pd.DataFrame] = None
        self.feature_search_ranking_: Optional[pd.DataFrame] = None
        self.feature_search_summary_: Optional[Dict[str, Any]] = None
        self.selected_features_: List[str] = []
        self.removed_features_: List[str] = []
        self.feature_selection_info_: Dict[str, Any] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.feature_importance_source_: Optional[str] = None
        self.data_train_ = None
        self.data_test_ = None
        self.best_data_train_ = None
        self.best_data_test_ = None
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None

    @property
    def data_(self):
        """``(data_train_, data_test_)``: datos completos usados en ``fit``.

        Incluyen todas las features preparadas (``feature_columns_``) y el
        target, con el índice original. No son la matriz one-hot, escalada
        ni imputada. Si ``test_size=None``, el segundo elemento es un
        DataFrame vacío con las mismas columnas.
        """
        self._check_fitted()
        return self.data_train_, self.data_test_

    @property
    def best_data_(self):
        """``(best_data_train_, best_data_test_)``: datos del modelo ganador.

        Contienen solo ``best_features_`` más el target, con el índice
        original. Si ``test_size=None``, el segundo elemento es un DataFrame
        vacío con las columnas del modelo.
        """
        self._check_fitted()
        return self.best_data_train_, self.best_data_test_

    @property
    def data(self):
        """Alias de :attr:`data_`."""
        return self.data_

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
        data: Any = None,
        target: Union[str, int, Sequence] = None,
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

        # 1b. Feature Engineering ---------------------------------------
        if self.associations:
            self.fe_transformer_ = FeatureEngineeringTransformer(
                associations=self.associations,
                handle_division_by_zero=self.fe_handle_div_zero,
                handle_invalid=self.fe_handle_invalid,
            )
            df = self.fe_transformer_.fit_transform(df)
        else:
            self.fe_transformer_ = None

        # 1c. Calidad de datos: solo operaciones a nivel de fila ---------
        # Quitar filas duplicadas (o sin target) es determinista: no estima
        # parámetros con los datos. Las correcciones que sí aprenden
        # parámetros se ajustan más abajo, después del holdout y solo con las
        # filas de entrenamiento.
        self.data_quality_report_ = None
        row_actions: List[Any] = []
        if self.warning_strategy != "none":
            df, row_report = analyze_and_solve_data_quality(
                df=df,
                target=target,
                task=self.task,
                strategy=self.warning_strategy,
                codes=ROW_LEVEL_CODES,
            )
            row_actions = list(row_report.actions_taken)
            for action in row_actions:
                self._log(f"  [DataQuality] {action.action.value} en "
                          f"'{action.column}': {action.justification}")

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
        # 2b. Calidad de datos: parámetros estimados solo con train ------
        # La detección y las correcciones se ajustan con las filas de
        # entrenamiento del holdout; luego se aplican a train y a test. Así la
        # mediana de imputación (o los cuantiles de outliers) no contiene
        # información del conjunto de prueba.
        if self.warning_strategy != "none":
            with profiler.stage("calidad_datos"):
                base = df.loc[mask].copy()
                reference = base.iloc[train_idx]
                original_shape = base.shape
                base, report = analyze_and_solve_data_quality(
                    df=base,
                    target=target,
                    task=self.task_,
                    strategy=self.warning_strategy,
                    reference=reference,
                    exclude_codes=ROW_LEVEL_CODES,
                )
                report.actions_taken = row_actions + list(report.actions_taken)
                report.original_shape = original_shape
                report.final_shape = base.shape
                self.data_quality_report_ = report
                for action in report.actions_taken:
                    self._log(f"  [DataQuality] {action.action.value} en "
                              f"'{action.column}': {action.justification}")
                for issue in report.issues:
                    if issue.severity.value in ("warning", "critical"):
                        self.warnings_.append(
                            f"[DataQuality:{issue.code}] "
                            f"{issue.column or 'global'}: {issue.message}"
                        )
                kept = [c for c in self.feature_columns_ if c in base.columns]
                if len(kept) != len(self.feature_columns_):
                    self.feature_columns_ = kept
                X = base[self.feature_columns_]
                self.feature_types_ = infer_feature_types(X)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        self.X_train_ = X_train
        self.X_test_ = X_test

        self.y_train_ = pd.Series(
            y_train, index=X_train.index, name=target
        )

        self.y_test_ = pd.Series(
            y_test, index=X_test.index, name=target
        )

        self.data_train_ = pd.concat(
            [self.X_train_, self.y_train_], axis=1
        )

        self.data_test_ = pd.concat(
            [self.X_test_, self.y_test_], axis=1
        )
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
            is_feature_search = self.feature_search not in (None, "none", False)
            by_model_features: Dict[str, List[str]] = {}

            if is_feature_search:
                with native_thread_limits(self.n_jobs), profiler.stage("seleccion_modelos"):
                    fs_strategy = (
                        self.feature_search
                        if self.feature_search not in ("auto", True)
                        else self.feature_search_strategy
                    )
                    if self.feature_search_metric != "auto":
                        fs_specs = resolve_metrics(
                            self.feature_search_metric, None, self.task_, labels=self.classes_
                        )
                    else:
                        fs_specs = specs

                    fs_engine = FeatureSubsetSearch(
                        strategy=fs_strategy,
                        min_features=self.feature_search_min_features,
                        max_features=self.feature_search_max_features,
                        max_combinations=self.feature_search_max_combinations,
                        metric=fs_specs[0],
                        n_jobs=self.feature_search_n_jobs or self.n_jobs,
                        random_state=self.random_state or 42,
                    )

                    start_fs_time = time.perf_counter()
                    candidate_results = []

                    for candidate in candidates:
                        if budget.exhausted and any(r.usable for r in candidate_results):
                            candidate_results.append(
                                CandidateResult(candidate.name, STATUS_SKIPPED,
                                                error="Presupuesto de tiempo agotado.")
                            )
                            continue

                        sparse = self.memory_optimization and candidate.supports_sparse

                        def build_sub(est, sub_cols):
                            return self._make_pipeline_for_features(est, sub_cols, sparse, memory)

                        cand_res, cand_features = fs_engine.search_for_model(
                            candidate_name=candidate.name,
                            estimator=candidate.estimator,
                            pipeline_builder=build_sub,
                            X=X_train,
                            y=y_train,
                            cv=cv_splitter,
                            specs=specs,
                            features=self.feature_columns_,
                            groups=groups_train,
                            budget=budget,
                            log=self._log if self.verbose else None,
                        )
                        candidate_results.append(cand_res)
                        by_model_features[candidate.name] = cand_features
                        if self.verbose:
                            if cand_res.usable:
                                mean_val = specs[0].to_display(np.mean(cand_res.scores[specs[0].key]))
                                self._log(
                                    f"      {candidate.name:<24} {specs[0].display}={float(mean_val):.4f} "
                                    f"({len(cand_features)} features: {cand_features})"
                                )
                            else:
                                self._log(f"      {candidate.name:<24} FALLÓ: {cand_res.error[:80]}")

                    total_fs_time = time.perf_counter() - start_fs_time
                    fs_engine.finalize(specs[0], self.feature_columns_, total_fs_time, budget=budget)

                    self.feature_search_results_ = fs_engine.results_df_
                    self.feature_search_ranking_ = fs_engine.ranking_df_
                    self.feature_search_summary_ = fs_engine.summary_
                    results = candidate_results
            else:
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
                                specs, groups_train, cfg, budget, profiler,
                                by_model_features=by_model_features if is_feature_search else None)

            if self.ensemble:
                chosen = self._try_ensemble(results, by_name, chosen, X_train, y_train,
                                            cv_splitter, specs, groups_train, memory,
                                            is_temporal, profiler)

            # 7. Entrenamiento final -----------------------------------
            self.best_features_ = chosen.get("features", list(self.feature_columns_))
            self.best_score_ = chosen["cv_mean"]
            self._prune_fe_transformer()

            with native_thread_limits(self.n_jobs), profiler.stage("entrenamiento_final"):
                final = clone(chosen["pipeline"])
                final.fit(X_train[self.best_features_], y_train)
                final.memory = None
        finally:
            if cache_dir is not None:
                shutil.rmtree(cache_dir, ignore_errors=True)

        self.pipeline_ = final
        self.best_data_train_ = pd.concat(
            [
                self.X_train_[self.best_features_],
                self.y_train_
            ],
            axis=1
        )

        self.best_data_test_ = pd.concat(
            [
                self.X_test_[self.best_features_],
                self.y_test_
            ],
            axis=1
        )
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
                y_pred = final.predict(X_test[self.best_features_])
                proba = final.predict_proba(X_test[self.best_features_]) if (
                    self.task_ == "classification" and hasattr(final, "predict_proba")) else None
            self.results_ = holdout_metrics(self.task_, y_test, y_pred, proba,
                                            labels=self.classes_)
            raw = specs[0].scorer(final, X_test[self.best_features_], y_test)
            self.test_score_ = float(specs[0].to_display(raw))
            self._eval = {"y_true": np.asarray(y_test), "y_pred": np.asarray(y_pred),
                          "proba": proba, "index": np.asarray(X_test.index)}
            self._log(f"[6/6] Prueba (evaluada una sola vez): {specs[0].display}="
                      f"{self.test_score_:.4f}")
        else:
            self.results_ = {}

        # 9. Explicabilidad --------------------------------------------
        if self.explain_enabled:
            with profiler.stage("explicabilidad"):
                self._explain(final, X_test[self.best_features_] if len(X_test) else X_test, y_test, specs)

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

    def _make_pipeline_for_features(
        self, estimator: Any, sub_features: Sequence[str], sparse: bool, memory: Any = None
    ) -> Any:
        sub_set = set(sub_features)
        sub_types = {
            kind: [c for c in cols if c in sub_set]
            for kind, cols in self.feature_types_.items()
        }
        sub_types = {k: v for k, v in sub_types.items() if v}
        preprocessor = build_preprocessor(
            sub_types,
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

    def _prune_fe_transformer(self) -> None:
        """Conserva en fe_transformer_ solo las variables derivadas necesarias para best_features_."""
        if self.fe_transformer_ is None or not self.associations:
            return
        expected = set(self.best_features_ if self.best_features_ is not None else self.feature_columns_)
        needed: Set[str] = set()
        to_check = [f for f in expected if f in self.associations]
        dep_map = {d.name: d.depends_on for d in self.fe_transformer_.derived_features_}

        while to_check:
            cur = to_check.pop()
            if cur not in needed:
                needed.add(cur)
                deps = dep_map.get(cur, set())
                for dep in deps:
                    if dep in self.associations and dep not in needed:
                        to_check.append(dep)

        if not needed:
            self.fe_transformer_ = None
        elif len(needed) < len(self.associations):
            pruned_associations = {k: v for k, v in self.associations.items() if k in needed}
            old_fe = self.fe_transformer_
            self.fe_transformer_ = FeatureEngineeringTransformer(
                associations=pruned_associations,
                handle_division_by_zero=self.fe_handle_div_zero,
                handle_invalid=self.fe_handle_invalid,
            )
            self.fe_transformer_.derived_features_ = [
                d for d in old_fe.derived_features_ if d.name in needed
            ]
            self.fe_transformer_.output_columns_ = list(pruned_associations.keys())

    def _tune(self, results, by_name, build, X, y, cv, specs, groups, cfg, budget,
              profiler, by_model_features: Optional[Dict[str, List[str]]
              ] = None) -> Dict[str, Any]:
        """Optimiza los mejores modelos y devuelve el elegido por CV."""
        key = specs[0].key
        usable = sorted((r for r in results if r.usable),
                        key=lambda r: float(np.mean(r.scores[key])), reverse=True)
        best = usable[0]

        def entry(result: CandidateResult) -> Dict[str, Any]:
            scores = result.scores[key]
            feats = (by_model_features or {}).get(result.name, list(self.feature_columns_))
            pipe = (self._make_pipeline_for_features(by_name[result.name].estimator, feats,
                                                    self.memory_optimization and by_name[result.name].supports_sparse)
                    if by_model_features else build(by_name[result.name]))
            return {"name": result.name, "pipeline": pipe,
                    "cv_raw": float(np.mean(scores)),
                    "cv_mean": float(specs[0].to_display(np.mean(scores))),
                    "cv_std": float(np.std(scores)), "source": "default",
                    "tuned_params": None, "estimator": by_name[result.name].estimator,
                    "features": feats}

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

                feats = (by_model_features or {}).get(result.name, list(self.feature_columns_))
                pipe_to_tune = (
                    self._make_pipeline_for_features(candidate.estimator, feats,
                                                     self.memory_optimization and candidate.supports_sparse)
                    if by_model_features else build(candidate)
                )
                X_tune = X[feats]
                try:
                    outcome = tune_pipeline(
                        pipe_to_tune, X_tune, y, cv, specs[0].scorer, local_strategy,
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
                        "features": feats,
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
        ens_features = chosen.get("features", list(self.feature_columns_))
        with native_thread_limits(self.n_jobs), profiler.stage("ensemble"):
            estimator = build_ensemble(self.ensemble, self.task_, members, self.random_state,
                                       shuffle=not is_temporal)
            pipeline = self._make_pipeline_for_features(estimator, ens_features, sparse=False, memory=memory)
            result = evaluate_candidate(name, pipeline, X[ens_features], y, cv, specs, groups=groups,
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
                "estimator": estimator, "features": ens_features}

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
        self._set_feature_selection_attributes(names)
        transformer = final[:-1]
        cols_for_cleaner = self.best_features_ if self.best_features_ is not None else self.feature_columns_
        self.cleaner_ = _CleanerAdapter(transformer, cols_for_cleaner,
                                        names or [f"x{i}" for i in range(
                                            getattr(model, "n_features_in_", 0))])

    def _set_feature_selection_attributes(self, selected: Optional[List[str]]) -> None:
        """Registra qué columnas entran al modelo final y cuáles se descartaron.

        ``selected_features_`` son las columnas que recibe el modelo **después**
        del preprocesamiento (una categórica aporta una columna por categoría
        one-hot): es lo que el modelo realmente usa. ``removed_features_`` son
        las columnas preparadas que ningún selector conservó, con su
        justificación en ``feature_selection_info_``.
        """
        prepared: List[str] = []
        try:
            raw = self.pipeline_.named_steps["prep"].get_feature_names_out()
            prepared = [str(n).split("__", 1)[-1] for n in raw]
        except Exception:  # pragma: no cover - preprocesador sin nombres
            prepared = []
        selected = list(selected or [])
        chosen = set(selected)
        self.selected_features_ = selected
        self.removed_features_ = [n for n in prepared if n not in chosen]
        reasons: List[str] = []
        if self.drop_constant:
            reasons.append("varianza cero en el fold de entrenamiento (drop_constant)")
        if self.feature_selection == "variance":
            reasons.append("varianza cero (feature_selection='variance')")
        elif self.feature_selection == "kbest":
            reasons.append("no superó el test F univariante del umbral k")
        elif self.feature_selection == "model":
            reasons.append("importancia por debajo de la mediana del ExtraTrees")
        if self.feature_search not in (None, "none", False):
            reasons.append(f"búsqueda de subconjuntos de variables (feature_search='{self.feature_search}')")
        self.feature_selection_info_ = {
            "method": self.feature_selection,
            "feature_search": self.feature_search,
            "best_features": list(self.best_features_ or []),
            "drop_constant": bool(self.drop_constant),
            "n_prepared": len(prepared),
            "n_selected": len(selected),
            "removed": list(self.removed_features_),
            "justification": "; ".join(reasons) or "sin selección de features",
        }
        # Importancia nativa y coeficientes no necesitan el conjunto de prueba:
        # se calculan siempre para que ``explain()`` funcione con explain=False.
        try:
            self.model_importances_ = model_importances(self.pipeline_)
        except Exception:  # pragma: no cover - modelos sin importancia
            self.model_importances_ = None
        try:
            self.coefficients_ = coefficients(self.pipeline_, self.classes_)
        except Exception:  # pragma: no cover - modelos sin coef_
            self.coefficients_ = None
        if self.model_importances_ is not None:
            self.feature_importance_ = self.model_importances_
            self.feature_importance_source_ = "model"

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
        if self.feature_importances_ is not None:
            self.feature_importance_ = self.feature_importances_
            self.feature_importance_source_ = "permutation_test"

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
            feature_search=to_jsonable(self.feature_search_summary_ or {}),
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
        expected_cols = self.best_features_ if self.best_features_ is not None else self.feature_columns_
        if kind == "pandas":
            frame = data
        elif kind == "polars":
            frame = polars_to_pandas(data)
        elif kind == "polars_lazy":
            frame = polars_to_pandas(data.collect())
        elif kind == "numpy":
            arr = np.asarray(data)
            if arr.ndim != 2 or (arr.shape[1] != len(expected_cols) and arr.shape[1] != len(self.feature_columns_)):
                raise AutoMLError(f"Se esperaban {len(expected_cols)} columnas.")
            if arr.shape[1] == len(expected_cols):
                frame = pd.DataFrame(arr, columns=expected_cols)
            else:
                frame = pd.DataFrame(arr, columns=self.feature_columns_)
        else:
            raise AutoMLError("predict acepta DataFrames (pandas/polars) o arrays NumPy.")

        # Aplicar feature engineering PRIMERO para crear columnas derivadas
        if self.fe_transformer_ is not None:
            frame = self.fe_transformer_.transform(frame)

        # Ahora validar que tenemos todas las columnas necesarias (originales + derivadas)
        missing = [c for c in expected_cols if c not in frame.columns]
        if missing:
            raise AutoMLError(f"Faltan columnas usadas en el entrenamiento: {missing}")
        return frame[expected_cols]

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

    def explain(self, df: Any = None, max_samples: int = 200) -> dict:
        """Diagnóstico de explicabilidad del modelo finalmente entrenado.

        Devuelve las importancias globales (permutación en el conjunto de
        prueba, nativas del modelo y coeficientes), el diagnóstico de errores,
        la calibración de probabilidades cuando aplica y, si se pasa ``df``,
        valores SHAP locales (requiere ``synaptix[explain]``).

        No reentrena ni ajusta nada: usa lo calculado en ``fit``. Las
        importancias miden contribución predictiva, no causalidad.

        Parameters
        ----------
        df : DataFrame, optional
            Filas a explicar localmente en el espacio de features ya
            transformadas por el modelo (preprocesamiento incluido).
        max_samples : int, default=200
            Muestra máxima para SHAP.
        """
        self._check_fitted()
        # Con ``explain=False`` los diagnósticos no se calcularon en ``fit``;
        # se derivan aquí de las predicciones del holdout ya almacenadas
        # (no se reajusta nada ni se vuelve a evaluar el modelo).
        diagnostics_block = self.diagnostics_
        if diagnostics_block is None and self._eval:
            try:
                diagnostics_block = diagnostics(
                    self.task_, self._eval["y_true"], self._eval["y_pred"],
                    labels=self.classes_, index=self._eval["index"])
            except Exception as exc:  # pragma: no cover - salvaguarda
                self._warn(f"No se pudo calcular el diagnóstico en explain(): {exc}")
                diagnostics_block = None
        result: Dict[str, Any] = {
            "task": self.task_,
            "model": self.best_name_,
            "target": self.target_,
            "global": {
                "permutation_test": self.feature_importances_,
                "model_native": self.model_importances_,
                "coefficients": self.coefficients_,
                "preferred": self.feature_importance_source_,
                "selected_features": list(self.selected_features_),
                "removed_features": list(self.removed_features_),
                "selection": dict(self.feature_selection_info_),
            },
            "diagnostics": diagnostics_block,
            "calibration": self._calibration(),
            "note": CAUSAL_NOTE,
            "local": None,
        }
        if df is not None:
            try:
                result["local"] = shap_values(
                    self.pipeline_, self._new_frame(df), max_samples=max_samples,
                    random_state=self.random_state)
            except MissingDependencyError as exc:
                result["local_note"] = str(exc)
        return result

    def _calibration(self) -> Optional[Dict[str, Any]]:
        """Calibración de probabilidades sobre el conjunto de prueba.

        Brier score multiclase (error cuadrático medio entre probabilidad y
        indicador de la clase real) y, en binario, la curva de fiabilidad en
        10 bins. El conjunto de prueba ya se usó para reportar métricas, así
        que esto describe el modelo, no lo reajusta.
        """
        if self.task_ != "classification" or not self._eval:
            return None
        proba = self._eval.get("proba")
        if proba is None:
            return None
        y_true = np.asarray(self._eval["y_true"])
        classes = np.asarray(self.classes_)
        onehot = (y_true[:, None] == classes[None, :]).astype(float)
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        block: Dict[str, Any] = {"brier_score": brier, "bins": None}
        if len(classes) == 2:
            from sklearn.calibration import calibration_curve

            positive = (y_true == classes[-1])
            observed, predicted = calibration_curve(
                positive, proba[:, classes.tolist().index(classes[-1])],
                n_bins=10, strategy="uniform")
            block["bins"] = pd.DataFrame({"prob_predicha": predicted,
                                          "frecuencia_observada": observed})
            block["clase_positiva"] = str(classes[-1])
        return block

    def explain_shap(self, df: Any, max_samples: int = 200) -> dict:
        """Valores SHAP del modelo final (requiere ``synaptix[explain]``)."""
        self._check_fitted()
        return shap_values(self.pipeline_, self._new_frame(df), max_samples=max_samples,
                           random_state=self.random_state)

    @property
    def causal_note(self) -> str:
        return CAUSAL_NOTE

    _DATA_ATTRIBUTES = (
        "data_train_",
        "data_test_",
        "best_data_train_",
        "best_data_test_",
        "X_train_",
        "X_test_",
        "y_train_",
        "y_test_",
    )

    def save(self, path: str) -> None:
        """Guarda el pipeline completo (preprocesamiento + modelo + metadatos).

        No se guardan los datos de entrenamiento ni de prueba
        (``data_``/``best_data_`` quedan disponibles solo en memoria).
        """
        self._check_fitted()
        state = self.__getstate_for_pickle__()
        with open(path, "wb") as file:
            pickle.dump(state, file)

    def __getstate_for_pickle__(self) -> dict:
        """Estado para serialización: excluye datos pesados y no necesarios."""
        state = self.__dict__.copy()
        keys_to_remove = [
            "data",
            "_eval",
            "profiler_",
            *self._DATA_ATTRIBUTES,
        ]
        for key in keys_to_remove:
            state.pop(key, None)
        return state

    def __getstate__(self) -> dict:
        return self.__getstate_for_pickle__()

    def __setstate__(self, state: dict) -> None:
        """Restaurar estado desde pickle."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_eval", {})
        self.__dict__.setdefault("profiler_", None)
        for name in self._DATA_ATTRIBUTES:
            self.__dict__.setdefault(name, None)

    @classmethod
    def load(cls, path: str) -> "AutoPipeline":
        """Carga un pipeline guardado con :meth:`save`."""
        with open(path, "rb") as file:
            state = pickle.load(file)
        # Crear instancia sin llamar a __init__
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj

    @property
    def best_model(self):
        return self.best_model_

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








