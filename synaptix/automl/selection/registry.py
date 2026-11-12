"""
Registro de modelos candidatos por tarea.

Cada modelo tiene un nombre canónico, alias y etiquetas de compatibilidad
(si acepta matrices dispersas, a partir de cuántas filas deja de ser
razonable por costo computacional). Los estimadores se construyen a partir
de los wrappers de :mod:`synaptix.supervised`, de modo que comparten los
mismos defaults que el resto de la librería.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.base import clone

from ... import supervised
from ..exceptions import AutoMLError


@dataclass(frozen=True)
class ModelSpec:
    """Descripción de un modelo candidato.

    Attributes
    ----------
    name : str
        Nombre canónico (``"random_forest"``).
    task : str
        ``"classification"`` o ``"regression"``.
    wrapper : type
        Clase de :mod:`synaptix.supervised`.
    aliases : tuple de str
        Nombres alternativos aceptados en ``models=``.
    supports_sparse : bool
        Si acepta la salida dispersa del one-hot.
    max_samples : int, optional
        Por encima de este número de filas se excluye de los candidatos
        por defecto (se puede pedir explícitamente).
    parallel : bool
        Si el estimador paraleliza internamente con ``n_jobs``.
    """

    name: str
    task: str
    wrapper: type
    aliases: Tuple[str, ...] = ()
    supports_sparse: bool = True
    max_samples: Optional[int] = None
    parallel: bool = False

    def build(self, random_state: Optional[int] = None, **params) -> Any:
        """Estimador de scikit-learn nuevo, con semilla y ``n_jobs=1``.

        El paralelismo se aplica en un solo nivel (validación cruzada o
        búsqueda); por eso los estimadores internos usan un solo proceso.
        """
        estimator = clone(self.wrapper().model)
        available = estimator.get_params()
        updates = dict(params)
        if "random_state" in available and "random_state" not in params:
            updates["random_state"] = random_state
        if self.parallel and "n_jobs" not in params:
            updates["n_jobs"] = 1
        unknown = set(updates) - set(available)
        if unknown:
            raise AutoMLError(f"Parámetros no válidos para '{self.name}': {sorted(unknown)}")
        return estimator.set_params(**updates)


def _specs() -> List[ModelSpec]:
    C, R = "classification", "regression"
    s = supervised
    return [
        ModelSpec("logistic", C, s.LogisticRegression, ("logistic_regression", "logreg")),
        ModelSpec("decision_tree", C, s.DecisionTreeClassifier),
        ModelSpec("random_forest", C, s.RandomForestClassifier, ("rf",), parallel=True),
        ModelSpec("extra_trees", C, s.ExtraTreesClassifier, parallel=True),
        ModelSpec("gradient_boosting", C, s.GradientBoostingClassifier, ("gbm",)),
        ModelSpec("hist_gradient_boosting", C, s.HistGradientBoostingClassifier,
                  ("hist_gb", "hgb"), supports_sparse=False),
        ModelSpec("svm", C, s.SVMClassifier, ("svc", "support_vector_machine"),
                  max_samples=20_000),
        ModelSpec("knn", C, s.KNNClassifier, ("k_neighbors",), max_samples=100_000,
                  parallel=True),
        ModelSpec("naive_bayes", C, s.NaiveBayes, ("gaussian_nb",), supports_sparse=False),
        ModelSpec("gaussian_process", C, s.GaussianProcessClassifier,
                  supports_sparse=False, max_samples=2_000),
        ModelSpec("linear", R, s.LinearRegression, ("linear_regression", "ols")),
        ModelSpec("ridge", R, s.RidgeRegression),
        ModelSpec("lasso", R, s.LassoRegression),
        ModelSpec("elastic_net", R, s.ElasticNetRegression, ("elasticnet",)),
        ModelSpec("bayesian_ridge", R, s.BayesianRidgeRegression),
        ModelSpec("decision_tree", R, s.DecisionTreeRegressor),
        ModelSpec("random_forest", R, s.RandomForestRegressor, ("rf",), parallel=True),
        ModelSpec("extra_trees", R, s.ExtraTreesRegressor, parallel=True),
        ModelSpec("gradient_boosting", R, s.GradientBoostingRegressor, ("gbm",)),
        ModelSpec("hist_gradient_boosting", R, s.HistGradientBoostingRegressor,
                  ("hist_gb", "hgb"), supports_sparse=False),
        ModelSpec("svm", R, s.SVR, ("svr", "support_vector_machine"), max_samples=20_000),
        ModelSpec("knn", R, s.KNNRegressor, ("k_neighbors",), max_samples=100_000,
                  parallel=True),
        ModelSpec("gaussian_process", R, s.GaussianProcessRegressor,
                  supports_sparse=False, max_samples=2_000),
    ]


SPECS: Dict[str, Dict[str, ModelSpec]] = {"classification": {}, "regression": {}}
for _spec in _specs():
    SPECS[_spec.task][_spec.name] = _spec

# Compatibilidad con synaptix.pipeline.grids: nombre -> clase SynaptIX.
MODEL_REGISTRY: Dict[str, Dict[str, type]] = {
    task: {name: spec.wrapper for name, spec in specs.items()}
    for task, specs in SPECS.items()
}

MODE_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "fast": {
        "classification": ["logistic", "decision_tree", "random_forest",
                           "hist_gradient_boosting", "naive_bayes"],
        "regression": ["linear", "ridge", "decision_tree", "random_forest",
                       "hist_gradient_boosting"],
    },
    "balanced": {
        "classification": ["logistic", "decision_tree", "random_forest", "extra_trees",
                           "gradient_boosting", "hist_gradient_boosting", "svm", "knn",
                           "naive_bayes"],
        "regression": ["linear", "ridge", "lasso", "elastic_net", "bayesian_ridge",
                       "decision_tree", "random_forest", "extra_trees", "gradient_boosting",
                       "hist_gradient_boosting", "knn"],
    },
}
MODE_CANDIDATES["exhaustive"] = {task: list(SPECS[task]) for task in SPECS}


def get_spec(task: str, name: str) -> ModelSpec:
    """Busca un modelo por nombre canónico o alias (sin distinguir mayúsculas)."""
    if task not in SPECS:
        raise AutoMLError(f"Tarea desconocida: {task!r}")
    key = name.lower()
    for spec in SPECS[task].values():
        if key == spec.name or key in spec.aliases:
            return spec
    raise AutoMLError(
        f"Modelo '{name}' no disponible para {task}. Opciones: {sorted(SPECS[task])}"
    )


def list_models(task: Optional[str] = None) -> Any:
    """Nombres de modelos disponibles (por tarea o todos)."""
    if task is None:
        return {t: sorted(specs) for t, specs in SPECS.items()}
    if task not in SPECS:
        raise AutoMLError(f"Tarea desconocida: {task!r}")
    return sorted(SPECS[task])


@dataclass
class Candidate:
    """Modelo a evaluar: nombre, estimador sin entrenar y su especificación."""

    name: str
    estimator: Any
    spec: Optional[ModelSpec] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def supports_sparse(self) -> bool:
        return True if self.spec is None else self.spec.supports_sparse


def select_candidates(
    task: str,
    mode: str = "balanced",
    models: Optional[Sequence[Any]] = None,
    exclude: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[Candidate], List[str]]:
    """Resuelve la lista de candidatos.

    Parameters
    ----------
    models : sequence, optional
        Cada elemento puede ser un nombre (``"svm"``), una tupla
        ``(nombre, {parámetros})``, un estimador de scikit-learn o un
        modelo de SynaptIX. Por defecto, los del ``mode``.
    exclude : sequence of str, optional
        Nombres a excluir.
    n_samples : int, optional
        Filas de entrenamiento; los modelos por defecto con
        ``max_samples`` menor se omiten (con nota).

    Returns
    -------
    (candidatos, notas)
    """
    if mode not in MODE_CANDIDATES:
        raise AutoMLError(f"mode debe ser uno de {list(MODE_CANDIDATES)} (recibido: {mode!r}).")
    notes: List[str] = []
    excluded = {get_spec(task, name).name for name in (exclude or [])}
    candidates: List[Candidate] = []

    if models is None:
        for name in MODE_CANDIDATES[mode][task]:
            spec = SPECS[task][name]
            if spec.name in excluded:
                continue
            if n_samples is not None and spec.max_samples and n_samples > spec.max_samples:
                notes.append(f"'{name}' omitido por costo: {n_samples} filas > "
                             f"{spec.max_samples} (pídelo en models= para incluirlo).")
                continue
            candidates.append(Candidate(name, spec.build(random_state), spec))
    else:
        if isinstance(models, (str, tuple)) or not isinstance(models, (list, set)):
            models = [models]
        for item in models:
            candidate = _candidate_from(item, task, random_state)
            if candidate.spec is not None and candidate.spec.name in excluded:
                continue
            spec = candidate.spec
            if (spec is not None and n_samples is not None and spec.max_samples
                    and n_samples > spec.max_samples):
                warnings.warn(f"'{spec.name}' con {n_samples} filas puede ser muy lento.",
                              UserWarning, stacklevel=3)
            candidates.append(candidate)

    names = [c.name for c in candidates]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        counter: Dict[str, int] = {}
        for c in candidates:
            if c.name in duplicated:
                counter[c.name] = counter.get(c.name, 0) + 1
                c.name = f"{c.name}_{counter[c.name]}"
    if not candidates:
        raise AutoMLError("No quedan modelos candidatos tras aplicar models/exclude_models.")
    return candidates, notes


def _candidate_from(item: Any, task: str, random_state: Optional[int]) -> Candidate:
    if isinstance(item, str):
        spec = get_spec(task, item)
        return Candidate(spec.name, spec.build(random_state), spec)
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        spec = get_spec(task, item[0])
        params = dict(item[1])
        return Candidate(spec.name, spec.build(random_state, **params), spec, params)
    estimator = getattr(item, "model", item)  # SynaptixModel -> estimador sklearn
    if hasattr(estimator, "fit") and hasattr(estimator, "get_params"):
        name = getattr(item, "name", None) or type(estimator).__name__
        return Candidate(str(name), clone(estimator))
    raise AutoMLError(f"Modelo no reconocido: {item!r}")
