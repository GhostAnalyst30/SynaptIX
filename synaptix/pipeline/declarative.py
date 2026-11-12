"""
Pipeline declarativo por pasos.

El usuario lista los pasos del proceso y el sistema rellena los
parámetros con valores sensatos; cada paso acepta sus propios
parámetros para sobreescribir los defaults.

El conjunto de prueba se separa antes de cualquier paso. La limpieza se
ajusta solo con el entrenamiento (y dentro de cada fold en la selección y
el ajuste de hiperparámetros), y los outliers se eliminan solo del
entrenamiento.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline

from ..automl.evaluation.cross_validation import holdout_split, make_cv
from ..automl.evaluation.metrics import resolve_metrics
from ..automl.features import CleanerTransformer
from ..automl.selection.ranking import build_ranking, evaluate_candidate, pick_best
from ..preprocessing import DataCleaner, remove_outliers
from .auto import detect_task
from .grids import DEFAULT_CANDIDATES, DEFAULT_GRIDS, resolve_model

VALID_STEPS = ("clean", "outliers", "model", "tune", "evaluate")


def _normalize_steps(steps: list) -> list:
    """Convierte cada paso a la forma ``(nombre, argumento, params)``."""
    normalized = []
    for step in steps:
        if isinstance(step, str):
            name, arg, params = step, None, {}
        elif isinstance(step, tuple):
            name = step[0]
            arg, params = None, {}
            for item in step[1:]:
                if isinstance(item, str):
                    arg = item
                elif isinstance(item, dict):
                    params = dict(item)
                else:
                    raise ValueError(f"Elemento de paso no reconocido: {item!r}")
        else:
            raise ValueError(f"Paso no reconocido: {step!r}")

        if name not in VALID_STEPS:
            raise ValueError(f"Paso '{name}' no soportado. Opciones: {VALID_STEPS}")
        normalized.append((name, arg, params))
    return normalized


class Pipeline:
    """Pipeline declarativo: lista los pasos, el sistema pone los parámetros.

    Parameters
    ----------
    steps : list
        Pasos en orden. Cada paso puede ser un string (usa defaults) o
        una tupla con parámetros propios:

        - ``"clean"`` o ``("clean", {"impute_strategy": "mean", ...})``
        - ``"outliers"`` o ``("outliers", {"method": "iqr", "threshold": 1.5})``
          (se aplica solo a las filas de entrenamiento, sobre los valores
          originales)
        - ``"model"`` (elige el mejor automáticamente),
          ``("model", "random_forest")`` o
          ``("model", "random_forest", {"n_estimators": 300})``
        - ``"tune"`` (rejilla por defecto del modelo) o
          ``("tune", {"param_grid": {...}})``
        - ``"evaluate"`` o ``("evaluate", {"plot": True})``
    task : {"classification", "regression"}, optional
        Se detecta automáticamente si no se indica.
    cv : int, default=5
        Folds para la selección automática y el tuning.
    test_size : float, default=0.2
        Fracción reservada para evaluación.
    random_state : int, default=42
        Semilla de reproducibilidad.
    verbose : bool, default=True
        Imprime el progreso de cada paso.

    Attributes
    ----------
    model_ : SynaptixModel
        Modelo entrenado (recibe datos ya limpios).
    cleaner_ : DataCleaner o None
        Limpieza ajustada con el entrenamiento.
    pipeline_ : sklearn.pipeline.Pipeline
        Limpieza + modelo, entrenado (recibe datos crudos).
    ranking_ : DataFrame o None
        Comparación de modelos (solo si el paso model fue automático).
    results_ : dict o None
        Métricas del paso evaluate.

    Ejemplo
    -------
    >>> from synaptix.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     "clean",
    ...     ("outliers", {"method": "iqr"}),
    ...     ("model", "random_forest"),
    ...     "tune",
    ...     ("evaluate", {"plot": True}),
    ... ])
    >>> pipe.fit(df, target="precio")
    >>> pipe.predict(df_nuevo)
    """

    def __init__(
        self,
        steps: list,
        task: Optional[str] = None,
        cv: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.steps = _normalize_steps(steps)
        self.task = task
        self.cv = cv
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

        self.task_: Optional[str] = None
        self.target_: Optional[str] = None
        self.cleaner_: Optional[DataCleaner] = None
        self.model_ = None
        self.model_name_: Optional[str] = None
        self.pipeline_: Optional[SkPipeline] = None
        self.ranking_: Optional[pd.DataFrame] = None
        self.best_params_: Optional[dict] = None
        self.results_: Optional[dict] = None
        self._X_train = self._X_test = self._y_train = self._y_test = None
        self._clean_params: Optional[dict] = None
        self._feature_columns: list = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target: str) -> "Pipeline":
        """Ejecuta los pasos en orden sobre el DataFrame.

        Parameters
        ----------
        df : DataFrame
            Datos crudos.
        target : str
            Columna objetivo.
        """
        if target not in df.columns:
            raise ValueError(f"La columna '{target}' no existe en el DataFrame.")
        self.target_ = target
        self.task_ = self.task or detect_task(df[target].dropna())

        data = df.dropna(subset=[target])
        self._split(data)
        self._clean_params = None
        model_fitted = False

        for index, (name, arg, params) in enumerate(self.steps, start=1):
            prefix = f"[{index}/{len(self.steps)}] {name}"

            if name == "clean":
                if model_fitted:
                    raise ValueError("El paso 'clean' debe ir antes de 'model'.")
                DataCleaner(**params)  # valida los parámetros
                self._clean_params = dict(params)
                self._log(f"{prefix}: se ajustará solo con datos de entrenamiento")

            elif name == "outliers":
                if model_fitted:
                    raise ValueError("El paso 'outliers' debe ir antes de 'model'.")
                params = dict(params)
                columns = params.pop("columns", None)
                if columns is None:
                    columns = self._X_train.select_dtypes(include=[np.number]).columns.tolist()
                train = self._X_train.assign(**{target: self._y_train})
                before = len(train)
                train = remove_outliers(train, columns=columns, **params)
                self._X_train = train.drop(columns=target)
                self._y_train = train[target]
                self._log(f"{prefix}: {before - len(train)} filas de entrenamiento eliminadas")

            elif name == "model":
                if arg is None:  # selección automática
                    self._auto_model(**params)
                else:
                    model_class = resolve_model(self.task_, arg)
                    self.model_name_ = arg
                    self._fit_final(model_class(**params).model)
                    self._log(f"{prefix}: {arg} entrenado")
                model_fitted = True

            elif name == "tune":
                if not model_fitted:
                    raise ValueError("El paso 'tune' debe ir después de 'model'.")
                grid = params.get("param_grid") or DEFAULT_GRIDS.get(self.model_name_, {})
                if not grid:
                    self._log(f"{prefix}: sin rejilla para {self.model_name_}, se omite")
                    continue
                cv, _ = make_cv(params.get("cv", self.cv), "auto", self.task_,
                                y=np.asarray(self._y_train), random_state=self.random_state)
                search = GridSearchCV(
                    self._make_pipeline(self.pipeline_.steps[-1][1]),
                    {f"model__{k}": v for k, v in grid.items()},
                    cv=cv,
                    scoring="accuracy" if self.task_ == "classification" else "r2",
                )
                search.fit(self._X_train, self._y_train)
                self.best_params_ = {k.split("__", 1)[1]: v
                                     for k, v in search.best_params_.items()}
                self._fit_final(search.best_estimator_.steps[-1][1])
                self._log(f"{prefix}: mejores parámetros {self.best_params_}")

            elif name == "evaluate":
                if not model_fitted:
                    raise ValueError("El paso 'evaluate' debe ir después de 'model'.")
                X_test = self._transform(self._X_test)
                self.results_ = self.model_.evaluate(
                    X_test, self._y_test,
                    plot=params.get("plot", False),
                    verbose=False,
                )
                main = "accuracy" if self.task_ == "classification" else "R2"
                self._log(f"{prefix}: {main} en test = {self.results_[main]:.4f}")

        if not model_fitted:
            raise ValueError("El pipeline necesita un paso 'model'.")
        return self

    # ------------------------------------------------------------------

    def _split(self, data: pd.DataFrame) -> None:
        X = data.drop(columns=self.target_)
        y = data[self.target_]
        self._feature_columns = list(X.columns)
        train_idx, test_idx, _ = holdout_split(
            len(X), self.test_size, self.task_, y=y.to_numpy(),
            random_state=self.random_state,
        )
        self._X_train, self._X_test = X.iloc[train_idx], X.iloc[test_idx]
        self._y_train, self._y_test = y.iloc[train_idx], y.iloc[test_idx]

    def _make_pipeline(self, estimator) -> SkPipeline:
        from sklearn.base import clone

        steps = []
        if self._clean_params is not None:
            steps.append(("clean", CleanerTransformer(**self._clean_params)))
        steps.append(("model", clone(estimator)))
        return SkPipeline(steps)

    def _fit_final(self, estimator) -> None:
        self.pipeline_ = self._make_pipeline(estimator).fit(self._X_train, self._y_train)
        self.cleaner_ = (self.pipeline_.named_steps["clean"].cleaner_
                         if "clean" in self.pipeline_.named_steps else None)
        wrapper = resolve_model(self.task_, self.model_name_)()
        wrapper.model = self.pipeline_.steps[-1][1]
        wrapper.fitted = True
        self.model_ = wrapper

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.cleaner_.transform(X) if self.cleaner_ is not None else X

    def _auto_model(self, models: Optional[list] = None) -> None:
        """Compara los candidatos por CV (limpieza dentro de cada fold) y entrena el mejor."""
        candidates = models or DEFAULT_CANDIDATES[self.task_]
        y_train = np.asarray(self._y_train)
        specs = resolve_metrics(None, [], self.task_,
                                labels=np.unique(y_train) if self.task_ == "classification"
                                else None)
        cv, _ = make_cv(self.cv, "auto", self.task_, y=y_train,
                        random_state=self.random_state)
        results = []
        for name in candidates:
            estimator = resolve_model(self.task_, name)().model
            results.append(evaluate_candidate(name, self._make_pipeline(estimator),
                                              self._X_train, y_train, cv, specs))
        self.ranking_ = build_ranking(results, specs)
        best = pick_best(results, specs)
        self.model_name_ = best.name
        self._fit_final(resolve_model(self.task_, best.name)().model)
        score = float(np.mean(best.scores[specs[0].key]))
        self._log(f"        modelo automático: {self.model_name_} "
                  f"({specs[0].display}={score:.4f})")

    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice sobre datos nuevos crudos, reaplicando la limpieza."""
        if self.model_ is None:
            raise RuntimeError("El pipeline no está entrenado. Llama a fit(df, target).")
        data = df.drop(columns=[self.target_], errors="ignore")[self._feature_columns]
        return self.pipeline_.predict(data)

    def __repr__(self) -> str:
        names = " -> ".join(name for name, _, _ in self.steps)
        status = self.model_name_ or "sin entrenar"
        return f"<Pipeline [{names}] ({status})>"
