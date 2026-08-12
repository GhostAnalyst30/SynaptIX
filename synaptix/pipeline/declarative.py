"""
Pipeline declarativo por pasos.

El usuario lista los pasos del proceso y el sistema rellena los
parámetros con valores sensatos; cada paso acepta sus propios
parámetros para sobreescribir los defaults.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..model_selection import GridSearch
from ..preprocessing import DataCleaner, remove_outliers, train_test_split
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
                    params = item
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
        Modelo entrenado.
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
        self.ranking_: Optional[pd.DataFrame] = None
        self.best_params_: Optional[dict] = None
        self.results_: Optional[dict] = None
        self._X_train = self._X_test = self._y_train = self._y_test = None

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

        data = df.dropna(subset=[target]).copy()
        model_fitted = False

        for index, (name, arg, params) in enumerate(self.steps, start=1):
            prefix = f"[{index}/{len(self.steps)}] {name}"

            if name == "clean":
                self.cleaner_ = DataCleaner(**params)
                data = self.cleaner_.clean(data, target=target)
                self._log(f"{prefix}: {data.shape[1] - 1} features listas")

            elif name == "outliers":
                columns = params.pop("columns", None)
                if columns is None:
                    columns = [
                        col for col in data.select_dtypes(include=[np.number]).columns
                        if col != target
                    ]
                before = len(data)
                data = remove_outliers(data, columns=columns, **params)
                self._log(f"{prefix}: {before - len(data)} filas con outliers eliminadas")

            elif name == "model":
                self._split(data)
                if arg is None:  # selección automática
                    self._auto_model(**params)
                else:
                    model_class = resolve_model(self.task_, arg)
                    self.model_name_ = arg
                    self.model_ = model_class(**params).fit(self._X_train, self._y_train)
                    self._log(f"{prefix}: {arg} entrenado")
                model_fitted = True

            elif name == "tune":
                if not model_fitted:
                    raise ValueError("El paso 'tune' debe ir después de 'model'.")
                grid = params.get("param_grid") or DEFAULT_GRIDS.get(self.model_name_, {})
                if not grid:
                    self._log(f"{prefix}: sin rejilla para {self.model_name_}, se omite")
                    continue
                cv = params.get("cv", self.cv)
                model_class = resolve_model(self.task_, self.model_name_)
                search = GridSearch(model_class(), grid, cv=cv)
                self.model_ = search.fit(self._X_train, self._y_train, verbose=False)
                self.best_params_ = search.best_params_
                self._log(f"{prefix}: mejores parámetros {self.best_params_}")

            elif name == "evaluate":
                if not model_fitted:
                    raise ValueError("El paso 'evaluate' debe ir después de 'model'.")
                self.results_ = self.model_.evaluate(
                    self._X_test, self._y_test,
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
        stratify = y if self.task_ == "classification" else None
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def _auto_model(self, models: Optional[list] = None) -> None:
        """Compara los candidatos por CV y entrena el mejor."""
        from sklearn.model_selection import cross_val_score

        candidates = models or DEFAULT_CANDIDATES[self.task_]
        scoring = "accuracy" if self.task_ == "classification" else "r2"
        rows = []
        for name in candidates:
            model = resolve_model(self.task_, name)()
            try:
                scores = cross_val_score(
                    model.model,
                    self._X_train.values, self._y_train.values,
                    cv=self.cv, scoring=scoring,
                )
                rows.append({"modelo": name, scoring: float(scores.mean())})
            except Exception:
                rows.append({"modelo": name, scoring: np.nan})

        self.ranking_ = (
            pd.DataFrame(rows).sort_values(scoring, ascending=False).reset_index(drop=True)
        )
        self.model_name_ = str(self.ranking_.iloc[0]["modelo"])
        self.model_ = resolve_model(self.task_, self.model_name_)().fit(
            self._X_train, self._y_train
        )
        self._log(
            f"        modelo automático: {self.model_name_} "
            f"({scoring}={self.ranking_.iloc[0][scoring]:.4f})"
        )

    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice sobre datos nuevos crudos, reaplicando la limpieza."""
        if self.model_ is None:
            raise RuntimeError("El pipeline no está entrenado. Llama a fit(df, target).")
        data = df.drop(columns=[self.target_], errors="ignore")
        if self.cleaner_ is not None:
            data = self.cleaner_.transform(data)
        return self.model_.predict(data)

    def __repr__(self) -> str:
        names = " -> ".join(name for name, _, _ in self.steps)
        status = self.model_name_ or "sin entrenar"
        return f"<Pipeline [{names}] ({status})>"
