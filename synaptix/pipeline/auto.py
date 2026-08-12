"""
AutoPipeline: machine learning de principio a fin en una llamada.

Pensado para usuarios sin experiencia: el sistema se encarga de la
limpieza, la elección del modelo y los hiperparámetros. Todo es
configurable por parámetros para usuarios avanzados.
"""

from __future__ import annotations

import pickle
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ..model_selection import GridSearch
from ..preprocessing import DataCleaner, train_test_split
from .grids import DEFAULT_CANDIDATES, DEFAULT_GRIDS, resolve_model


def detect_task(y: pd.Series) -> str:
    """Detecta si el target es de clasificación o regresión.

    Reglas: texto, categoría o booleano -> clasificación; numérico con
    pocos valores únicos (<= 20 y <= 5% de las filas) -> clasificación;
    en otro caso -> regresión.
    """
    if y.dtype == object or str(y.dtype) in ("category", "bool", "boolean"):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 20 and n_unique <= max(2, int(0.05 * len(y))):
        return "classification"
    return "regression"


class AutoPipeline:
    """Pipeline automático: ``fit(df, target)`` y el sistema hace el resto.

    Flujo: detectar tarea -> limpiar (imputar, codificar, escalar) ->
    dividir train/test -> comparar modelos con validación cruzada ->
    ajustar hiperparámetros del mejor -> evaluar en el set de prueba.

    Parameters
    ----------
    task : {"classification", "regression"}, optional
        Se detecta automáticamente si no se indica.
    models : list de str, optional
        Subconjunto de modelos a comparar (por ejemplo
        ``["random_forest", "svm"]``). Por defecto, los candidatos
        estándar de la tarea.
    cv : int, default=5
        Folds de validación cruzada.
    tune : bool, default=True
        Ajustar hiperparámetros del mejor modelo con GridSearch.
    test_size : float, default=0.2
        Fracción reservada para la evaluación final.
    impute_strategy : str, default="median"
        Estrategia de imputación de nulos.
    encode_method : str, default="onehot"
        Codificación de categóricas.
    scale_method : str o None, default="standard"
        Escalado de numéricas.
    param_grids : dict, optional
        Rejillas propias por nombre de modelo; sobreescriben las de
        :mod:`synaptix.pipeline.grids`. Ej.:
        ``{"random_forest": {"n_estimators": [300]}}``.
    random_state : int, default=42
        Semilla de reproducibilidad.
    verbose : bool, default=True
        Imprime el progreso de cada etapa.

    Attributes
    ----------
    task_ : str
        Tarea detectada o indicada.
    cleaner_ : DataCleaner
        Limpiador ajustado (reutilizable con ``transform``).
    ranking_ : DataFrame
        Comparación de modelos con validación cruzada.
    best_model_ : SynaptixModel
        Modelo ganador, ajustado y entrenado.
    best_params_ : dict o None
        Hiperparámetros elegidos por GridSearch.
    results_ : dict
        Métricas del set de prueba.

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
        task: Optional[Literal["classification", "regression"]] = None,
        models: Optional[list] = None,
        cv: int = 5,
        tune: bool = True,
        test_size: float = 0.2,
        impute_strategy: str = "median",
        encode_method: str = "onehot",
        scale_method: Optional[str] = "standard",
        param_grids: Optional[dict] = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.task = task
        self.models = models
        self.cv = cv
        self.tune = tune
        self.test_size = test_size
        self.impute_strategy = impute_strategy
        self.encode_method = encode_method
        self.scale_method = scale_method
        self.param_grids = param_grids or {}
        self.random_state = random_state
        self.verbose = verbose

        self.task_: Optional[str] = None
        self.target_: Optional[str] = None
        self.cleaner_: Optional[DataCleaner] = None
        self.ranking_: Optional[pd.DataFrame] = None
        self.best_model_ = None
        self.best_name_: Optional[str] = None
        self.best_params_: Optional[dict] = None
        self.results_: Optional[dict] = None
        self._X_test = None
        self._y_test = None

    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def fit(self, df: pd.DataFrame, target: str) -> "AutoPipeline":
        """Ejecuta el pipeline completo sobre un DataFrame.

        Parameters
        ----------
        df : DataFrame
            Datos crudos (pueden tener nulos y categóricas).
        target : str
            Nombre de la columna objetivo.

        Returns
        -------
        AutoPipeline
            El propio pipeline, listo para ``predict`` y ``report``.
        """
        if target not in df.columns:
            raise ValueError(f"La columna '{target}' no existe en el DataFrame.")
        self.target_ = target

        # 1. Detectar la tarea
        self.task_ = self.task or detect_task(df[target].dropna())
        self._log(f"[1/5] Tarea detectada: {self.task_}")

        # 2. Limpiar
        self.cleaner_ = DataCleaner(
            impute_strategy=self.impute_strategy,
            encode_method=self.encode_method,
            scale_method=self.scale_method,
        )
        df_clean = self.cleaner_.clean(df.dropna(subset=[target]), target=target)
        X = df_clean.drop(columns=target)
        y = df_clean[target]
        self._log(
            f"[2/5] Datos limpios: {X.shape[0]} filas, {X.shape[1]} features "
            f"(imputación, codificación y escalado aplicados)"
        )

        # 3. Dividir
        stratify = y if self.task_ == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        self._X_test, self._y_test = X_test, y_test

        # 4. Comparar modelos
        candidates = self.models or DEFAULT_CANDIDATES[self.task_]
        scoring = "accuracy" if self.task_ == "classification" else "r2"
        rows = []
        for name in candidates:
            model = resolve_model(self.task_, name)()
            start = time.perf_counter()
            try:
                scores = cross_val_score(
                    model.model, X_train.values, y_train.values,
                    cv=self.cv, scoring=scoring,
                )
                rows.append({
                    "modelo": name,
                    scoring: float(scores.mean()),
                    "desviacion": float(scores.std()),
                    "tiempo_s": round(time.perf_counter() - start, 3),
                })
            except Exception as error:
                rows.append({
                    "modelo": name, scoring: np.nan,
                    "desviacion": np.nan, "tiempo_s": np.nan,
                    "error": str(error),
                })

        self.ranking_ = (
            pd.DataFrame(rows).sort_values(scoring, ascending=False).reset_index(drop=True)
        )
        self.best_name_ = str(self.ranking_.iloc[0]["modelo"])
        best_score = self.ranking_.iloc[0][scoring]
        self._log(
            f"[3/5] {len(candidates)} modelos comparados ({self.cv} folds). "
            f"Mejor: {self.best_name_} ({scoring}={best_score:.4f})"
        )

        # 5. Ajustar hiperparámetros y entrenar el ganador
        model_class = resolve_model(self.task_, self.best_name_)
        grid = self.param_grids.get(self.best_name_, DEFAULT_GRIDS.get(self.best_name_, {}))

        if self.tune and grid:
            search = GridSearch(model_class(), grid, cv=self.cv)
            self.best_model_ = search.fit(X_train, y_train, verbose=False)
            self.best_params_ = search.best_params_
            self._log(f"[4/5] Hiperparámetros ajustados: {self.best_params_}")
        else:
            self.best_model_ = model_class().fit(X_train, y_train)
            self.best_params_ = None
            self._log("[4/5] Entrenado con parámetros por defecto")

        # 6. Evaluación final
        self.results_ = self.best_model_.evaluate(X_test, y_test, verbose=False)
        main_metric = "accuracy" if self.task_ == "classification" else "R2"
        self._log(
            f"[5/5] Evaluación final ({main_metric} en test): "
            f"{self.results_[main_metric]:.4f}"
        )
        return self

    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predice sobre datos nuevos crudos (aplica la misma limpieza).

        Parameters
        ----------
        df : DataFrame
            Datos nuevos con las columnas originales (el target, si
            aparece, se ignora).
        """
        self._check_fitted()
        data = df.drop(columns=[self.target_], errors="ignore")
        X = self.cleaner_.transform(data)
        return self.best_model_.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidades por clase para datos nuevos crudos."""
        self._check_fitted()
        data = df.drop(columns=[self.target_], errors="ignore")
        X = self.cleaner_.transform(data)
        return self.best_model_.predict_proba(X)

    def report(self, plot: bool = True) -> dict:
        """Imprime el resumen del pipeline y los gráficos de evaluación.

        Parameters
        ----------
        plot : bool, default=True
            Mostrar gráficos (matriz de confusión o real vs. predicho).
        """
        self._check_fitted()
        print("\n================ AutoPipeline ================")
        print(f"Tarea        : {self.task_}")
        print(f"Target       : {self.target_}")
        print(f"Mejor modelo : {self.best_name_}")
        if self.best_params_:
            print(f"Parámetros   : {self.best_params_}")
        print("\nRanking de modelos (validación cruzada):")
        print(self.ranking_.to_string(index=False))
        print("\nMétricas en el set de prueba:")
        for key, value in self.results_.items():
            print(f"  {key:<12}: {value:.4f}")

        if plot:
            self.best_model_.evaluate(
                self._X_test, self._y_test, plot=True, verbose=False
            )
        return self.results_

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Guarda el pipeline completo (limpieza + modelo) en disco."""
        self._check_fitted()
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> "AutoPipeline":
        """Carga un pipeline guardado con :meth:`save`."""
        with open(path, "rb") as file:
            return pickle.load(file)

    def _check_fitted(self) -> None:
        if self.best_model_ is None:
            raise RuntimeError("El pipeline no está entrenado. Llama a fit(df, target).")

    def __repr__(self) -> str:
        if self.best_model_ is None:
            return "<AutoPipeline (sin entrenar)>"
        return f"<AutoPipeline ({self.task_}, mejor: {self.best_name_})>"
