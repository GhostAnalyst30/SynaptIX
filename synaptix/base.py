"""
Clases base de SynaptIX.

Define ``SynaptixModel``, la clase base de todos los modelos supervisados
de la librería. Provee una API unificada estilo scikit-learn:

    fit / predict / evaluate / fit_evaluate / summary / save / load

Ejemplo
-------
>>> from synaptix.supervised import RandomForestClassifier
>>> model = RandomForestClassifier(n_estimators=100)
>>> model.fit(X_train, y_train)
>>> model.evaluate(X_test, y_test)
>>> # O dividir, entrenar y evaluar train/test en un paso:
>>> resultado = model.fit_evaluate(X, y, test_size=0.3, metrics=["accuracy", "f1"])
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray, list]
TestSize = Optional[Union[float, int]]


def _from_polars(obj: Any) -> Any:
    """Convierte DataFrames/Series de polars a pandas (si aplica).

    Permite usar directamente datos que vienen de statslibx con backend
    polars, sin que polars sea una dependencia de synaptix.
    """
    if type(obj).__module__.split(".")[0] == "polars" and hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return obj


def to_matrix(X: ArrayLike) -> np.ndarray:
    """Convierte la entrada a una matriz NumPy 2D.

    Parameters
    ----------
    X : DataFrame (pandas o polars), Series, ndarray o lista
        Datos de entrada.

    Returns
    -------
    ndarray
        Matriz de forma ``(n_muestras, n_features)``.
    """
    X = _from_polars(X)
    if isinstance(X, pd.DataFrame):
        return X.values
    if isinstance(X, pd.Series):
        return X.values.reshape(-1, 1)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def to_vector(y: ArrayLike) -> np.ndarray:
    """Convierte la entrada a un vector NumPy 1D.

    Parameters
    ----------
    y : DataFrame (pandas o polars), Series, ndarray o lista
        Variable objetivo.

    Returns
    -------
    ndarray
        Vector de forma ``(n_muestras,)``.
    """
    y = _from_polars(y)
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values
    y = np.asarray(y)
    return y.ravel()


def feature_names(X: ArrayLike) -> Optional[list]:
    """Devuelve los nombres de columnas si ``X`` es un DataFrame."""
    X = _from_polars(X)
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def check_test_size(test_size: TestSize, n_samples: int) -> int:
    """Valida ``test_size`` y devuelve el nº de muestras de prueba.

    Acepta un float en (0, 1) (proporción) o un entero >= 1 (cantidad).
    """
    if isinstance(test_size, bool):
        raise ValueError(f"test_size no puede ser booleano (recibido: {test_size!r}).")
    if isinstance(test_size, (float, np.floating)):
        if not 0 < test_size < 1:
            raise ValueError(f"test_size debe estar en (0, 1) (recibido: {test_size!r}).")
        n_test = int(np.ceil(test_size * n_samples))
    elif isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
    else:
        raise ValueError(
            f"test_size debe ser un float en (0, 1) o un entero (recibido: {test_size!r})."
        )
    if n_test < 1 or n_test >= n_samples:
        raise ValueError(
            f"test_size={test_size} deja un conjunto de train o test vacío "
            f"con {n_samples} muestras."
        )
    return n_test


def split_data(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    test_size: TestSize = 0.2,
    random_state: Optional[int] = 42,
    stratify: bool = False,
):
    """Divide X (y opcionalmente y) en train/test conservando DataFrames.

    Con ``stratify=True`` estratifica por ``y`` si cada clase tiene al
    menos 2 muestras y caben todas en ambos conjuntos; si no, avisa y
    hace un split aleatorio simple.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` o ``(X_train, X_test)`` si
        ``y`` es None.
    """
    from sklearn.model_selection import train_test_split

    X = _from_polars(X)
    n_samples = len(X)
    n_test = check_test_size(test_size, n_samples)

    if y is None:
        return train_test_split(X, test_size=n_test, random_state=random_state)

    y = _from_polars(y)
    strat = None
    if stratify:
        _, counts = np.unique(to_vector(y), return_counts=True)
        if counts.min() >= 2 and len(counts) <= min(n_test, n_samples - n_test):
            strat = to_vector(y)
        else:
            warnings.warn(
                "No se puede estratificar con este test_size (clases con muy pocas "
                "muestras); se usa un split aleatorio simple.",
                stacklevel=3,
            )
    return train_test_split(X, y, test_size=n_test, random_state=random_state, stratify=strat)


def _print_train_test(result) -> None:
    print(f"\n=== Train vs. test: {result.model_name} ===")
    if result.test_size is not None:
        print(
            f"  test_size : {result.test_size} "
            f"({result.n_train} train / {result.n_test} test)"
        )
    if result.fit_time is not None:
        print(f"  Tiempo    : {result.fit_time:.3f} s")
    print(result.to_frame().to_string(float_format=lambda v: f"{v:.4f}"))


class TrainTestMixin:
    """Agrega ``fit_evaluate`` a modelos supervisados con ``fit``/``evaluate``."""

    _default_fit_params: dict = {}

    def fit_evaluate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        test_size: TestSize = 0.2,
        metrics: Optional[Sequence[str]] = None,
        random_state: Optional[int] = 42,
        stratify: Optional[bool] = None,
        plot: bool = False,
        verbose: bool = True,
        fit_params: Optional[dict] = None,
    ):
        """Divide en train/test, entrena y devuelve las métricas de ambos conjuntos.

        Parameters
        ----------
        X : DataFrame, ndarray o lista
            Features completas.
        y : Series, ndarray o lista
            Variable objetivo completa.
        test_size : float, int o None, default=0.2
            Proporción (float en (0, 1)) o cantidad (int) de muestras de
            prueba. Con ``None`` entrena con todo y solo reporta train.
        metrics : sequence of str, optional
            Métricas a calcular (ver ``synaptix.metrics.available_metrics``).
            Por defecto las de :meth:`evaluate`.
        random_state : int, optional, default=42
            Semilla del split.
        stratify : bool, optional
            Estratificar por ``y``. Por defecto True en clasificación.
        plot : bool, default=False
            Gráficos de evaluación sobre el conjunto de prueba.
        verbose : bool, default=True
            Imprime la tabla train vs. test.
        fit_params : dict, optional
            Argumentos extra para ``fit`` (por ejemplo
            ``{"epochs": 50, "batch_size": 16}`` en redes neuronales).

        Returns
        -------
        TrainTestResult
            ``{"train": métricas, "test": métricas}``. ``.to_frame()``
            devuelve la tabla con la diferencia test - train (útil para
            detectar sobreajuste).
        """
        from .results import TrainTestResult

        task = getattr(self, "task", None)
        if stratify is None:
            stratify = task == "classification"

        if test_size is None:
            X_train, y_train, X_test, y_test = X, y, None, None
        else:
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )

        start = time.perf_counter()
        self.fit(X_train, y_train, **{**self._default_fit_params, **(fit_params or {})})
        fit_time = time.perf_counter() - start

        train_metrics = self.evaluate(X_train, y_train, verbose=False, metrics=metrics)
        test_metrics = None
        if X_test is not None:
            test_metrics = self.evaluate(X_test, y_test, verbose=False, metrics=metrics)

        shape = np.shape(_from_polars(X_train))
        if len(shape) > 2:
            info: dict = {"input_shape": tuple(shape[1:])}
        else:
            info = {"n_features": shape[1] if len(shape) == 2 else 1}
        estimator = getattr(self, "model", None)
        if hasattr(estimator, "get_params"):
            info["params"] = estimator.get_params()
        if task == "classification":
            info["clases"] = np.unique(to_vector(y)).tolist()

        result = TrainTestResult(
            train_metrics,
            test_metrics,
            model_name=getattr(self, "name", type(self).__name__),
            task=task,
            test_size=test_size,
            n_train=len(to_vector(y_train)),
            n_test=None if y_test is None else len(to_vector(y_test)),
            fit_time=round(fit_time, 4),
            info=info,
        )

        if verbose:
            _print_train_test(result)
        if plot and X_test is not None and hasattr(self, "_plot_evaluation"):
            self._plot_evaluation(to_vector(y_test), self.predict(X_test))
        return result


class SynaptixModel(TrainTestMixin):
    """Clase base para todos los modelos supervisados de SynaptIX.

    Envuelve un estimador de scikit-learn y expone una API unificada.
    Acepta DataFrames de pandas, arrays de NumPy o listas de Python.

    Attributes
    ----------
    model : objeto sklearn
        Estimador subyacente.
    task : str
        ``"regression"`` o ``"classification"``.
    fitted : bool
        Indica si el modelo ya fue entrenado.
    feature_names_ : list o None
        Nombres de las features (si se entrenó con un DataFrame).
    """

    task: str = "regression"

    def __init__(self, model: Any, name: Optional[str] = None):
        self.model = model
        self.name = name or type(self).__name__
        self.fitted = False
        self.feature_names_: Optional[list] = None

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SynaptixModel":
        """Entrena el modelo.

        Parameters
        ----------
        X : DataFrame, ndarray o lista
            Features de entrenamiento ``(n_muestras, n_features)``.
        y : Series, ndarray o lista
            Variable objetivo ``(n_muestras,)``.

        Returns
        -------
        SynaptixModel
            El propio modelo entrenado (permite encadenar llamadas).
        """
        self.feature_names_ = feature_names(X)
        self.model.fit(to_matrix(X), to_vector(y))
        self.fitted = True
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Genera predicciones para nuevas muestras.

        Parameters
        ----------
        X : DataFrame, ndarray o lista
            Features ``(n_muestras, n_features)``.

        Returns
        -------
        ndarray
            Predicciones ``(n_muestras,)``.
        """
        self._check_fitted()
        return self.model.predict(to_matrix(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Probabilidades por clase (solo clasificadores compatibles)."""
        self._check_fitted()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{self.name} no soporta predicción de probabilidades."
            )
        return self.model.predict_proba(to_matrix(X))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Score por defecto del estimador (R² o accuracy)."""
        self._check_fitted()
        return float(self.model.score(to_matrix(X), to_vector(y)))

    def evaluate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        plot: bool = False,
        verbose: bool = True,
        metrics: Optional[Sequence[str]] = None,
    ) -> dict:
        """Evalúa el modelo e imprime un reporte de métricas.

        Parameters
        ----------
        X : DataFrame, ndarray o lista
            Features de prueba.
        y : Series, ndarray o lista
            Valores reales.
        plot : bool, default=False
            Si es True, muestra gráficos relevantes (matriz de confusión
            en clasificación, real vs. predicho en regresión).
        verbose : bool, default=True
            Si es True, imprime las métricas en consola.
        metrics : sequence of str, optional
            Métricas a calcular (sin distinguir mayúsculas). Por defecto:
            accuracy/precision/recall/f1 en clasificación y
            MAE/MSE/RMSE/R2/MAPE en regresión. Ver
            ``synaptix.metrics.available_metrics(task)``.

        Returns
        -------
        EvaluationResult
            Diccionario con las métricas calculadas (compatible con el
            dict de siempre). Con ViewX instalado permite además
            ``.to_html()``, ``.to_report()`` y ``.to_presentation()``.
        """
        from . import metrics as sx_metrics

        self._check_fitted()
        y_true = to_vector(y)
        y_pred = self.predict(X)

        if self.task == "classification":
            proba = None
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(to_matrix(X))
            results = sx_metrics.classification_metrics(
                y_true,
                y_pred,
                metrics=metrics,
                y_proba=proba,
                labels=getattr(self.model, "classes_", None),
            )
            if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
                results.y_proba = proba[:, 1]
        else:
            results = sx_metrics.regression_metrics(y_true, y_pred, metrics=metrics)

        results.model_name = self.name

        if verbose:
            print(f"\n=== Evaluación: {self.name} ===")
            for key, value in results.items():
                print(f"  {key:<12}: {value:.4f}")

        if plot:
            self._plot_evaluation(y_true, y_pred)

        return results

    def _plot_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        from . import visualization as sx_viz

        if self.task == "classification":
            sx_viz.plot_confusion_matrix(y_true, y_pred)
        else:
            sx_viz.plot_predictions(y_true, y_pred)

    def diagnostics(
        self,
        X: ArrayLike,
        y: ArrayLike,
        alpha: float = 0.05,
        normality_method: str = "shapiro",
        verbose: bool = True,
    ) -> dict:
        """Diagnóstico estadístico del modelo usando StatsLibX (opcional).

        Requiere ``statslibx`` (``pip install synaptix[stats]``).

        En regresión ejecuta tests inferenciales sobre los residuos:
        normalidad y media cero. En clasificación ejecuta un test
        chi-cuadrado de asociación entre etiquetas reales y predichas.

        Parameters
        ----------
        X : DataFrame, ndarray o lista
            Features de prueba.
        y : Series, ndarray o lista
            Valores reales.
        alpha : float, default=0.05
            Nivel de significancia de los tests.
        normality_method : str, default="shapiro"
            Método del test de normalidad de statslibx
            ("shapiro", "ks", "anderson", "jarque_bera", "all").
        verbose : bool, default=True
            Imprime la interpretación de cada test.

        Returns
        -------
        dict
            Resultados de statslibx (``TestResult``) por nombre de test.
            En regresión: ``normalidad_residuos`` y ``media_residuos_cero``.
            En clasificación: ``asociacion_real_predicho``.

        Raises
        ------
        ImportError
            Si statslibx no está instalado.
        """
        try:
            from statslibx import InferentialStats
        except ImportError as exc:
            raise ImportError(
                "StatsLibX no está instalado. "
                "Instala con: pip install synaptix[stats] "
                "(requerido para diagnostics)."
            ) from exc

        self._check_fitted()
        y_true = to_vector(y)
        y_pred = self.predict(X)

        results: dict = {}

        if self.task == "classification":
            df = pd.DataFrame({"real": y_true, "predicho": y_pred})
            inferential = InferentialStats(df)
            results["asociacion_real_predicho"] = inferential.chi_square_test(
                "real", "predicho", alpha=alpha
            )
        else:
            residuals = y_true.astype(float) - y_pred.astype(float)
            df = pd.DataFrame({"residuos": residuals})
            inferential = InferentialStats(df)
            results["normalidad_residuos"] = inferential.normality_test(
                "residuos", method=normality_method, alpha=alpha
            )
            results["media_residuos_cero"] = inferential.t_test_1sample(
                "residuos", popmean=0.0, alpha=alpha
            )

        if verbose:
            print(f"\n=== Diagnóstico estadístico: {self.name} ===")
            for test_name, test_result in results.items():
                interpretation = getattr(test_result, "interpretation", test_result)
                print(f"  {test_name}: {interpretation}")

        return results

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Imprime un resumen del modelo: tipo, tarea, parámetros y features."""
        print(f"\n=== {self.name} ===")
        print(f"  Tarea      : {self.task}")
        print(f"  Entrenado  : {'sí' if self.fitted else 'no'}")
        if self.feature_names_:
            print(f"  Features   : {self.feature_names_}")
        print("  Parámetros :")
        for key, value in self.model.get_params().items():
            print(f"    {key} = {value}")

    def feature_importances(self) -> Optional[pd.Series]:
        """Importancia de features (modelos de árbol) o coeficientes (lineales).

        Returns
        -------
        Series o None
            Serie de pandas ordenada de mayor a menor importancia, o None
            si el modelo no expone importancias.
        """
        self._check_fitted()
        values = None
        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            values = np.ravel(self.model.coef_)

        if values is None:
            return None

        index = self.feature_names_ or [f"x{i}" for i in range(len(values))]
        return pd.Series(values, index=index).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Guarda el modelo entrenado en disco (formato pickle).

        Parameters
        ----------
        path : str
            Ruta del archivo, por ejemplo ``"modelo.pkl"``.
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> "SynaptixModel":
        """Carga un modelo previamente guardado con :meth:`save`.

        Parameters
        ----------
        path : str
            Ruta del archivo ``.pkl``.

        Returns
        -------
        SynaptixModel
            Instancia del modelo restaurado.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as file:
            return pickle.load(file)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                f"{self.name} no está entrenado. Llama a fit(X, y) primero."
            )

    def __repr__(self) -> str:
        status = "entrenado" if self.fitted else "sin entrenar"
        return f"<{self.name} ({self.task}, {status})>"
