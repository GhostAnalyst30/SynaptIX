"""
Clases base de SynaptIX.

Define ``SynaptixModel``, la clase base de todos los modelos supervisados
de la librería. Provee una API unificada estilo scikit-learn:

    fit / predict / evaluate / summary / save / load

Ejemplo
-------
>>> from synaptix.supervised import RandomForestClassifier
>>> model = RandomForestClassifier(n_estimators=100)
>>> model.fit(X_train, y_train)
>>> model.evaluate(X_test, y_test)
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray, list]


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


class SynaptixModel:
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
            results = sx_metrics.classification_metrics(y_true, y_pred)
        else:
            results = sx_metrics.regression_metrics(y_true, y_pred)

        results.model_name = self.name
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(to_matrix(X))
            if proba.ndim == 2 and proba.shape[1] == 2:
                results.y_proba = proba[:, 1]

        if verbose:
            print(f"\n=== Evaluación: {self.name} ===")
            for key, value in results.items():
                print(f"  {key:<12}: {value:.4f}")

        if plot:
            from . import visualization as sx_viz

            if self.task == "classification":
                sx_viz.plot_confusion_matrix(y_true, y_pred)
            else:
                sx_viz.plot_predictions(y_true, y_pred)

        return results

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
