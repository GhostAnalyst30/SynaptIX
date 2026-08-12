"""
Transformadores de datos: escalado, codificación e imputación.

Wrappers de alto nivel sobre scikit-learn que aceptan y devuelven
DataFrames de pandas, preservando los nombres de columnas.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder as _SkLabelEncoder,
    MinMaxScaler,
    OneHotEncoder as _SkOneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from ..base import ArrayLike


class Scaler:
    """Escalador de features numéricas.

    Parameters
    ----------
    method : {"standard", "minmax", "robust"}, default="standard"
        - ``standard``: media 0 y desviación estándar 1.
        - ``minmax``: rango [0, 1].
        - ``robust``: usa mediana e IQR (resistente a outliers).

    Ejemplo
    -------
    >>> from synaptix.preprocessing import Scaler
    >>> scaler = Scaler("standard")
    >>> X_scaled = scaler.fit_transform(X)
    >>> X_original = scaler.inverse_transform(X_scaled)
    """

    _METHODS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(self, method: Literal["standard", "minmax", "robust"] = "standard"):
        if method not in self._METHODS:
            raise ValueError(
                f"Método '{method}' no soportado. Usa: {list(self._METHODS)}"
            )
        self.method = method
        self.scaler = self._METHODS[method]()
        self._columns: Optional[list] = None

    def fit(self, X: ArrayLike) -> "Scaler":
        """Aprende los parámetros de escalado a partir de ``X``."""
        if isinstance(X, pd.DataFrame):
            self._columns = list(X.columns)
        self.scaler.fit(np.asarray(X))
        return self

    def transform(self, X: ArrayLike) -> Union[pd.DataFrame, np.ndarray]:
        """Aplica el escalado. Devuelve DataFrame si la entrada lo era."""
        values = self.scaler.transform(np.asarray(X))
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, columns=X.columns, index=X.index)
        return values

    def fit_transform(self, X: ArrayLike) -> Union[pd.DataFrame, np.ndarray]:
        """Equivale a ``fit(X)`` seguido de ``transform(X)``."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> Union[pd.DataFrame, np.ndarray]:
        """Revierte el escalado a la escala original."""
        values = self.scaler.inverse_transform(np.asarray(X))
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(values, columns=X.columns, index=X.index)
        return values


class Encoder:
    """Codificador de variables categóricas.

    Parameters
    ----------
    method : {"onehot", "label"}, default="onehot"
        - ``onehot``: crea una columna binaria por categoría.
        - ``label``: asigna un entero a cada categoría.

    Ejemplo
    -------
    >>> from synaptix.preprocessing import Encoder
    >>> encoder = Encoder("onehot")
    >>> df_encoded = encoder.fit_transform(df, columns=["ciudad", "genero"])
    """

    def __init__(self, method: Literal["onehot", "label"] = "onehot"):
        if method not in ("onehot", "label"):
            raise ValueError("method debe ser 'onehot' o 'label'")
        self.method = method
        self._encoders: dict = {}
        self._columns: list = []

    def fit(self, df: pd.DataFrame, columns: Optional[list] = None) -> "Encoder":
        """Aprende las categorías de las columnas indicadas.

        Si ``columns`` es None, detecta automáticamente las columnas
        de tipo object o category.
        """
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._columns = columns

        for col in columns:
            if self.method == "label":
                encoder = _SkLabelEncoder()
                encoder.fit(df[col].astype(str))
            else:
                encoder = _SkOneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(df[[col]].astype(str))
            self._encoders[col] = encoder
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la codificación y devuelve un nuevo DataFrame."""
        result = df.copy()
        for col, encoder in self._encoders.items():
            if self.method == "label":
                result[col] = encoder.transform(result[col].astype(str))
            else:
                encoded = encoder.transform(result[[col]].astype(str))
                names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=names, index=result.index)
                result = pd.concat([result.drop(columns=[col]), encoded_df], axis=1)
        return result

    def fit_transform(
        self, df: pd.DataFrame, columns: Optional[list] = None
    ) -> pd.DataFrame:
        """Equivale a ``fit`` seguido de ``transform``."""
        return self.fit(df, columns).transform(df)


class Imputer:
    """Imputador de valores faltantes (NaN).

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent", "constant"}, default="mean"
        Estrategia para columnas numéricas. Las columnas categóricas
        siempre se imputan con la moda (``most_frequent``).
    fill_value : Any, optional
        Valor a usar cuando ``strategy="constant"``.

    Ejemplo
    -------
    >>> from synaptix.preprocessing import Imputer
    >>> imputer = Imputer("median")
    >>> df_limpio = imputer.fit_transform(df)
    """

    def __init__(
        self,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean",
        fill_value=None,
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        self._num_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None
        self._num_cols: list = []
        self._cat_cols: list = []

    def fit(self, df: pd.DataFrame) -> "Imputer":
        """Aprende los valores de imputación por columna."""
        self._num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if self._num_cols:
            self._num_imputer = SimpleImputer(
                strategy=self.strategy, fill_value=self.fill_value
            )
            self._num_imputer.fit(df[self._num_cols])

        if self._cat_cols:
            self._cat_imputer = SimpleImputer(strategy="most_frequent")
            self._cat_imputer.fit(df[self._cat_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rellena los valores faltantes y devuelve un nuevo DataFrame."""
        result = df.copy()
        if self._num_imputer is not None and self._num_cols:
            result[self._num_cols] = self._num_imputer.transform(result[self._num_cols])
        if self._cat_imputer is not None and self._cat_cols:
            result[self._cat_cols] = self._cat_imputer.transform(result[self._cat_cols])
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Equivale a ``fit`` seguido de ``transform``."""
        return self.fit(df).transform(df)
