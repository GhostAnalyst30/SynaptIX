"""Detección y eliminación de outliers."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detecta outliers en columnas numéricas.

    Parameters
    ----------
    df : DataFrame
        Datos a analizar.
    columns : list, optional
        Columnas a revisar. Por defecto, todas las numéricas.
    method : {"iqr", "zscore"}, default="iqr"
        - ``iqr``: fuera de ``[Q1 - t*IQR, Q3 + t*IQR]``.
        - ``zscore``: |z| mayor que ``threshold`` (usa 3.0 como valor típico).
    threshold : float, default=1.5
        Umbral del método (1.5 para IQR, 3.0 recomendado para z-score).

    Returns
    -------
    DataFrame
        Máscara booleana con la misma forma que las columnas analizadas;
        True indica que el valor es outlier.

    Ejemplo
    -------
    >>> from synaptix.preprocessing import detect_outliers
    >>> mask = detect_outliers(df, method="iqr")
    >>> mask.sum()  # cantidad de outliers por columna
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    mask = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        series = df[col]
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            mask[col] = (series < lower) | (series > upper)
        elif method == "zscore":
            std = series.std()
            if std == 0 or np.isnan(std):
                continue
            z = (series - series.mean()) / std
            mask[col] = z.abs() > threshold
        else:
            raise ValueError("method debe ser 'iqr' o 'zscore'")

    return mask


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Elimina las filas que contienen outliers.

    Parameters
    ----------
    df : DataFrame
        Datos a limpiar.
    columns : list, optional
        Columnas a revisar. Por defecto, todas las numéricas.
    method : {"iqr", "zscore"}, default="iqr"
        Método de detección (ver :func:`detect_outliers`).
    threshold : float, default=1.5
        Umbral del método.

    Returns
    -------
    DataFrame
        Copia de ``df`` sin las filas con outliers.

    Ejemplo
    -------
    >>> from synaptix.preprocessing import remove_outliers
    >>> df_limpio = remove_outliers(df, columns=["precio"], method="iqr")
    """
    mask = detect_outliers(df, columns=columns, method=method, threshold=threshold)
    rows_with_outliers = mask.any(axis=1)
    return df.loc[~rows_with_outliers].copy()
