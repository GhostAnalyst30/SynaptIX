"""
synaptix.datasets - Datasets de ejemplo incluidos en el paquete.

Disponibles:
- ``iris``              : clasificación de especies de flores (150 filas).
- ``penguins``          : pingüinos de Palmer, clasificación (344 filas).
- ``titanic``           : pasajeros del Titanic, clasificación binaria.
- ``sp500_companies``   : empresas del S&P 500, datos financieros.
- ``course_completion`` : finalización de cursos en línea (dataset grande).

La firma de ``load_dataset`` es compatible con la de statslibx y viewx:
soporta ``backend="polars"`` (requiere polars instalado) y
``return_X_y=True`` para obtener directamente features y objetivo.

Ejemplo
-------
>>> from synaptix.datasets import load_dataset, list_datasets
>>> list_datasets()
>>> df = load_dataset("iris")            # con o sin extensión .csv
>>> X, y = load_dataset("iris", return_X_y=True)
"""

from typing import Optional, Tuple, Union

import io
import pkgutil

import numpy as np
import pandas as pd

_AVAILABLE = [
    "iris",
    "penguins",
    "titanic",
    "sp500_companies",
    "course_completion",
]

# Columna objetivo por defecto de cada dataset (para return_X_y).
_TARGETS = {
    "iris": "species",
    "penguins": "species",
    "titanic": "Survived",
    "course_completion": "Completed",
}

__all__ = ["load_dataset", "list_datasets"]


def list_datasets() -> list:
    """Devuelve la lista de datasets incluidos en el paquete."""
    return list(_AVAILABLE)


def load_dataset(
    name: str,
    backend: str = "pandas",
    return_X_y: Optional[Union[bool, str]] = None,
) -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """Carga un dataset interno del paquete.

    Parameters
    ----------
    name : str
        Nombre del dataset, con o sin extensión ``.csv``.
        Ver :func:`list_datasets` para las opciones.
    backend : {"pandas", "polars"}, default="pandas"
        Formato del DataFrame devuelto. ``"polars"`` requiere tener
        polars instalado (``pip install synaptix[polars]``).
    return_X_y : bool o str, optional
        Si es True, devuelve la tupla ``(X, y)`` como arrays de NumPy
        usando la columna objetivo por defecto del dataset. Si es un
        string, se usa como nombre de la columna objetivo.

    Returns
    -------
    DataFrame o tuple de ndarray
        El dataset completo, o ``(X, y)`` si ``return_X_y`` está activo.

    Ejemplo
    -------
    >>> df = load_dataset("iris")
    >>> X, y = load_dataset("iris", return_X_y=True)
    """
    if not name.endswith(".csv"):
        name = f"{name}.csv"

    base = name[:-4]
    if base not in _AVAILABLE:
        raise FileNotFoundError(
            f"Dataset '{base}' no encontrado. Disponibles: {_AVAILABLE}"
        )

    data_bytes = pkgutil.get_data("synaptix.datasets", name)
    if data_bytes is None:
        raise FileNotFoundError(f"Dataset '{name}' no encontrado en el paquete.")
    df = pd.read_csv(io.BytesIO(data_bytes))

    if return_X_y:
        target = return_X_y if isinstance(return_X_y, str) else _TARGETS.get(base)
        if target is None:
            raise ValueError(
                f"El dataset '{base}' no tiene columna objetivo por defecto. "
                "Pasa el nombre de la columna: return_X_y='columna'."
            )
        if target not in df.columns:
            raise KeyError(
                f"Columna objetivo '{target}' no existe en '{base}'. "
                f"Columnas: {list(df.columns)}"
            )
        X = df.drop(columns=target).to_numpy()
        y = df[target].to_numpy()
        return X, y

    if backend == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars no está instalado. Instala con: pip install synaptix[polars]"
            ) from exc
        return pl.from_pandas(df)

    if backend != "pandas":
        raise ValueError(f"Backend no soportado: {backend!r}. Usa 'pandas' o 'polars'.")

    return df
