"""
synaptix.datasets - Datasets de ejemplo incluidos en el paquete.

Disponibles:
- ``iris``              : clasificación de especies de flores (150 filas).
- ``penguins``          : pingüinos de Palmer, clasificación (344 filas).
- ``titanic``           : pasajeros del Titanic, clasificación binaria.
- ``sp500_companies``   : empresas del S&P 500, datos financieros.
- ``course_completion`` : finalización de cursos en línea (dataset grande).

Ejemplo
-------
>>> from synaptix.datasets import load_dataset, list_datasets
>>> list_datasets()
>>> df = load_dataset("iris")   # con o sin extensión .csv
"""

import io
import pkgutil

import pandas as pd

_AVAILABLE = [
    "iris",
    "penguins",
    "titanic",
    "sp500_companies",
    "course_completion",
]

__all__ = ["load_dataset", "list_datasets"]


def list_datasets() -> list:
    """Devuelve la lista de datasets incluidos en el paquete."""
    return list(_AVAILABLE)


def load_dataset(name: str) -> pd.DataFrame:
    """Carga un dataset interno del paquete.

    Parameters
    ----------
    name : str
        Nombre del dataset, con o sin extensión ``.csv``.
        Ver :func:`list_datasets` para las opciones.

    Returns
    -------
    DataFrame
        El dataset como DataFrame de pandas.

    Ejemplo
    -------
    >>> df = load_dataset("iris")
    >>> df.head()
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
    return pd.read_csv(io.BytesIO(data_bytes))
