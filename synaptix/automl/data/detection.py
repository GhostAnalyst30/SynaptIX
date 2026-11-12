"""
Detección de la tarea (clasificación o regresión) y de tipos de variables.

La tarea se decide con varios criterios y se devuelve la evidencia. Una
variable numérica con pocos valores únicos no se asume categórica: solo
se reporta como "numérica de baja cardinalidad".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# Máximo de clases que se aceptan para inferir clasificación desde un
# target numérico entero.
MAX_INFERRED_CLASSES = 20


@dataclass
class TaskDetection:
    """Resultado de :func:`detect_task_details`.

    Attributes
    ----------
    task : str
        ``"classification"`` o ``"regression"``.
    confidence : str
        ``"high"``, ``"medium"`` o ``"low"`` (ambiguo).
    reasons : list de str
        Criterios que llevaron a la decisión.
    """

    task: str
    confidence: str
    reasons: List[str] = field(default_factory=list)


def _is_textual(series: pd.Series) -> bool:
    return (
        ptypes.is_object_dtype(series)
        or ptypes.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def detect_task_details(y: pd.Series) -> TaskDetection:
    """Infiere la tarea a partir del target con varios criterios.

    Criterios, en orden:

    1. Booleano, texto o categoría -> clasificación (alta confianza).
    2. Numérico con valores no enteros -> regresión (alta confianza).
    3. Entero con 2 valores -> clasificación binaria (alta confianza).
    4. Entero con <= 20 valores, códigos consecutivos y pocos únicos
       respecto a n -> clasificación (media).
    5. Entero con muchos valores únicos -> regresión (media).
    6. Casos intermedios -> se elige y se marca como ambiguo (baja).
    """
    y = pd.Series(y).dropna()
    if len(y) == 0:
        raise ValueError("El target no tiene valores (todos son nulos).")

    if ptypes.is_bool_dtype(y):
        return TaskDetection("classification", "high", ["target booleano"])
    if _is_textual(y):
        return TaskDetection("classification", "high", [f"target de tipo {y.dtype}"])
    if not ptypes.is_numeric_dtype(y):
        return TaskDetection("classification", "medium", [f"target no numérico ({y.dtype})"])

    values = y.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("El target contiene valores infinitos.")
    is_integer_valued = bool(np.all(np.mod(values, 1) == 0))
    n_unique = int(y.nunique())
    n = len(y)

    if not is_integer_valued:
        return TaskDetection(
            "regression", "high", ["target numérico con valores no enteros"]
        )
    if n_unique == 2:
        return TaskDetection("classification", "high", ["target entero con 2 valores"])
    if n_unique == 1:
        return TaskDetection("classification", "low", ["target constante (una sola clase)"])

    uniques = np.sort(np.unique(values))
    consecutive = bool(np.all(np.diff(uniques) == 1))
    starts_low = uniques[0] in (0, 1)
    ratio = n_unique / n
    reasons = [
        f"target entero con {n_unique} valores únicos ({ratio:.1%} de {n} filas)",
        "códigos consecutivos" if consecutive else "valores no consecutivos",
    ]

    if n_unique <= MAX_INFERRED_CLASSES:
        if ratio <= 0.05 and (consecutive or n_unique <= 10):
            return TaskDetection("classification", "medium", reasons)
        if consecutive and starts_low:
            return TaskDetection("classification", "low", reasons)
        return TaskDetection("regression", "low", reasons)
    return TaskDetection("regression", "medium", reasons)


def detect_task(y: pd.Series) -> str:
    """Detecta si el target es de clasificación o de regresión.

    Combina varios criterios (ver :func:`detect_task_details`). Si el caso
    es ambiguo emite un ``UserWarning`` recomendando indicar ``task=``.
    """
    result = detect_task_details(y)
    if result.confidence == "low":
        warnings.warn(
            f"Detección de tarea ambigua: se eligió '{result.task}' "
            f"({'; '.join(result.reasons)}). Indica task='classification' o "
            "task='regression' para fijarla.",
            UserWarning,
            stacklevel=2,
        )
    return result.task


FEATURE_KINDS = ("numeric", "categorical", "boolean", "datetime")


def infer_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Clasifica las columnas por tipo para el preprocesamiento.

    Returns
    -------
    dict
        ``{"numeric": [...], "categorical": [...], "boolean": [...],
        "datetime": [...]}``. Las numéricas con pocos valores se mantienen
        numéricas.
    """
    types: Dict[str, List[str]] = {kind: [] for kind in FEATURE_KINDS}
    for col in df.columns:
        series = df[col]
        if ptypes.is_bool_dtype(series):
            types["boolean"].append(col)
        elif ptypes.is_datetime64_any_dtype(series):
            types["datetime"].append(col)
        elif ptypes.is_numeric_dtype(series):
            types["numeric"].append(col)
        elif _is_textual(series):
            sample = series.dropna().iloc[:1000]
            if (
                ptypes.is_object_dtype(series)
                and len(sample)
                and sample.map(type).isin([bool, np.bool_]).all()
            ):
                types["boolean"].append(col)
            else:
                types["categorical"].append(col)
        else:
            types["categorical"].append(col)
    return types
