"""
Cross Validation Engine: estrategias de validación según la estructura de
los datos, no solo según la tarea.

- Datos temporales (``time_column``): ``TimeSeriesSplit`` y holdout
  cronológico; nunca se mezclan observaciones futuras con pasadas.
- Datos agrupados (``groups``, incluidos grupos espaciales): las filas de
  un mismo grupo nunca quedan en train y validación a la vez.
- En otro caso: ``StratifiedKFold`` (clasificación) o ``KFold``, siempre
  con ``shuffle=True`` y ``random_state``.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from ..exceptions import AutoMLError

CVStrategy = Literal[
    "auto", "kfold", "stratified", "repeated", "repeated_stratified",
    "group", "stratified_group", "timeseries",
]
STRATEGIES = (
    "auto", "kfold", "stratified", "repeated", "repeated_stratified",
    "group", "stratified_group", "timeseries",
)


def is_splitter(obj: Any) -> bool:
    """True si ``obj`` es un splitter de scikit-learn (o compatible)."""
    return hasattr(obj, "split") and hasattr(obj, "get_n_splits")


def resolve_strategy(
    strategy: str, task: str, has_groups: bool, is_temporal: bool
) -> str:
    """Traduce ``"auto"`` a una estrategia concreta según los datos."""
    if strategy not in STRATEGIES:
        raise AutoMLError(f"cv_strategy debe ser uno de {STRATEGIES} (recibido: {strategy!r}).")
    if strategy != "auto":
        if strategy in ("group", "stratified_group") and not has_groups:
            raise AutoMLError(f"cv_strategy='{strategy}' requiere pasar groups= en fit().")
        if is_temporal and strategy != "timeseries":
            warnings.warn(
                f"Hay time_column pero cv_strategy='{strategy}': la validación "
                "mezclará pasado y futuro.", UserWarning, stacklevel=3)
        return strategy
    if is_temporal:
        return "timeseries"
    if has_groups:
        return "stratified_group" if task == "classification" else "group"
    return "stratified" if task == "classification" else "kfold"


def make_cv(
    cv: Union[int, Any],
    strategy: str,
    task: str,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    is_temporal: bool = False,
    n_repeats: int = 3,
    random_state: Optional[int] = None,
):
    """Construye el splitter de validación cruzada.

    Parameters
    ----------
    cv : int o splitter
        Número de folds, o un objeto con ``split``/``get_n_splits`` que se
        usa tal cual.
    strategy : str
        Ver :data:`STRATEGIES`.
    task : str
        ``"classification"`` o ``"regression"``.
    y : array, optional
        Target de entrenamiento (para validar los folds estratificados).
    groups : array, optional
        Grupo de cada fila.
    is_temporal : bool
        Si los datos tienen orden temporal.

    Returns
    -------
    (splitter, str)
        El splitter y el nombre de la estrategia usada.
    """
    if is_splitter(cv):
        return cv, type(cv).__name__
    if not isinstance(cv, (int, np.integer)) or isinstance(cv, bool) or cv < 2:
        raise AutoMLError(f"cv debe ser un entero >= 2 o un splitter (recibido: {cv!r}).")

    name = resolve_strategy(strategy, task, groups is not None, is_temporal)
    n_splits = int(cv)
    if name in ("stratified", "repeated_stratified", "stratified_group") and y is not None:
        n_splits = _fit_folds_to_classes(n_splits, y)
    if name in ("group", "stratified_group"):
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise AutoMLError("La validación por grupos necesita al menos 2 grupos distintos.")
        if n_groups < n_splits:
            warnings.warn(f"Solo hay {n_groups} grupos: cv se reduce a {n_groups}.",
                          UserWarning, stacklevel=3)
            n_splits = n_groups

    if name == "kfold":
        return KFold(n_splits, shuffle=True, random_state=random_state), name
    if name == "stratified":
        return StratifiedKFold(n_splits, shuffle=True, random_state=random_state), name
    if name == "repeated":
        return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state), name
    if name == "repeated_stratified":
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=random_state), name
    if name == "group":
        return GroupKFold(n_splits), name
    if name == "stratified_group":
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state), name
    return TimeSeriesSplit(n_splits), name


def _fit_folds_to_classes(n_splits: int, y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    smallest = int(counts.min())
    if smallest >= n_splits:
        return n_splits
    if smallest < 2:
        raise AutoMLError(
            "Una clase tiene menos de 2 ejemplos en el entrenamiento: no se puede "
            "validar de forma estratificada. Agrupa o elimina clases muy raras."
        )
    warnings.warn(
        f"La clase menos frecuente tiene {smallest} ejemplos: cv se reduce de "
        f"{n_splits} a {smallest}.", UserWarning, stacklevel=4)
    return smallest


def holdout_split(
    n_samples: int,
    test_size: Union[float, int],
    task: str,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    is_temporal: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Separa el conjunto de prueba antes de ajustar nada.

    - Temporal: las últimas filas (los datos ya vienen ordenados).
    - Con grupos: ``GroupShuffleSplit`` (grupos completos en test).
    - Clasificación: estratificado si todas las clases tienen >= 2 filas.

    Returns
    -------
    (train_idx, test_idx, descripción)
    """
    indices = np.arange(n_samples)
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise AutoMLError(f"test_size debe estar en (0, 1) (recibido: {test_size}).")
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = int(test_size)
    if n_test < 1 or n_test >= n_samples:
        raise AutoMLError(f"test_size={test_size} deja train o test vacío con {n_samples} filas.")

    if is_temporal:
        return indices[:-n_test], indices[-n_test:], "cronológico (últimas filas)"
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))
        return train_idx, test_idx, "por grupos (GroupShuffleSplit)"

    stratify = None
    description = "aleatorio"
    if task == "classification" and y is not None:
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2 and len(counts) <= min(n_test, n_samples - n_test):
            stratify = y
            description = "estratificado"
        else:
            warnings.warn(
                "No se puede estratificar el holdout (clases con < 2 ejemplos); "
                "se usa un split aleatorio.", UserWarning, stacklevel=3)
    train_idx, test_idx = train_test_split(
        indices, test_size=n_test, random_state=random_state, stratify=stratify
    )
    return np.sort(train_idx), np.sort(test_idx), description
