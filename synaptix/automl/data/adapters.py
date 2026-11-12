"""
Capa de entrada de datos: pandas, polars (eager y lazy), NumPy y archivos.

Pandas es la interfaz principal. Polars es opcional (``pip install
synaptix[automl]``) y solo se usa donde aporta según los benchmarks
(``benchmarks/bench_engines.py``): perfilado columnar de datos grandes y
entradas que ya son polars (incluida la lectura diferida de un
``LazyFrame`` con proyección de columnas). La conversión a pandas para
scikit-learn ocurre una sola vez.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import AutoMLError, MissingDependencyError

Engine = Literal["auto", "pandas", "polars"]
InputKind = Literal["pandas", "polars", "polars_lazy", "numpy", "path"]

# engine="auto" perfila con polars a partir de este número de celdas
# (filas x columnas). Por debajo la ganancia medida es < 0.15 s y la memoria
# pico se duplica (benchmarks/results/README.md).
AUTO_POLARS_PROFILE_MIN_CELLS = 1_000_000

_FILE_SUFFIXES = (".csv", ".parquet", ".pq")


def polars_available() -> bool:
    """True si polars está instalado."""
    try:
        import polars  # noqa: F401
    except ImportError:
        return False
    return True


def _import_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise MissingDependencyError("polars", "automl", "engine='polars'") from exc
    return pl


def input_kind(data: Any) -> InputKind:
    """Detecta el tipo de entrada sin convertirla."""
    if isinstance(data, pd.DataFrame):
        return "pandas"
    if isinstance(data, (str, os.PathLike)):
        if str(data).lower().endswith(_FILE_SUFFIXES):
            return "path"
        raise AutoMLError(f"Ruta no soportada (usa {_FILE_SUFFIXES}): {data!r}")
    module = type(data).__module__.split(".")[0]
    if module == "polars":
        name = type(data).__name__
        if name == "LazyFrame":
            return "polars_lazy"
        if name == "DataFrame":
            return "polars"
    if isinstance(data, np.ndarray):
        return "numpy"
    raise AutoMLError(
        f"Tipo de datos no soportado: {type(data).__name__}. Usa pandas.DataFrame, "
        "polars.DataFrame/LazyFrame, numpy.ndarray o una ruta .csv/.parquet."
    )


def resolve_engine(engine: Engine, kind: InputKind) -> str:
    """Elige el motor de lectura.

    - Explícito: se respeta (``"polars"`` exige polars instalado).
    - ``"auto"``: polars solo si la entrada ya es polars. Los archivos y los
      DataFrames de pandas se leen con pandas: leer con polars y convertir
      a pandas (scikit-learn lo necesita) no fue más rápido en los
      benchmarks. El perfilado sí puede usar polars, ver
      :func:`use_polars_profiling`.
    """
    if engine not in ("auto", "pandas", "polars"):
        raise AutoMLError(f"engine debe ser 'auto', 'pandas' o 'polars' (recibido: {engine!r}).")
    if engine == "polars":
        _import_polars()
        return "polars"
    if engine == "pandas":
        return "pandas"
    if kind in ("polars", "polars_lazy"):
        return "polars"
    return "pandas"


def use_polars_profiling(engine: Engine, n_cells: int, memory_optimization: bool = False) -> bool:
    """Con ``engine="auto"``, decide si el perfilado de datos en pandas se hace con polars.

    Solo con polars instalado, ``n_cells >= AUTO_POLARS_PROFILE_MIN_CELLS``
    y sin ``memory_optimization`` (la copia en polars duplica la memoria
    pico del perfilado).
    """
    return (engine == "auto" and not memory_optimization
            and n_cells >= AUTO_POLARS_PROFILE_MIN_CELLS and polars_available())


def pandas_to_polars(frame: pd.DataFrame) -> Any:
    """Copia columnar en polars para el perfilado (``None`` si no es convertible)."""
    pl = _import_polars()
    try:
        return pl.from_pandas(frame)
    except Exception:  # noqa: BLE001 - columnas object con tipos mezclados
        return None


def polars_to_pandas(frame: Any) -> pd.DataFrame:
    """Convierte un DataFrame de polars a pandas una única vez.

    Usa la ruta de pyarrow si está disponible; si no, convierte columna a
    columna con ``to_numpy`` (sin copia para numéricas sin nulos).
    """
    try:
        return frame.to_pandas()
    except (ImportError, ModuleNotFoundError):
        pass
    pl = _import_polars()
    columns = {}
    for name, series in zip(frame.columns, frame.get_columns()):
        dtype = series.dtype
        if dtype == pl.Categorical or dtype == pl.Enum:
            columns[name] = pd.Categorical(series.cast(pl.Utf8).to_list())
        elif dtype == pl.Utf8 or dtype == pl.Boolean and series.null_count() > 0:
            columns[name] = pd.Series(series.to_list(), dtype=object)
        else:
            columns[name] = series.to_numpy()
    return pd.DataFrame(columns)


@dataclass
class LoadedData:
    """Datos de entrada normalizados.

    Attributes
    ----------
    frame : DataFrame
        Datos en pandas listos para scikit-learn.
    kind : str
        Tipo de la entrada original.
    engine : str
        Motor usado para leer y perfilar.
    polars_frame : polars.DataFrame o None
        Versión polars materializada (solo con engine polars), reutilizada
        por el perfilado para no volver a convertir.
    load_seconds : float
        Tiempo de lectura y materialización.
    """

    frame: pd.DataFrame
    kind: str
    engine: str
    polars_frame: Any = None
    load_seconds: float = 0.0
    convert_seconds: float = 0.0
    notes: list = field(default_factory=list)


def load_data(
    data: Any,
    target: Union[str, int, Sequence, np.ndarray, pd.Series],
    engine: Engine = "auto",
    drop_columns: Optional[Sequence[str]] = None,
) -> tuple:
    """Normaliza la entrada a un DataFrame de pandas y el nombre del target.

    Parameters
    ----------
    data : DataFrame, polars.DataFrame, polars.LazyFrame, ndarray o ruta
        Datos de entrada.
    target : str, int o array-like
        Nombre de la columna objetivo; con NumPy también se acepta el
        índice de columna o un vector ``y`` aparte.
    engine : {"auto", "pandas", "polars"}
        Motor de lectura y perfilado.
    drop_columns : sequence of str, optional
        Columnas a no leer (proyección en lectura diferida).

    Returns
    -------
    (LoadedData, str)
    """
    kind = input_kind(data)
    drop = set(drop_columns or [])
    start = time.perf_counter()

    if kind == "numpy":
        loaded, target_name = _from_numpy(data, target)
        loaded.load_seconds = time.perf_counter() - start
        return loaded, target_name

    if not isinstance(target, str):
        raise AutoMLError("target debe ser el nombre (str) de la columna objetivo.")

    if kind == "path" and not os.path.exists(data):
        raise FileNotFoundError(data)

    chosen = resolve_engine(engine, kind)

    if chosen == "pandas":
        if kind == "pandas":
            frame = data
        elif kind == "path":
            frame = _read_pandas(str(data))
        else:  # polars de entrada con engine="pandas" explícito
            eager = data.collect() if kind == "polars_lazy" else data
            frame = polars_to_pandas(eager)
        if target not in frame.columns:
            raise AutoMLError(f"La columna '{target}' no existe en los datos.")
        keep = [c for c in frame.columns if c not in drop or c == target]
        if len(keep) != frame.shape[1]:
            frame = frame[keep]
        return (
            LoadedData(frame, kind, "pandas", load_seconds=time.perf_counter() - start),
            target,
        )

    pl = _import_polars()
    if kind == "path":
        path = str(data)
        lazy = pl.scan_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pl.scan_csv(path)
    elif kind == "polars_lazy":
        lazy = data
    elif kind == "polars":
        lazy = data.lazy()
    else:  # pandas con engine="polars" explícito
        lazy = pl.from_pandas(data).lazy()

    schema_names = lazy.collect_schema().names()
    if target not in schema_names:
        raise AutoMLError(f"La columna '{target}' no existe en los datos.")
    selected = [c for c in schema_names if c not in drop or c == target]
    eager = lazy.select(selected).collect()
    load_seconds = time.perf_counter() - start

    convert_start = time.perf_counter()
    frame = data if kind == "pandas" and not drop else polars_to_pandas(eager)
    convert_seconds = time.perf_counter() - convert_start
    return (
        LoadedData(
            frame,
            kind,
            "polars",
            polars_frame=eager,
            load_seconds=load_seconds,
            convert_seconds=convert_seconds,
        ),
        target,
    )


def _read_pandas(path: str) -> pd.DataFrame:
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _from_numpy(data: np.ndarray, target) -> tuple:
    if data.ndim != 2:
        raise AutoMLError(f"Se esperaba un array 2D (recibido: ndim={data.ndim}).")
    columns = [f"x{i}" for i in range(data.shape[1])]
    if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
        if not -data.shape[1] <= target < data.shape[1]:
            raise AutoMLError(f"Índice de target fuera de rango: {target}.")
        frame = pd.DataFrame(data, columns=columns)
        target_name = columns[int(target)]
        frame = _restore_numeric(frame)
        return LoadedData(frame, "numpy", "pandas"), target_name
    if isinstance(target, str) and target in columns:
        frame = _restore_numeric(pd.DataFrame(data, columns=columns))
        return LoadedData(frame, "numpy", "pandas"), target
    y = np.asarray(target)
    if y.ndim != 1 or len(y) != data.shape[0]:
        raise AutoMLError(
            "Con un array NumPy, target debe ser un índice de columna o un vector "
            f"con {data.shape[0]} valores."
        )
    frame = _restore_numeric(pd.DataFrame(data, columns=columns))
    frame["target"] = y
    return LoadedData(frame, "numpy", "pandas"), "target"


def _restore_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Un ndarray de dtype object pierde los tipos: recupera las numéricas."""
    if all(dtype != object for dtype in frame.dtypes):
        return frame
    for col in frame.columns:
        if frame[col].dtype == object:
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().sum() == frame[col].notna().sum():
                frame[col] = converted
    return frame
