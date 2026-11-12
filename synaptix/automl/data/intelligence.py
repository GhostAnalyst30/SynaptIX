"""
Data Intelligence: análisis de los datos antes de entrenar.

Calcula estadísticas por columna (nulos, cardinalidad, frecuencia del
valor más común), duplicados, desbalance de clases y sospechas de fuga
de información. Nunca elimina columnas: genera advertencias para que el
usuario decida (por ejemplo con ``ignore_columns``).

El perfilado tiene dos backends con el mismo resultado: pandas y polars.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .detection import infer_feature_types

# Umbrales de las advertencias (documentados en el reporte).
HIGH_MISSING = 0.5
QUASI_CONSTANT = 0.99
ID_UNIQUE_RATIO = 0.95
HIGH_CARDINALITY_ABS = 50
HIGH_CARDINALITY_REL = 0.5
LOW_CARDINALITY_NUMERIC = 10
IMBALANCE_RATIO = 5.0
RARE_CLASS_COUNT = 10
LEAKAGE_CORRELATION = 0.99
LEAKAGE_PURITY = 0.99
LEAKAGE_MAX_ROWS = 50_000


@dataclass
class DataWarning:
    """Advertencia sobre la calidad o la metodología."""

    code: str
    message: str
    column: Optional[str] = None
    severity: str = "warning"  # "info" | "warning"


@dataclass
class DataProfile:
    """Perfil de un dataset.

    Attributes
    ----------
    n_rows, n_columns : int
        Forma del dataset (incluye el target).
    memory_mb : float
        Memoria aproximada del dataset.
    n_duplicates : int
        Filas idénticas a otra anterior.
    columns : dict
        Por columna: ``kind``, ``dtype``, ``n_missing``, ``pct_missing``,
        ``n_unique`` y ``top_freq`` (proporción del valor más común).
    feature_types : dict
        Columnas por tipo (numeric, categorical, boolean, datetime).
    target : dict
        Resumen del target.
    warnings : list de DataWarning
        Advertencias detectadas.
    engine : str
        Backend usado para el perfilado.
    seconds : float
        Tiempo de perfilado.
    """

    n_rows: int
    n_columns: int
    memory_mb: float
    n_duplicates: int
    columns: Dict[str, Dict[str, Any]]
    feature_types: Dict[str, List[str]]
    target: Dict[str, Any] = field(default_factory=dict)
    warnings: List[DataWarning] = field(default_factory=list)
    engine: str = "pandas"
    seconds: float = 0.0

    def add(self, code: str, message: str, column: Optional[str] = None,
            severity: str = "warning") -> None:
        self.warnings.append(DataWarning(code, message, column, severity))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = [asdict(w) for w in self.warnings]
        return data

    def summary_frame(self) -> pd.DataFrame:
        """Tabla con una fila por columna."""
        return pd.DataFrame.from_dict(self.columns, orient="index")

    def warnings_frame(self) -> pd.DataFrame:
        """Tabla de advertencias."""
        return pd.DataFrame([asdict(w) for w in self.warnings],
                            columns=["code", "message", "column", "severity"])

    def __repr__(self) -> str:
        return (
            f"<DataProfile {self.n_rows}x{self.n_columns}, "
            f"{len(self.warnings)} advertencias, engine={self.engine}>"
        )


# ---------------------------------------------------------------------------
# Estadísticas por columna
# ---------------------------------------------------------------------------
def _pandas_stats(df: pd.DataFrame) -> tuple:
    n = len(df)
    n_missing = df.isna().sum()
    n_unique = df.nunique(dropna=True)
    stats = {}
    for col in df.columns:
        series = df[col]
        if n_unique[col] == 0:
            top = 0.0
        else:
            top = float(series.value_counts(dropna=True).iloc[0]) / max(n - n_missing[col], 1)
        stats[col] = {
            "dtype": str(series.dtype),
            "n_missing": int(n_missing[col]),
            "pct_missing": float(n_missing[col] / n) if n else 0.0,
            "n_unique": int(n_unique[col]),
            "top_freq": top,
            "integer": bool(ptypes.is_integer_dtype(series)),
        }
    deep = n <= 100_000
    memory_mb = float(df.memory_usage(deep=deep).sum()) / 1e6
    n_duplicates = int(df.duplicated().sum())
    return stats, memory_mb, n_duplicates


def _polars_stats(frame) -> tuple:
    import polars as pl

    n = frame.height
    lazy = frame.lazy()
    exprs = []
    for col in frame.columns:
        c = pl.col(col)
        # Repetir ``c.drop_nulls()`` en varias expresiones del mismo select
        # lazy falla en polars 2.x ("zip node received non-equal length").
        exprs += [
            c.null_count().alias(f"{col}__nulls"),
            (c.n_unique() - c.null_count().gt(0).cast(pl.UInt32)).alias(f"{col}__unique"),
            c.drop_nulls().value_counts(sort=True).struct.field("count").first()
            .alias(f"{col}__top"),
        ]
    row = lazy.select(exprs).collect().row(0, named=True)
    stats = {}
    for col, dtype in frame.schema.items():
        nulls = int(row[f"{col}__nulls"])
        top = row[f"{col}__top"]
        stats[col] = {
            "dtype": str(dtype),
            "n_missing": nulls,
            "pct_missing": nulls / n if n else 0.0,
            "n_unique": int(row[f"{col}__unique"]),
            "top_freq": float(top) / max(n - nulls, 1) if top is not None else 0.0,
            "integer": bool(dtype.is_integer()),
        }
    memory_mb = float(frame.estimated_size("mb"))
    n_duplicates = int(n - frame.unique().height)
    return stats, memory_mb, n_duplicates


# ---------------------------------------------------------------------------
# Perfil completo
# ---------------------------------------------------------------------------
def profile_data(
    df: pd.DataFrame,
    target: str,
    task: Optional[str] = None,
    polars_frame: Any = None,
) -> DataProfile:
    """Perfila el dataset completo y genera advertencias.

    Parameters
    ----------
    df : DataFrame
        Datos en pandas (se usan para la inferencia de tipos).
    target : str
        Columna objetivo.
    task : str, optional
        Tarea, para analizar el balance de clases.
    polars_frame : polars.DataFrame, optional
        Si se pasa, las estadísticas se calculan con polars.
    """
    start = time.perf_counter()
    if polars_frame is not None:
        stats, memory_mb, n_duplicates = _polars_stats(polars_frame)
        engine = "polars"
    else:
        stats, memory_mb, n_duplicates = _pandas_stats(df)
        engine = "pandas"

    features = df.drop(columns=[target])
    feature_types = infer_feature_types(features)
    kind_of = {col: kind for kind, cols in feature_types.items() for col in cols}
    for col, info in stats.items():
        info["kind"] = "target" if col == target else kind_of.get(col, "unknown")

    profile = DataProfile(
        n_rows=len(df),
        n_columns=df.shape[1],
        memory_mb=round(memory_mb, 3),
        n_duplicates=n_duplicates,
        columns=stats,
        feature_types=feature_types,
        engine=engine,
    )
    _column_warnings(profile, target)
    _target_summary(profile, df[target], task)
    profile.seconds = time.perf_counter() - start
    return profile


def _column_warnings(profile: DataProfile, target: str) -> None:
    n = profile.n_rows
    for col, info in profile.columns.items():
        if col == target:
            continue
        kind = info["kind"]
        non_null = n - info["n_missing"]
        if info["n_missing"] == n:
            profile.add("all_missing", "La columna está completamente vacía.", col)
            continue
        if info["pct_missing"] >= HIGH_MISSING:
            profile.add("high_missing",
                        f"{info['pct_missing']:.0%} de valores faltantes.", col)
        elif info["n_missing"] > 0:
            profile.add("missing", f"{info['n_missing']} valores faltantes "
                        f"({info['pct_missing']:.1%}); se imputan dentro de cada fold.",
                        col, "info")
        if info["n_unique"] <= 1:
            profile.add("constant", "Columna constante: no aporta información. "
                        "Usa drop_constant=True o ignore_columns para excluirla.", col)
            continue
        if info["top_freq"] >= QUASI_CONSTANT:
            profile.add("quasi_constant",
                        f"Un solo valor ocupa el {info['top_freq']:.1%} de las filas.", col)
        unique_ratio = info["n_unique"] / max(non_null, 1)
        if n >= 20 and unique_ratio >= ID_UNIQUE_RATIO and (
            info["integer"] or kind == "categorical"
        ):
            profile.add("potential_id",
                        f"{unique_ratio:.0%} de valores únicos: posible identificador. "
                        "Si lo es, exclúyelo con ignore_columns.", col)
        elif kind == "categorical" and (
            info["n_unique"] > HIGH_CARDINALITY_ABS
            or info["n_unique"] > HIGH_CARDINALITY_REL * non_null
        ):
            profile.add("high_cardinality",
                        f"{info['n_unique']} categorías: las poco frecuentes se agrupan "
                        "como 'infrequent' en la codificación.", col)
        if kind == "numeric" and info["n_unique"] <= LOW_CARDINALITY_NUMERIC:
            profile.add("low_cardinality_numeric",
                        f"Numérica con {info['n_unique']} valores: se trata como numérica. "
                        "Si es un código categórico, conviértela a str/category.",
                        col, "info")
    if profile.n_duplicates:
        profile.add("duplicates",
                    f"{profile.n_duplicates} filas duplicadas. No se eliminan; si caen "
                    "en train y test a la vez inflan la evaluación.", None)


def _target_summary(profile: DataProfile, y: pd.Series, task: Optional[str]) -> None:
    n_missing = int(y.isna().sum())
    summary: Dict[str, Any] = {"name": y.name, "dtype": str(y.dtype), "n_missing": n_missing}
    if n_missing:
        profile.add("target_missing",
                    f"{n_missing} filas sin target: se descartan antes de entrenar.",
                    str(y.name))
    if task == "classification":
        counts = y.value_counts(dropna=True)
        summary["classes"] = {str(k): int(v) for k, v in counts.items()}
        summary["n_classes"] = int(len(counts))
        if len(counts) >= 2:
            ratio = float(counts.iloc[0] / counts.iloc[-1])
            summary["imbalance_ratio"] = round(ratio, 3)
            if ratio >= IMBALANCE_RATIO:
                profile.add("class_imbalance",
                            f"Desbalance de clases {ratio:.1f}:1. accuracy puede ser "
                            "engañosa: considera scoring='balanced_accuracy' o 'f1_weighted'.",
                            str(y.name))
            rare = [str(k) for k, v in counts.items() if v < RARE_CLASS_COUNT]
            if rare:
                profile.add("rare_class",
                            f"Clases con menos de {RARE_CLASS_COUNT} ejemplos: {rare}. "
                            "Las métricas por clase serán inestables.", str(y.name))
    elif task == "regression":
        values = pd.to_numeric(y, errors="coerce").dropna()
        if len(values):
            summary.update({
                "mean": float(values.mean()), "std": float(values.std()),
                "min": float(values.min()), "max": float(values.max()),
                "skew": float(values.skew()) if len(values) > 2 else 0.0,
            })
    profile.target = summary


# ---------------------------------------------------------------------------
# Sospechas de fuga (solo con datos de entrenamiento)
# ---------------------------------------------------------------------------
def leakage_checks(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    target_name: str,
    random_state: Optional[int] = 0,
) -> List[DataWarning]:
    """Busca features sospechosamente predictivas del target.

    Se ejecuta solo sobre el conjunto de entrenamiento. Las señales son
    heurísticas: una advertencia no prueba fuga, indica qué revisar.

    - Nombre de la columna que contiene el nombre del target.
    - Regresión: correlación absoluta >= 0.99 con el target.
    - Clasificación: la feature sola (por valor o por cuantiles) predice
      la clase con >= 99 % de acierto.
    """
    found: List[DataWarning] = []
    if len(X) > LEAKAGE_MAX_ROWS:
        idx = np.random.default_rng(random_state).choice(len(X), LEAKAGE_MAX_ROWS, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    y = pd.Series(np.asarray(y), index=X.index)

    lowered = str(target_name).lower()
    if len(lowered) >= 3:
        for col in X.columns:
            if lowered in str(col).lower():
                found.append(DataWarning(
                    "leakage_name",
                    f"El nombre contiene el del target ('{target_name}'): verifica que "
                    "no se calcule a partir de él.", col))

    if task == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        numeric = X.select_dtypes(include=[np.number])
        if not numeric.empty and y_num.std() > 0:
            corr = numeric.corrwith(y_num).abs()
            for col, value in corr.items():
                if value >= LEAKAGE_CORRELATION:
                    found.append(DataWarning(
                        "leakage_correlation",
                        f"|correlación| = {value:.4f} con el target: posible fuga.", col))
        return found

    majority = y.value_counts(normalize=True).iloc[0]
    if majority >= LEAKAGE_PURITY:
        return found
    for col in X.columns:
        series = X[col]
        if series.nunique(dropna=True) <= 1:
            continue
        if ptypes.is_numeric_dtype(series) and series.nunique() > 20:
            keys = pd.qcut(series, q=20, duplicates="drop")
        elif series.nunique() <= 0.5 * len(series):
            keys = series
        else:
            continue
        table = pd.crosstab(keys, y)
        purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
        if purity >= LEAKAGE_PURITY:
            found.append(DataWarning(
                "leakage_purity",
                f"Esta feature sola predice el target con {purity:.1%} de acierto: "
                "posible fuga (o una feature excepcionalmente buena).", col))
    return found


def train_test_overlap(X_train: pd.DataFrame, X_test: pd.DataFrame) -> int:
    """Número de filas de test idénticas a alguna fila de entrenamiento."""
    hash_train = pd.util.hash_pandas_object(X_train, index=False).to_numpy()
    hash_test = pd.util.hash_pandas_object(X_test, index=False).to_numpy()
    return int(np.isin(hash_test, hash_train).sum())
