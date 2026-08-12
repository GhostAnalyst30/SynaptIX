"""
DataCleaner: análisis y limpieza automática de DataFrames.

Analiza un DataFrame (nulos, tipos, cardinalidad, outliers), reporta
los problemas encontrados y aplica un pipeline de limpieza estándar:
imputación -> codificación -> escalado.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from .outliers import detect_outliers
from .transformers import Encoder, Imputer, Scaler


class DataCleaner:
    """Limpieza automática de datos tabulares.

    Parameters
    ----------
    impute_strategy : {"mean", "median", "most_frequent"}, default="median"
        Estrategia de imputación para columnas numéricas.
    encode_method : {"onehot", "label"}, default="onehot"
        Codificación para columnas categóricas.
    scale_method : {"standard", "minmax", "robust"} o None, default="standard"
        Escalado para columnas numéricas. None para no escalar.

    Ejemplo
    -------
    >>> from synaptix.preprocessing import DataCleaner
    >>> cleaner = DataCleaner()
    >>> cleaner.analyze(df)          # reporte de problemas
    >>> df_listo = cleaner.clean(df) # imputa + codifica + escala
    """

    def __init__(
        self,
        impute_strategy: Literal["mean", "median", "most_frequent"] = "median",
        encode_method: Literal["onehot", "label"] = "onehot",
        scale_method: Optional[Literal["standard", "minmax", "robust"]] = "standard",
    ):
        self.impute_strategy = impute_strategy
        self.encode_method = encode_method
        self.scale_method = scale_method

        self.imputer_: Optional[Imputer] = None
        self.encoder_: Optional[Encoder] = None
        self.scaler_: Optional[Scaler] = None
        self._scaled_cols: list = []
        self._output_cols: list = []

    def analyze(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """Analiza el DataFrame y reporta problemas de calidad.

        Parameters
        ----------
        df : DataFrame
            Datos a analizar.
        verbose : bool, default=True
            Si es True, imprime el reporte en consola.

        Returns
        -------
        dict
            Resumen con nulos por columna, tipos, duplicados,
            outliers y columnas de alta cardinalidad.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0].to_dict()
        duplicates = int(df.duplicated().sum())

        outlier_counts = {}
        if num_cols:
            mask = detect_outliers(df[num_cols].dropna())
            outlier_counts = {
                col: int(count) for col, count in mask.sum().items() if count > 0
            }

        high_cardinality = [
            col for col in cat_cols if df[col].nunique() > max(20, len(df) * 0.5)
        ]

        report = {
            "filas": len(df),
            "columnas_numericas": num_cols,
            "columnas_categoricas": cat_cols,
            "nulos": nulls,
            "duplicados": duplicates,
            "outliers": outlier_counts,
            "alta_cardinalidad": high_cardinality,
        }

        if verbose:
            print("\n=== Análisis del DataFrame ===")
            print(f"  Filas               : {report['filas']}")
            print(f"  Numéricas           : {len(num_cols)} -> {num_cols}")
            print(f"  Categóricas         : {len(cat_cols)} -> {cat_cols}")
            print(f"  Duplicados          : {duplicates}")
            if nulls:
                print("  Nulos por columna   :")
                for col, count in nulls.items():
                    pct = 100 * count / len(df)
                    print(f"    {col}: {count} ({pct:.1f}%)")
            else:
                print("  Nulos               : ninguno")
            if outlier_counts:
                print(f"  Outliers (IQR)      : {outlier_counts}")
            if high_cardinality:
                print(f"  Alta cardinalidad   : {high_cardinality}")

        return report

    def clean(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Aplica el pipeline de limpieza completo.

        Orden: eliminar duplicados -> imputar nulos -> codificar
        categóricas -> escalar numéricas. La columna ``target`` (si se
        indica) se excluye del escalado y la codificación.

        Parameters
        ----------
        df : DataFrame
            Datos a limpiar.
        target : str, optional
            Nombre de la columna objetivo a preservar sin transformar.
        drop_duplicates : bool, default=True
            Eliminar filas duplicadas.

        Returns
        -------
        DataFrame
            Datos listos para entrenar un modelo.
        """
        result = df.copy()

        if drop_duplicates:
            result = result.drop_duplicates().reset_index(drop=True)

        y = None
        if target is not None:
            y = result[target]
            result = result.drop(columns=[target])

        self.imputer_ = Imputer(strategy=self.impute_strategy)
        result = self.imputer_.fit_transform(result)

        cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            self.encoder_ = Encoder(method=self.encode_method)
            result = self.encoder_.fit_transform(result, columns=cat_cols)

        if self.scale_method is not None:
            num_cols = result.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                self.scaler_ = Scaler(method=self.scale_method)
                result[num_cols] = self.scaler_.fit_transform(result[num_cols])
                self._scaled_cols = num_cols

        self._output_cols = list(result.columns)

        if y is not None:
            result[target] = y.values

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a datos nuevos las transformaciones ya ajustadas por :meth:`clean`.

        Usa el imputador, codificador y escalador aprendidos previamente,
        y alinea las columnas del resultado con las del entrenamiento
        (rellena con 0 las categorías one-hot que no aparezcan).

        Parameters
        ----------
        df : DataFrame
            Datos nuevos con las mismas columnas originales (sin el target).

        Returns
        -------
        DataFrame
            Datos transformados, con las mismas columnas que produjo
            :meth:`clean` en el entrenamiento.

        Ejemplo
        -------
        >>> cleaner = DataCleaner()
        >>> df_train = cleaner.clean(df, target="precio")
        >>> X_nuevo = cleaner.transform(df_nuevo)   # datos de producción
        """
        if self.imputer_ is None:
            raise RuntimeError("Llama a clean(df) primero para ajustar el cleaner.")

        result = df.copy()
        result = self.imputer_.transform(result)

        if self.encoder_ is not None:
            result = self.encoder_.transform(result)

        if self.scaler_ is not None and self._scaled_cols:
            present = [col for col in self._scaled_cols if col in result.columns]
            if present != self._scaled_cols:
                missing = set(self._scaled_cols) - set(present)
                for col in missing:
                    result[col] = 0.0
            result[self._scaled_cols] = self.scaler_.transform(result[self._scaled_cols])

        # Alinear con las columnas del entrenamiento
        for col in self._output_cols:
            if col not in result.columns:
                result[col] = 0.0
        return result[self._output_cols]
