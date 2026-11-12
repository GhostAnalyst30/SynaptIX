"""Selección de columnas compatible con pandas 2 y pandas 3."""

from typing import List

import pandas as pd


def text_columns(df: pd.DataFrame) -> List[str]:
    """Columnas de texto o categóricas (object, str/string o category)."""
    return [
        col for col in df.columns
        if isinstance(df[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(df[col].dtype)
        or pd.api.types.is_string_dtype(df[col].dtype)
    ]
