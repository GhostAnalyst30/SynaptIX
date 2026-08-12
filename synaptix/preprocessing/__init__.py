"""
synaptix.preprocessing - Preprocesamiento de datos.

Incluye:
- ``Scaler``            : escalado standard / minmax / robust.
- ``Encoder``           : codificación onehot / label.
- ``Imputer``           : imputación de valores faltantes.
- ``DataCleaner``       : análisis y limpieza automática de DataFrames.
- ``detect_outliers``   : detección de outliers (IQR / z-score).
- ``remove_outliers``   : eliminación de filas con outliers.
- ``train_test_split``  : división train/test (de scikit-learn).

Ejemplo
-------
>>> from synaptix.preprocessing import DataCleaner, train_test_split
>>> cleaner = DataCleaner()
>>> df = cleaner.clean(df, target="precio")
>>> X_train, X_test, y_train, y_test = train_test_split(
...     df.drop(columns="precio"), df["precio"], test_size=0.2
... )
"""

from sklearn.model_selection import train_test_split

from .cleaner import DataCleaner
from .outliers import detect_outliers, remove_outliers
from .transformers import Encoder, Imputer, Scaler

__all__ = [
    "Scaler",
    "Encoder",
    "Imputer",
    "DataCleaner",
    "detect_outliers",
    "remove_outliers",
    "train_test_split",
]
