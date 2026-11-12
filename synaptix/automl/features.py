"""
Preprocessing Engine: construcción del preprocesador sin fuga de datos.

Todo se arma como transformadores de scikit-learn y se coloca dentro de un
``sklearn.pipeline.Pipeline`` junto al modelo. Así cada fold de validación
cruzada (y cada combinación de hiperparámetros) reajusta la imputación,
el escalado, la codificación y la selección de features solo con sus datos
de entrenamiento.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .exceptions import AutoMLError

ImputeStrategy = Literal["mean", "median", "most_frequent", "constant", "knn"]
ScaleMethod = Optional[Literal["standard", "minmax", "robust"]]
EncodeMethod = Literal["onehot", "ordinal", "label"]
FeatureSelection = Optional[Literal["variance", "kbest", "model"]]

_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_MISSING_TOKEN = "__missing__"


# ---------------------------------------------------------------------------
# Transformadores sin estado (funciones a nivel de módulo para poder pickle)
# ---------------------------------------------------------------------------
def _to_object_str(X):
    """Convierte categóricas a texto conservando los nulos como NaN."""
    frame = pd.DataFrame(X)
    mask = frame.isna().to_numpy()
    values = np.array(frame.astype(str).to_numpy(dtype=object), dtype=object, copy=True)
    values[mask] = np.nan
    return values


def _to_float(X):
    """Convierte booleanas (incluidas las anulables) a float con NaN."""
    frame = pd.DataFrame(X)
    out = np.empty(frame.shape, dtype=float)
    for i, col in enumerate(frame.columns):
        series = frame[col]
        if series.dtype == object:
            series = series.map({True: 1.0, False: 0.0})
        out[:, i] = (
            pd.to_numeric(series, errors="coerce")
            .astype("Float64")
            .to_numpy(dtype=float, na_value=np.nan)
        )
    return out


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Extrae año, mes, día, día de la semana y hora de columnas fecha.

    No aprende nada de los datos (sin riesgo de fuga); los nulos quedan
    como NaN para el imputador posterior.
    """

    _PARTS = ("year", "month", "day", "dayofweek", "hour")

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        frame.columns = self.columns_
        parts = []
        for col in self.columns_:
            values = pd.to_datetime(frame[col], errors="coerce")
            for part in self._PARTS:
                attr = "day_of_week" if part == "dayofweek" else part
                parts.append(getattr(values.dt, attr).astype("float64").to_numpy())
        return np.column_stack(parts) if parts else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features=None):
        cols = self.columns_ if input_features is None else list(input_features)
        return np.array([f"{c}_{p}" for c in cols for p in self._PARTS], dtype=object)


class ClippedKBest(SelectKBest):
    """``SelectKBest`` que usa todas las features si ``k`` supera su número.

    El número de columnas tras el one-hot no se conoce al construir el
    pipeline; esto evita errores sin cambiar el criterio de selección.
    """

    def _check_params(self, X, y):
        if not (self.k == "all" or (isinstance(self.k, (int, np.integer)) and self.k >= 0)):
            raise ValueError(f"k debe ser un entero >= 0 o 'all' (recibido: {self.k!r}).")


# ---------------------------------------------------------------------------
# Construcción del preprocesador
# ---------------------------------------------------------------------------
def build_preprocessor(
    feature_types: Dict[str, List[str]],
    impute_strategy: ImputeStrategy = "median",
    scale_method: ScaleMethod = "standard",
    encode_method: EncodeMethod = "onehot",
    max_onehot_categories: Optional[int] = 20,
    sparse: bool = False,
) -> ColumnTransformer:
    """Arma el ``ColumnTransformer`` según los tipos de columna.

    Parameters
    ----------
    feature_types : dict
        Salida de :func:`synaptix.automl.data.infer_feature_types`.
    impute_strategy : {"mean", "median", "most_frequent", "constant", "knn"}
        Imputación de numéricas. ``"knn"`` usa ``KNNImputer`` después de
        escalar (las distancias necesitan escalas comparables).
    scale_method : {"standard", "minmax", "robust"} o None
        Escalado de numéricas.
    encode_method : {"onehot", "ordinal"}
        Codificación de categóricas (``"label"`` es alias de ``"ordinal"``).
        Las categorías desconocidas en inferencia no generan error.
    max_onehot_categories : int, optional
        Máximo de columnas one-hot por variable; el resto se agrupa como
        categoría "infrequent".
    sparse : bool
        Salida dispersa del one-hot (ahorra memoria con alta cardinalidad).
    """
    if impute_strategy not in ("mean", "median", "most_frequent", "constant", "knn"):
        raise AutoMLError(f"impute_strategy no soportada: {impute_strategy!r}")
    if scale_method is not None and scale_method not in _SCALERS:
        raise AutoMLError(f"scale_method no soportado: {scale_method!r}")
    if encode_method == "label":
        encode_method = "ordinal"
    if encode_method not in ("onehot", "ordinal"):
        raise AutoMLError(f"encode_method no soportado: {encode_method!r}")

    transformers = []
    numeric = list(feature_types.get("numeric", []))
    if numeric:
        transformers.append(("num", _numeric_pipeline(impute_strategy, scale_method), numeric))

    boolean = list(feature_types.get("boolean", []))
    if boolean:
        transformers.append(("bool", Pipeline([
            ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ]), boolean))

    dates = list(feature_types.get("datetime", []))
    if dates:
        transformers.append(("date", Pipeline([
            ("parts", DatetimeFeatures()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]), dates))

    categorical = list(feature_types.get("categorical", []))
    if categorical:
        if encode_method == "onehot":
            encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=max_onehot_categories,
                sparse_output=sparse,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
            )
        transformers.append(("cat", Pipeline([
            ("to_str", FunctionTransformer(_to_object_str, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=_MISSING_TOKEN)),
            ("encode", encoder),
        ]), categorical))

    if not transformers:
        raise AutoMLError("No hay columnas de entrada utilizables para entrenar.")
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0 if sparse else 0.0,
    )


def _numeric_pipeline(impute_strategy: str, scale_method: ScaleMethod) -> Pipeline:
    scaler = _SCALERS[scale_method]() if scale_method else None
    if impute_strategy == "knn":
        steps = []
        if scaler is not None:
            steps.append(("scale", scaler))
        steps.append(("impute", KNNImputer(keep_empty_features=True)))
        return Pipeline(steps)
    steps = [("impute", SimpleImputer(strategy=impute_strategy, keep_empty_features=True))]
    if scaler is not None:
        steps.append(("scale", scaler))
    return Pipeline(steps)


def build_feature_selector(
    method: FeatureSelection,
    task: str,
    k: Optional[Union[int, float]] = None,
    random_state: Optional[int] = None,
):
    """Selector de features (se ajusta dentro de cada fold).

    Parameters
    ----------
    method : {"variance", "kbest", "model"} o None
        - ``variance``: elimina features constantes.
        - ``kbest``: test F univariante (``k`` entero = número de features,
          float en (0, 1] = proporción; por defecto 50 %).
        - ``model``: importancia de un ExtraTrees (umbral: la mediana).
    """
    if method is None:
        return None
    if method == "variance":
        return VarianceThreshold(0.0)
    score_func = f_classif if task == "classification" else f_regression
    if method == "kbest":
        if k is None:
            k = 0.5
        if isinstance(k, float):
            if not 0 < k <= 1:
                raise AutoMLError("feature_selection_k float debe estar en (0, 1].")
            return SelectPercentile(score_func, percentile=k * 100)
        return ClippedKBest(score_func, k=int(k))
    if method == "model":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        cls = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
        return SelectFromModel(cls(n_estimators=100, random_state=random_state, n_jobs=1),
                               threshold="median")
    raise AutoMLError(f"feature_selection no soportado: {method!r}")


def make_model_pipeline(
    preprocessor,
    model,
    drop_constant: bool = False,
    selector=None,
    memory=None,
) -> Pipeline:
    """Pipeline completo ``prep -> [constantes] -> [selección] -> modelo``."""
    steps = [("prep", preprocessor)]
    if drop_constant:
        steps.append(("drop_constant", VarianceThreshold(0.0)))
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", model))
    return Pipeline(steps, memory=memory)


def output_feature_names(pipeline: Pipeline) -> List[str]:
    """Nombres de las features que recibe el modelo, sin prefijos internos."""
    names = np.asarray(pipeline.named_steps["prep"].get_feature_names_out(), dtype=object)
    for name, step in pipeline.steps[1:-1]:
        if hasattr(step, "get_support"):
            names = names[step.get_support()]
    return [str(n).split("__", 1)[-1] for n in names]


# ---------------------------------------------------------------------------
# Compatibilidad: DataCleaner como transformador de scikit-learn
# ---------------------------------------------------------------------------
class CleanerTransformer(BaseEstimator, TransformerMixin):
    """Envuelve :class:`synaptix.preprocessing.DataCleaner` para usarlo
    dentro de un ``Pipeline`` (se ajusta solo con los datos de cada fold).
    """

    def __init__(self, impute_strategy="median", encode_method="onehot",
                 scale_method="standard"):
        self.impute_strategy = impute_strategy
        self.encode_method = encode_method
        self.scale_method = scale_method

    def fit(self, X, y=None):
        from ..preprocessing import DataCleaner

        self.cleaner_ = DataCleaner(
            impute_strategy=self.impute_strategy,
            encode_method=self.encode_method,
            scale_method=self.scale_method,
        )
        self.cleaner_.clean(pd.DataFrame(X), drop_duplicates=False)
        return self

    def transform(self, X):
        return self.cleaner_.transform(pd.DataFrame(X)).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.cleaner_._output_cols, dtype=object)


def feature_columns(
    df: pd.DataFrame,
    target: str,
    ignore: Optional[Sequence[str]] = None,
) -> List[str]:
    """Columnas de entrada: todas menos el target y las ignoradas."""
    skip = set(ignore or []) | {target}
    return [c for c in df.columns if c not in skip]
