import warnings

import numpy as np
import pandas as pd
import pytest

from synaptix.automl.data import (
    detect_task,
    detect_task_details,
    infer_feature_types,
    leakage_checks,
    load_data,
    profile_data,
    resolve_engine,
)
from synaptix.automl.data.intelligence import train_test_overlap
from synaptix.automl.exceptions import AutoMLError

from .conftest import make_classification_df


# ---------------------------------------------------------------- detección
@pytest.mark.parametrize(
    "values, task, confidence",
    [
        (["a", "b", "a"] * 10, "classification", "high"),
        ([True, False] * 10, "classification", "high"),
        (np.linspace(0, 1, 50), "regression", "high"),
        ([0, 1] * 30, "classification", "high"),
        (np.random.default_rng(0).integers(0, 4, 1000), "classification", "medium"),
        (np.arange(500), "regression", "medium"),
    ],
)
def test_detect_task_details(values, task, confidence):
    result = detect_task_details(pd.Series(values))
    assert result.task == task
    assert result.confidence == confidence
    assert result.reasons


def test_detect_task_ambiguous_warns():
    # 12 valores enteros consecutivos en 40 filas: ambiguo
    y = pd.Series(np.random.default_rng(1).integers(1, 13, 40))
    with pytest.warns(UserWarning, match="ambigua"):
        detect_task(y)


def test_low_cardinality_numeric_is_not_categorical():
    df = pd.DataFrame({"hijos": [0, 1, 2, 3] * 10, "ciudad": ["x", "y"] * 20,
                       "activo": [True, False] * 20,
                       "fecha": pd.date_range("2020-01-01", periods=40)})
    types = infer_feature_types(df)
    assert types["numeric"] == ["hijos"]
    assert types["categorical"] == ["ciudad"]
    assert types["boolean"] == ["activo"]
    assert types["datetime"] == ["fecha"]


# ---------------------------------------------------------------- perfilado
def _codes(profile):
    return {(w.code, w.column) for w in profile.warnings}


def test_profile_warnings():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "id": np.arange(n),
        "constante": 1.0,
        "casi": [0] * 199 + [1],
        "vacia": np.nan,
        "nulos": np.where(rng.random(n) < 0.6, np.nan, rng.normal(size=n)),
        "x": rng.normal(size=n),
        "y": ["a"] * 190 + ["b"] * 10,
    })
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    profile = profile_data(df, "y", task="classification")
    codes = _codes(profile)
    assert ("constant", "constante") in codes
    assert ("quasi_constant", "casi") in codes
    assert ("all_missing", "vacia") in codes
    assert ("high_missing", "nulos") in codes
    assert ("duplicates", None) in codes
    assert ("class_imbalance", "y") in codes
    assert profile.n_duplicates == 3
    assert profile.target["n_classes"] == 2
    # nunca elimina columnas
    assert set(profile.columns) == set(df.columns)


def test_potential_id_and_low_cardinality_numeric():
    df = pd.DataFrame({"cliente": [f"C{i}" for i in range(100)],
                       "nivel": [1, 2, 3] * 33 + [1],
                       "y": [0, 1] * 50})
    codes = _codes(profile_data(df, "y", task="classification"))
    assert ("potential_id", "cliente") in codes
    assert ("low_cardinality_numeric", "nivel") in codes


def test_leakage_checks_detect_target_copy():
    df = make_classification_df(n=400)
    X = df.drop(columns="y").assign(y_codificado=(df["y"] == "si").astype(int))
    found = {(w.code, w.column) for w in leakage_checks(X, df["y"], "classification", "y_cod")}
    assert ("leakage_purity", "y_codificado") in found
    # una feature legítima no se marca
    assert not any(col == "c" for _, col in found)


def test_leakage_checks_regression_correlation():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=300))
    X = pd.DataFrame({"copia": y * 2 + 1e-4 * rng.normal(size=300), "ruido": rng.normal(size=300)})
    found = {(w.code, w.column) for w in leakage_checks(X, y, "regression", "precio")}
    assert ("leakage_correlation", "copia") in found
    assert ("leakage_correlation", "ruido") not in found


def test_train_test_overlap():
    a = pd.DataFrame({"x": [1, 2, 3], "z": ["a", "b", "c"]})
    b = pd.DataFrame({"x": [3, 4], "z": ["c", "d"]})
    assert train_test_overlap(a, b) == 1


# ---------------------------------------------------------------- adaptadores
def test_numpy_input_with_vector_target():
    X = np.random.default_rng(0).normal(size=(20, 3))
    loaded, target = load_data(X, np.arange(20) % 2)
    assert target == "target"
    assert list(loaded.frame.columns) == ["x0", "x1", "x2", "target"]


def test_numpy_input_with_column_index():
    X = np.random.default_rng(0).normal(size=(20, 3))
    loaded, target = load_data(X, 2)
    assert target == "x2"


def test_engine_resolution_without_polars_input():
    assert resolve_engine("auto", "pandas") == "pandas"
    assert resolve_engine("auto", "numpy") == "pandas"
    assert resolve_engine("auto", "path") == "pandas"
    with pytest.raises(AutoMLError):
        resolve_engine("spark", "pandas")


def test_auto_engine_profiles_large_data_with_polars(monkeypatch):
    pytest.importorskip("polars")
    import synaptix as sx
    from synaptix.automl.data import adapters

    assert adapters.use_polars_profiling("auto", 2_000_000)
    assert not adapters.use_polars_profiling("auto", 2_000_000, memory_optimization=True)
    assert not adapters.use_polars_profiling("auto", 10_000)
    assert not adapters.use_polars_profiling("pandas", 2_000_000)

    monkeypatch.setattr(adapters, "AUTO_POLARS_PROFILE_MIN_CELLS", 100)
    df = make_classification_df(n=120)
    kwargs = dict(mode="fast", cv=3, verbose=False, explain=False, models=["logistic"])
    pipe = sx.AutoPipeline(**kwargs).fit(df, "y")
    assert pipe._engine_used == "pandas"
    assert pipe.profile_.engine == "polars"
    lean = sx.AutoPipeline(**kwargs, memory_optimization=True).fit(df, "y")
    assert lean.profile_.engine == "pandas"
    def stats(profile):
        return {col: {k: v for k, v in info.items() if k != "dtype"}
                for col, info in profile.columns.items()}

    assert stats(lean.profile_) == stats(pipe.profile_)


def test_pandas_input_is_not_copied():
    df = make_classification_df(n=50)
    loaded, _ = load_data(df, "y")
    assert loaded.frame is df


def test_missing_target_column():
    with pytest.raises(AutoMLError):
        load_data(pd.DataFrame({"a": [1]}), "y")


# ---------------------------------------------------------------- polars
def test_polars_profile_matches_pandas():
    pl = pytest.importorskip("polars")
    df = make_classification_df(n=200, with_nulls=True)
    df["flag"] = df["flag"].astype(bool)
    profile_pd = profile_data(df, "y", task="classification")
    profile_pl = profile_data(df, "y", task="classification", polars_frame=pl.from_pandas(df))
    assert profile_pl.engine == "polars"
    assert profile_pl.n_duplicates == profile_pd.n_duplicates
    for col in df.columns:
        a, b = profile_pd.columns[col], profile_pl.columns[col]
        assert a["n_missing"] == b["n_missing"], col
        assert a["n_unique"] == b["n_unique"], col
        assert a["top_freq"] == pytest.approx(b["top_freq"]), col
    assert _codes(profile_pd) == _codes(profile_pl)


def test_polars_lazy_projection_and_conversion():
    pl = pytest.importorskip("polars")
    df = make_classification_df(n=100)
    lazy = pl.from_pandas(df).lazy()
    loaded, target = load_data(lazy, "y", engine="auto", drop_columns=["d"])
    assert loaded.engine == "polars"
    assert "d" not in loaded.frame.columns
    assert isinstance(loaded.frame, pd.DataFrame)
    assert loaded.frame.shape == (100, df.shape[1] - 1)
