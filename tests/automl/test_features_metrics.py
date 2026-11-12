import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from synaptix.automl.data import infer_feature_types
from synaptix.automl.evaluation import (
    holdout_split,
    make_cv,
    pr_auc_score,
    resolve_metric,
    resolve_metrics,
    safe_mape,
)
from synaptix.automl.exceptions import AutoMLError
from synaptix.automl.features import (
    build_feature_selector,
    build_preprocessor,
    make_model_pipeline,
    output_feature_names,
)

from .conftest import make_classification_df


# ---------------------------------------------------------------- preprocesamiento
def test_unknown_categories_and_all_nan_column():
    df = make_classification_df(n=120, with_nulls=True)
    df["vacia"] = np.nan
    X, y = df.drop(columns="y"), df["y"]
    pre = build_preprocessor(infer_feature_types(X))
    pipe = make_model_pipeline(pre, LogisticRegression(max_iter=500)).fit(X, y)
    nuevo = X.head(5).copy()
    nuevo["color"] = "morado"  # categoría nunca vista
    nuevo.loc[:, "a"] = np.nan
    assert len(pipe.predict(nuevo)) == 5
    names = output_feature_names(pipe)
    assert "vacia" in names  # la columna vacía se conserva (imputada) sin romper la forma


def test_boolean_with_missing_and_datetime():
    df = pd.DataFrame({
        "activo": pd.array([True, False, None, True] * 10, dtype="boolean"),
        "fecha": pd.date_range("2024-01-01", periods=40, freq="D"),
        "x": np.arange(40.0),
    })
    pre = build_preprocessor(infer_feature_types(df))
    out = pre.fit_transform(df)
    assert out.shape == (40, 1 + 5 + 1)
    assert not np.isnan(out).any()


@pytest.mark.parametrize("impute", ["mean", "median", "most_frequent", "knn"])
@pytest.mark.parametrize("scale", ["standard", "minmax", "robust", None])
def test_numeric_options(impute, scale):
    df = make_classification_df(n=60, with_nulls=True, with_cats=False)
    X = df.drop(columns="y")
    out = build_preprocessor(infer_feature_types(X), impute_strategy=impute,
                             scale_method=scale).fit_transform(X)
    assert not np.isnan(out).any()


def test_sparse_and_ordinal():
    df = make_classification_df(n=60)
    X = df.drop(columns="y")
    types = infer_feature_types(X)
    assert hasattr(build_preprocessor(types, sparse=True).fit_transform(X), "toarray")
    ordinal = build_preprocessor(types, encode_method="ordinal").fit_transform(X)
    assert ordinal.shape[1] == X.shape[1]


def test_invalid_options():
    with pytest.raises(AutoMLError):
        build_preprocessor({"numeric": ["a"]}, impute_strategy="zero")
    with pytest.raises(AutoMLError):
        build_preprocessor({})
    with pytest.raises(AutoMLError):
        build_feature_selector("pca", "classification")


def test_kbest_clips_k():
    df = make_classification_df(n=80, with_cats=False)
    X, y = df.drop(columns="y"), df["y"]
    pipe = make_model_pipeline(build_preprocessor(infer_feature_types(X)),
                               LogisticRegression(),
                               selector=build_feature_selector("kbest", "classification", k=50))
    pipe.fit(X, y)
    assert len(output_feature_names(pipe)) == 4


# ---------------------------------------------------------------- métricas
def test_safe_mape_near_zero():
    assert safe_mape([100, 200], [110, 180]) == pytest.approx(10.0)
    with pytest.warns(UserWarning):
        assert np.isnan(safe_mape([0, 0, 0, 1], [1, 1, 1, 1]))
    # un solo cero en 100 valores se excluye
    y = np.r_[0.0, np.full(99, 10.0)]
    assert safe_mape(y, y + 1) == pytest.approx(10.0)


def test_pr_auc_binary_and_multiclass():
    y = np.array(["a", "b", "a", "b"])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    assert pr_auc_score(y, proba, labels=["a", "b"]) == pytest.approx(1.0)
    y3 = np.array([0, 1, 2, 0, 1, 2])
    p3 = np.eye(3)[y3] * 0.8 + 0.2 / 3
    assert pr_auc_score(y3, p3, labels=[0, 1, 2]) == pytest.approx(1.0)


def test_metric_aliases():
    assert resolve_metric("rmse", "regression").key == "neg_root_mean_squared_error"
    assert resolve_metric("rmse", "regression").display == "rmse"
    assert not resolve_metric("mae", "regression").greater_is_better
    assert resolve_metric("roc_auc", "classification", labels=[0, 1, 2]).key == \
        "roc_auc_ovr_weighted"
    specs = resolve_metrics("f1", ["accuracy", "f1_weighted"], "classification")
    assert [s.key for s in specs] == ["f1_weighted", "accuracy"]
    with pytest.raises(AutoMLError):
        resolve_metric("no_existe", "regression")


# ---------------------------------------------------------------- validación
def test_cv_strategies():
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(8), 5)
    assert make_cv(5, "auto", "classification", y)[1] == "stratified"
    assert make_cv(5, "auto", "regression", y)[1] == "kfold"
    assert make_cv(5, "auto", "classification", y, groups=groups)[1] == "stratified_group"
    assert make_cv(5, "auto", "regression", y, is_temporal=True)[1] == "timeseries"
    splitter, name = make_cv(3, "repeated_stratified", "classification", y, n_repeats=2)
    assert splitter.get_n_splits() == 6
    with pytest.raises(AutoMLError):
        make_cv(5, "group", "classification", y)


def test_cv_reduces_folds_for_rare_classes():
    y = np.array([0] * 30 + [1] * 3)
    with pytest.warns(UserWarning, match="cv se reduce"):
        splitter, _ = make_cv(5, "stratified", "classification", y)
    assert splitter.get_n_splits() == 3


def test_group_cv_never_mixes_groups():
    y = np.arange(40) % 2
    groups = np.repeat(np.arange(8), 5)
    splitter, _ = make_cv(4, "auto", "classification", y, groups=groups)
    for train, valid in splitter.split(np.zeros(40), y, groups):
        assert not set(groups[train]) & set(groups[valid])


def test_timeseries_cv_never_uses_future():
    splitter, _ = make_cv(4, "timeseries", "regression", np.arange(50.0), is_temporal=True)
    for train, valid in splitter.split(np.zeros(50)):
        assert train.max() < valid.min()


def test_holdout_split_variants():
    y = np.array([0, 1] * 50)
    train, test, desc = holdout_split(100, 0.2, "classification", y, random_state=0)
    assert len(test) == 20 and desc == "estratificado"
    train, test, desc = holdout_split(100, 0.2, "regression", is_temporal=True)
    assert test.min() == 80 and train.max() == 79
    groups = np.repeat(np.arange(20), 5)
    train, test, _ = holdout_split(100, 0.2, "regression", groups=groups, random_state=0)
    assert not set(groups[train]) & set(groups[test])
