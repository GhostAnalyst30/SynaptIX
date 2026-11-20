"""Pruebas de la API de acceso a los datos usados por AutoPipeline."""

import numpy as np
import pandas as pd
import pytest

import synaptix as sx

from .conftest import FAST, make_classification_df

LOCAL = dict(FAST, models=["logistic"], tune=False)


def _index_union(*frames):
    values = set()
    for frame in frames:
        values.update(frame.index.tolist())
    return values


def test_access_before_fit_raises():
    pipe = sx.AutoPipeline(**FAST)
    assert pipe.data_train_ is None
    assert pipe.data_test_ is None
    with pytest.raises(RuntimeError):
        pipe.data_
    with pytest.raises(RuntimeError):
        pipe.best_data_
    with pytest.raises(RuntimeError):
        pipe.data


def test_train_test_split_and_target_alignment():
    df = make_classification_df(n=200)
    pipe = sx.AutoPipeline(**LOCAL).fit(df, target="y")

    assert len(pipe.X_train_) == pipe.n_train_
    assert len(pipe.X_test_) == pipe.n_test_
    assert pipe.n_train_ + pipe.n_test_ == len(df)

    expected_columns = pipe.feature_columns_ + [pipe.target_]
    assert list(pipe.data_train_.columns) == expected_columns
    assert list(pipe.data_test_.columns) == expected_columns
    assert list(pipe.X_train_.columns) == pipe.feature_columns_
    assert list(pipe.X_test_.columns) == pipe.feature_columns_

    assert pipe.y_train_.index.equals(pipe.X_train_.index)
    assert pipe.y_test_.index.equals(pipe.X_test_.index)
    np.testing.assert_array_equal(
        pipe.y_train_.to_numpy(), df.loc[pipe.X_train_.index, "y"].to_numpy()
    )
    np.testing.assert_array_equal(
        pipe.y_test_.to_numpy(), df.loc[pipe.X_test_.index, "y"].to_numpy()
    )

    pd.testing.assert_frame_equal(
        pipe.data_train_, pd.concat([pipe.X_train_, pipe.y_train_], axis=1)
    )
    pd.testing.assert_frame_equal(
        pipe.data_test_, pd.concat([pipe.X_test_, pipe.y_test_], axis=1)
    )


def test_original_indices_are_preserved_and_disjoint():
    df = make_classification_df(n=180)
    df.index = df.index + 10_000
    pipe = sx.AutoPipeline(**LOCAL).fit(df, target="y")

    train_index = set(pipe.X_train_.index)
    test_index = set(pipe.X_test_.index)
    assert not train_index & test_index
    assert train_index | test_index == set(df.index)
    assert set(pipe.data_train_.index) == train_index
    assert set(pipe.data_test_.index) == test_index
    # Las features son las originales, no una matriz one-hot/escalada.
    assert not any("__" in column for column in pipe.X_train_.columns)
    pd.testing.assert_frame_equal(
        pipe.X_train_,
        df.loc[pipe.X_train_.index, pipe.feature_columns_],
    )


def test_data_and_best_data_aliases():
    df = make_classification_df(n=160)
    pipe = sx.AutoPipeline(**LOCAL).fit(df, target="y")

    train, test = pipe.data_
    alias_train, alias_test = pipe.data
    pd.testing.assert_frame_equal(train, alias_train)
    pd.testing.assert_frame_equal(test, alias_test)

    best_train, best_test = pipe.best_data_
    expected_best_columns = pipe.best_features_ + [pipe.target_]
    assert list(best_train.columns) == expected_best_columns
    assert list(best_test.columns) == expected_best_columns
    pd.testing.assert_frame_equal(
        best_train, pipe.data_train_[expected_best_columns]
    )
    pd.testing.assert_frame_equal(
        best_test, pipe.data_test_[expected_best_columns]
    )


def test_test_size_none_returns_empty_frames_with_columns():
    df = make_classification_df(n=150)
    pipe = sx.AutoPipeline(**LOCAL, test_size=None).fit(df, target="y")

    expected_columns = pipe.feature_columns_ + [pipe.target_]
    expected_best_columns = pipe.best_features_ + [pipe.target_]
    assert pipe.n_test_ == 0
    assert len(pipe.X_test_) == 0
    assert len(pipe.y_test_) == 0
    assert len(pipe.data_test_) == 0
    assert len(pipe.best_data_test_) == 0
    assert list(pipe.X_test_.columns) == pipe.feature_columns_
    assert list(pipe.data_test_.columns) == expected_columns
    assert list(pipe.best_data_test_.columns) == expected_best_columns
    assert pipe.data_test_.index.empty


def test_feature_search_uses_best_subset():
    df = make_classification_df(n=180)
    pipe = sx.AutoPipeline(
        **LOCAL,
        feature_search="forward",
        feature_search_max_features=2,
    ).fit(df, target="y")

    assert pipe.best_features_
    assert set(pipe.best_features_) <= set(pipe.feature_columns_)
    assert list(pipe.data_train_.columns) == pipe.feature_columns_ + [pipe.target_]
    expected_best_columns = pipe.best_features_ + [pipe.target_]
    assert list(pipe.best_data_train_.columns) == expected_best_columns
    assert list(pipe.best_data_test_.columns) == expected_best_columns
    pd.testing.assert_frame_equal(
        pipe.best_data_train_, pipe.data_train_[expected_best_columns]
    )
    pd.testing.assert_frame_equal(
        pipe.best_data_test_, pipe.data_test_[expected_best_columns]
    )


def test_associations_are_included_in_captured_data():
    df = make_classification_df(n=160)
    pipe = sx.AutoPipeline(
        **LOCAL, associations={"a_x_b": "a * b"}
    ).fit(df, target="y")

    assert "a_x_b" in pipe.feature_columns_
    assert "a_x_b" in pipe.data_train_.columns
    assert "a_x_b" in pipe.best_data_train_.columns
    expected = (
        df.loc[pipe.data_train_.index, "a"]
        * df.loc[pipe.data_train_.index, "b"]
    ).to_numpy()
    np.testing.assert_allclose(
        pipe.data_train_["a_x_b"].to_numpy(), expected
    )
    np.testing.assert_allclose(
        pipe.best_data_train_["a_x_b"].to_numpy(), expected
    )


def test_warning_strategy_row_changes_are_respected():
    df = make_classification_df(n=120)
    duplicated = pd.concat([df, df.iloc[:10]], ignore_index=True)
    assert duplicated.duplicated().sum() == 10

    pipe = sx.AutoPipeline(
        **LOCAL, warning_strategy="safe"
    ).fit(duplicated, target="y")

    assert pipe.n_train_ + pipe.n_test_ == len(df)
    stored_index = _index_union(pipe.data_train_, pipe.data_test_)
    assert stored_index == set(range(len(df)))
    np.testing.assert_array_equal(
        pipe.data_train_.loc[pipe.data_train_.index, "y"].to_numpy(),
        df.loc[pipe.data_train_.index, "y"].to_numpy(),
    )
    np.testing.assert_array_equal(
        pipe.data_test_.loc[pipe.data_test_.index, "y"].to_numpy(),
        df.loc[pipe.data_test_.index, "y"].to_numpy(),
    )


def test_pandas_polars_and_numpy_inputs():
    df = make_classification_df(n=150)
    pandas_pipe = sx.AutoPipeline(**LOCAL).fit(df, target="y")
    assert len(pandas_pipe.data_train_) + len(pandas_pipe.data_test_) == len(df)

    pl = pytest.importorskip("polars")
    polars_pipe = sx.AutoPipeline(**LOCAL).fit(pl.from_pandas(df), target="y")
    assert len(polars_pipe.data_train_) + len(polars_pipe.data_test_) == len(df)
    assert list(polars_pipe.data_train_.columns) == (
        polars_pipe.feature_columns_ + ["y"]
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    numpy_pipe = sx.AutoPipeline(**LOCAL).fit(X, y)
    assert numpy_pipe.target_ == "target"
    assert list(numpy_pipe.data_train_.columns) == ["x0", "x1", "x2", "target"]
    np.testing.assert_allclose(
        numpy_pipe.X_train_.to_numpy(),
        X[numpy_pipe.X_train_.index.to_numpy()],
    )
    np.testing.assert_array_equal(
        numpy_pipe.y_train_.to_numpy(),
        y[numpy_pipe.y_train_.index.to_numpy()],
    )


def test_refit_replaces_previous_data():
    pipe = sx.AutoPipeline(**LOCAL)
    first = make_classification_df(n=140, seed=1)
    pipe.fit(first, target="y")
    first_train_index = set(pipe.X_train_.index)

    second = make_classification_df(n=220, seed=2)
    second.index = second.index + 5_000
    pipe.fit(second, target="y")

    assert pipe.n_train_ + pipe.n_test_ == len(second)
    assert _index_union(pipe.data_train_, pipe.data_test_) == set(second.index)
    assert set(pipe.X_train_.index) | set(pipe.X_test_.index) == set(second.index)
    assert set(pipe.X_train_.index) & set(pipe.X_test_.index) == set()
    assert not first_train_index & set(pipe.X_train_.index)
    assert pipe.data_train_.shape[0] + pipe.data_test_.shape[0] == len(second)


def test_ignore_columns_are_excluded_from_captured_data():
    df = make_classification_df(n=150)
    pipe = sx.AutoPipeline(**LOCAL, ignore_columns=["c"]).fit(df, target="y")

    assert "c" not in pipe.feature_columns_
    for frame in (pipe.X_train_, pipe.X_test_, pipe.data_train_, pipe.data_test_):
        assert "c" not in frame.columns


def test_warning_strategy_column_corrections_are_reflected():
    df = make_classification_df(n=150, with_nulls=True)
    pipe = sx.AutoPipeline(**LOCAL, warning_strategy="safe").fit(df, target="y")

    assert pipe.data_train_["a"].isna().sum() == 0
    assert pipe.data_test_["a"].isna().sum() == 0
    assert pipe.best_data_train_["a"].isna().sum() == 0
    assert pipe.best_data_test_["a"].isna().sum() == 0


def test_no_leakage_between_train_and_test():
    df = make_classification_df(n=200)
    pipe = sx.AutoPipeline(**LOCAL).fit(df, target="y")

    train_index = set(pipe.data_train_.index)
    test_index = set(pipe.data_test_.index)
    assert not train_index & test_index
    assert train_index | test_index == set(df.index)

    for frame in (pipe.data_train_, pipe.data_test_):
        np.testing.assert_array_equal(
            frame["y"].to_numpy(),
            df.loc[frame.index, "y"].to_numpy(),
        )
    # best_data_ no añade columnas ni filas distintas de las de data_.
    assert set(pipe.best_data_train_.index) == train_index
    assert set(pipe.best_data_test_.index) == test_index
    assert set(pipe.best_data_train_.columns) <= set(pipe.data_train_.columns)
