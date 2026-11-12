import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

import synaptix as sx
from synaptix.automl import AllModelsFailedError, AutoMLError, ExperimentRecord

from .conftest import FAST, make_classification_df, make_regression_df


class Broken(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        raise RuntimeError("fallo intencional")

    def predict(self, X):  # pragma: no cover
        return np.zeros(len(X))


# ---------------------------------------------------------------- casos básicos
@pytest.mark.parametrize("with_nulls", [False, True])
def test_binary_classification(with_nulls):
    df = make_classification_df(n=250, with_nulls=with_nulls)
    pipe = sx.AutoPipeline(**FAST).fit(df, target="y")
    assert pipe.task_ == "classification"
    assert pipe.results_["accuracy"] > 0.75
    assert {"precision", "recall", "f1", "roc_auc", "pr_auc", "mcc",
            "log_loss", "balanced_accuracy"} <= set(pipe.results_)
    assert set(pipe.predict(df.head(10))) <= {"si", "no"}
    assert pipe.predict_proba(df.head(3)).shape == (3, 2)


def test_multiclass_classification():
    df = make_classification_df(n=300, n_classes=3)
    pipe = sx.AutoPipeline(**FAST, scoring="f1_weighted").fit(df, target="y")
    assert pipe.ranking_.columns[2] == "f1_weighted"
    assert pipe.results_["f1"] > 0.6
    assert pipe.predict_proba(df.head(4)).shape == (4, 3)


@pytest.mark.parametrize("with_nulls", [False, True])
def test_regression(with_nulls):
    df = make_regression_df(with_nulls=with_nulls)
    pipe = sx.AutoPipeline(**FAST, scoring="rmse").fit(df, target="y")
    assert pipe.task_ == "regression"
    assert pipe.results_["R2"] > 0.8
    assert {"MAE", "MSE", "RMSE", "R2", "MAPE", "MedAE"} <= set(pipe.results_)
    # rmse se ordena ascendente
    ok = pipe.ranking_[pipe.ranking_["estado"] == "ok"]["rmse"]
    assert ok.is_monotonic_increasing


def test_small_dataset():
    df = make_classification_df(n=30)
    pipe = sx.AutoPipeline(**FAST).fit(df, target="y")
    assert pipe.n_test_ == 6


def test_imbalanced_dataset_warns_about_accuracy():
    df = make_classification_df(n=400)
    keep = (df["y"] == "si") | (np.arange(len(df)) % 8 == 0)
    df = df[keep]
    pipe = sx.AutoPipeline(**FAST).fit(df, target="y")
    assert any("desbalanceadas" in w for w in pipe.warnings_)
    assert any(w.code == "class_imbalance" for w in pipe.profile_.warnings)


def test_constant_columns_are_kept_unless_requested():
    df = make_classification_df(n=150).assign(const=5.0, const_cat="x")
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(df, target="y")
    assert "const" in pipe.feature_columns_
    assert any(w.code == "constant" for w in pipe.profile_.warnings)
    dropped = sx.AutoPipeline(**FAST, models=["logistic"], drop_constant=True).fit(df, "y")
    names = dropped.best_model_.feature_names_
    assert "const" not in names


def test_single_class_raises():
    df = pd.DataFrame({"a": range(30), "y": ["x"] * 30})
    with pytest.raises(AutoMLError, match="una sola clase"):
        sx.AutoPipeline(**FAST).fit(df, target="y")


def test_target_with_nulls_is_dropped_and_reported():
    df = make_classification_df(n=200)
    df.loc[:9, "y"] = None
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(df, target="y")
    assert pipe.n_train_ + pipe.n_test_ == 190
    assert any(w.code == "target_missing" for w in pipe.profile_.warnings)


def test_unknown_categories_at_inference(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(clf_df, target="y")
    nuevo = clf_df.drop(columns="y").head(5).assign(color="morado")
    assert len(pipe.predict(nuevo)) == 5


def test_missing_columns_at_inference_raise(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(clf_df, target="y")
    with pytest.raises(AutoMLError, match="Faltan columnas"):
        pipe.predict(clf_df.drop(columns=["y", "a"]))


# ---------------------------------------------------------------- errores de modelos
def test_failing_model_never_wins(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=[Broken(), "logistic"]).fit(clf_df, target="y")
    row = pipe.ranking_.set_index("modelo").loc["Broken"]
    assert row["estado"] == "failed"
    assert "fallo intencional" in row["error"]
    assert pipe.best_name_ == "logistic"
    assert pipe.ranking_.iloc[-1]["modelo"] == "Broken"
    assert any("Broken" in w for w in pipe.warnings_)


def test_all_models_fail(clf_df):
    with pytest.raises(AllModelsFailedError) as info:
        sx.AutoPipeline(**FAST, models=[Broken()]).fit(clf_df, target="y")
    assert "Broken" in info.value.errors


def test_all_nan_column_does_not_crash(clf_df):
    df = clf_df.assign(a=np.nan)
    pipe = sx.AutoPipeline(**FAST, models=["naive_bayes"]).fit(df, target="y")
    assert any(w.code == "all_missing" for w in pipe.profile_.warnings)


# ---------------------------------------------------------------- entradas
def test_numpy_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(X, y)
    assert pipe.task_ == "classification"
    assert pipe.predict(X[:5]).shape == (5,)


def test_polars_input():
    pl = pytest.importorskip("polars")
    df = make_classification_df(n=200)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(pl.from_pandas(df), target="y")
    assert pipe._engine_used == "polars"
    assert pipe.profile_.engine == "polars"
    preds_pl = pipe.predict(pl.from_pandas(df.head(10)))
    preds_pd = pipe.predict(df.head(10))
    assert (preds_pl == preds_pd).all()


def test_engine_polars_equals_pandas_results():
    pl = pytest.importorskip("polars")
    df = make_classification_df(n=200, with_nulls=True)
    a = sx.AutoPipeline(**FAST, models=["logistic"], engine="pandas").fit(df, "y")
    b = sx.AutoPipeline(**FAST, models=["logistic"], engine="polars").fit(df, "y")
    pd.testing.assert_frame_equal(
        a.ranking_.drop(columns=["tiempo_ajuste_s", "tiempo_prediccion_s", "tiempo_s"]),
        b.ranking_.drop(columns=["tiempo_ajuste_s", "tiempo_prediccion_s", "tiempo_s"]))


def test_csv_path_input(tmp_path):
    df = make_classification_df(n=150)
    path = tmp_path / "datos.csv"
    df.to_csv(path, index=False)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(str(path), target="y")
    assert pipe.n_train_ + pipe.n_test_ == 150


# ---------------------------------------------------------------- validación estructurada
def test_time_series_validation():
    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "fecha": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x": rng.normal(size=n),
    })
    df["y"] = np.cumsum(rng.normal(size=n)) + df["x"]
    shuffled = df.sample(frac=1, random_state=0)
    pipe = sx.AutoPipeline(**FAST, models=["ridge"]).fit(shuffled, target="y",
                                                        time_column="fecha")
    assert pipe.cv_strategy_ == "timeseries"
    assert pipe.holdout_.startswith("cronológico")
    assert "fecha" not in pipe.feature_columns_
    test_dates = df.loc[pipe._eval["index"], "fecha"]
    assert test_dates.min() > df["fecha"].iloc[: pipe.n_train_].max()


def test_group_validation():
    df = make_classification_df(n=240)
    df["paciente"] = np.repeat(np.arange(40), 6)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(df, target="y", groups="paciente")
    assert pipe.cv_strategy_ == "stratified_group"
    assert "paciente" not in pipe.feature_columns_
    test_groups = set(df.loc[pipe._eval["index"], "paciente"])
    train_groups = set(df["paciente"]) - test_groups
    assert test_groups and not test_groups & train_groups


def test_custom_cv_splitter(reg_df):
    pipe = sx.AutoPipeline(**{**FAST, "cv": KFold(4)}, models=["ridge"]).fit(reg_df, "y")
    assert pipe.cv_strategy_ == "KFold"


# ---------------------------------------------------------------- reproducibilidad
def test_reproducibility(clf_df):
    kwargs = dict(**FAST, models=["random_forest", "logistic"], tuning="random", n_iter=4)
    a = sx.AutoPipeline(**kwargs, random_state=7).fit(clf_df, "y")
    b = sx.AutoPipeline(**kwargs, random_state=7).fit(clf_df, "y")
    cols = ["modelo", "estado", "accuracy", "desviacion"]
    pd.testing.assert_frame_equal(a.ranking_[cols], b.ranking_[cols])
    assert a.best_params_ == b.best_params_
    assert (a.predict(clf_df) == b.predict(clf_df)).all()


# ---------------------------------------------------------------- compatibilidad
def test_legacy_api_and_attributes(clf_df, tmp_path):
    pipe = sx.AutoPipeline(cv=3, verbose=False)
    pipe.fit(clf_df, target="y")
    X_clean = pipe.cleaner_.transform(clf_df.drop(columns="y"))
    assert (pipe.best_model_.predict(X_clean) == pipe.predict(clf_df)).all()
    assert pipe.report(plot=False) == pipe.results_
    path = tmp_path / "pipe.pkl"
    pipe.save(str(path))
    loaded = sx.AutoPipeline.load(str(path))
    assert (loaded.predict(clf_df) == pipe.predict(clf_df)).all()
    from synaptix.pipeline import DEFAULT_GRIDS, MODEL_REGISTRY, detect_task
    from synaptix.pipeline.auto import AutoPipeline as Legacy

    assert Legacy is sx.AutoPipeline
    assert "random_forest" in MODEL_REGISTRY["classification"]
    assert DEFAULT_GRIDS["logistic"]
    assert detect_task(clf_df["y"]) == "classification"


def test_pickle_has_no_raw_data(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], memory_optimization=True).fit(clf_df, "y")
    blob = pickle.dumps(pipe)
    for value in clf_df["a"].head(20):
        assert np.float64(value).tobytes() not in blob
    assert not any(isinstance(v, pd.DataFrame) and len(v) == pipe.n_test_
                   for v in vars(pipe).values())


def test_memory_optimization_uses_sparse(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], memory_optimization=True).fit(clf_df, "y")
    out = pipe.pipeline_[:-1].transform(clf_df.drop(columns="y"))
    assert hasattr(out, "toarray")


def test_parallel_matches_sequential(clf_df):
    a = sx.AutoPipeline(**FAST, models=["logistic", "random_forest"], n_jobs=1).fit(clf_df, "y")
    b = sx.AutoPipeline(**FAST, models=["logistic", "random_forest"], n_jobs=2).fit(clf_df, "y")
    assert np.allclose(a.ranking_["accuracy"], b.ranking_["accuracy"])


def test_cache_option(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], cache=True, tuning="grid").fit(clf_df, "y")
    assert pipe.pipeline_.memory is None


# ---------------------------------------------------------------- reportes y tracking
def test_report_exports(clf_df, tmp_path):
    pipe = sx.AutoPipeline(mode="fast", cv=3, verbose=False).fit(clf_df, "y")
    html_path, json_path = tmp_path / "r.html", tmp_path / "r.json"
    pipe.report(plot=True, save=str(html_path))
    content = html_path.read_text(encoding="utf-8")
    assert "Métricas de prueba" in content and "data:image/png;base64" in content
    pipe.report(plot=False, save=str(json_path))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment"]["selected_model"] == pipe.best_name_
    sections = list(payload["report"])
    assert sections[0] == "1_dataset" and sections[-1] == "12_advertencias"


def test_experiment_record(clf_df, tmp_path):
    pipe = sx.AutoPipeline(**FAST, models=["logistic", "knn"]).fit(clf_df, "y")
    record = pipe.experiment_
    assert record.random_state == 42
    assert record.environment["packages"]["scikit-learn"]
    assert record.environment["packages"]["synaptix"] == sx.__version__
    assert record.data["n_rows"] == len(clf_df)
    assert len(record.ranking) == 2
    path = tmp_path / "exp.json"
    record.to_json(str(path))
    restored = ExperimentRecord.from_json(str(path))
    assert restored.experiment_id == record.experiment_id
    csv = record.to_csv()
    assert csv.splitlines()[0].startswith("experiment_id,modelo")


def test_explainability(reg_df):
    pipe = sx.AutoPipeline(mode="fast", cv=3, verbose=False, models=["ridge"]).fit(reg_df, "y")
    top = pipe.feature_importances_.iloc[0]["feature"]
    assert top == "a"
    assert pipe.coefficients_ is not None
    assert "residuals" in pipe.diagnostics_
    with pytest.raises(ImportError, match="shap"):
        try:
            import shap  # noqa: F401
            pytest.skip("shap instalado")
        except ImportError:
            pipe.explain_shap(reg_df.head(20))


def test_classification_diagnostics(clf_df):
    pipe = sx.AutoPipeline(mode="fast", cv=3, verbose=False, models=["logistic"]).fit(clf_df, "y")
    cm = pipe.diagnostics_["confusion_matrix"]
    assert cm.to_numpy().sum() == pipe.n_test_


def test_declarative_pipeline_has_no_leakage():
    df = sx.load_dataset("penguins")
    from synaptix.pipeline import Pipeline

    pipe = Pipeline(["clean", "model", "evaluate"], cv=3, verbose=False).fit(df, "species")
    n_train = len(pipe._X_train)
    assert pipe.cleaner_.scaler_.scaler.n_samples_seen_ == n_train
    assert pipe.results_["accuracy"] > 0.9
