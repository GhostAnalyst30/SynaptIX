"""Tuning, ensembles, selección de modelos, presets, SHAP y validación de parámetros."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

import synaptix as sx
from synaptix.automl import AutoMLError
from synaptix.automl.ensemble import BlendingClassifier, BlendingRegressor, build_ensemble
from synaptix.automl.selection.registry import MODE_CANDIDATES, select_candidates

from .conftest import FAST, make_classification_df, make_regression_df


# ---------------------------------------------------------------- tuning
@pytest.mark.parametrize("strategy", ["grid", "random"])
def test_tuning_strategies(clf_df, strategy):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], tuning=strategy, n_iter=5).fit(clf_df, "y")
    outcome = pipe.tuning_results_["logistic"]
    assert outcome.strategy == strategy
    assert outcome.n_evaluated >= 1
    assert set(pipe.best_params_) == {"C"}
    assert pipe.cv_score_["source"] == f"tuning:{strategy}"


def test_halving_used_with_enough_rows_and_falls_back_otherwise():
    big = make_classification_df(n=1500, with_cats=False)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], tuning="halving", n_iter=9).fit(big, "y")
    assert pipe.tuning_results_["logistic"].strategy == "halving"

    small = make_classification_df(n=200)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], tuning="halving", n_iter=4).fit(small, "y")
    outcome = pipe.tuning_results_["logistic"]
    assert outcome.strategy == "random"
    assert "halving" in outcome.note.lower()


@pytest.mark.parametrize("strategy", ["optuna", "bayesian"])
def test_optuna_tuning_is_reproducible(clf_df, strategy):
    pytest.importorskip("optuna")
    kwargs = dict(**FAST, models=["random_forest"], tuning=strategy, n_iter=6, random_state=3)
    a = sx.AutoPipeline(**kwargs).fit(clf_df, "y")
    b = sx.AutoPipeline(**kwargs).fit(clf_df, "y")
    outcome = a.tuning_results_["random_forest"]
    assert outcome.strategy == "optuna"
    assert outcome.n_evaluated == 6
    assert a.tuning_results_["random_forest"].best_params == \
        b.tuning_results_["random_forest"].best_params


def test_custom_grid_and_distributions(clf_df, reg_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"],
                           param_grids={"logistic": {"C": [0.37]}}).fit(clf_df, "y")
    assert pipe.best_params_ == {"C": 0.37}

    pipe = sx.AutoPipeline(**FAST, models=["ridge"], tuning="random", n_iter=5,
                           param_distributions={"ridge": {"alpha": ("float", 2.0, 3.0, False)}}
                           ).fit(reg_df, "y")
    trials = pipe.tuning_results_["ridge"].trials
    assert trials["params"].map(lambda p: 2.0 <= p["alpha"] <= 3.0).all()
    # Si la búsqueda no mejora los defaults en CV, se conservan los defaults.
    if pipe.cv_score_["source"] == "default":
        assert pipe.best_params_ is None
    else:
        assert 2.0 <= pipe.best_params_["alpha"] <= 3.0


def test_time_budget_skips_models_but_keeps_a_valid_winner(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic", "random_forest", "extra_trees", "knn"],
                           time_budget=1e-6).fit(clf_df, "y")
    estados = pipe.ranking_.set_index("modelo")["estado"]
    assert estados["logistic"] == "ok"
    assert (estados.drop("logistic") == "skipped").all()
    assert pipe.best_name_ == "logistic"
    assert len(pipe.predict(clf_df.head(5))) == 5


def test_tune_false_keeps_defaults(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], tune=False).fit(clf_df, "y")
    assert pipe.best_params_ is None
    assert pipe.tuning_results_ == {}


# ---------------------------------------------------------------- ensembles
@pytest.mark.parametrize("kind", ["voting", "stacking", "blending"])
def test_ensembles_are_evaluated_with_same_cv(clf_df, kind):
    pipe = sx.AutoPipeline(**FAST, models=["logistic", "random_forest", "naive_bayes"],
                           tuning=None, ensemble=kind).fit(clf_df, "y")
    info = pipe.ensemble_result_
    assert info["status"] == "ok"
    assert len(info["members"]) == 3
    row = pipe.ranking_.set_index("modelo").loc[f"ensemble_{kind}"]
    assert row["estado"] == "ok"
    best_single = pipe.ranking_[~pipe.ranking_["modelo"].str.startswith("ensemble")]
    best_single_score = best_single["accuracy"].max()
    assert info["selected"] == (row["accuracy"] > best_single_score)
    assert pipe.best_name_ == (f"ensemble_{kind}" if info["selected"] else info["best_single"])
    assert len(pipe.predict(clf_df.head(4))) == 4


@pytest.mark.parametrize("kind", ["voting", "stacking", "blending"])
def test_regression_ensembles(reg_df, kind):
    pipe = sx.AutoPipeline(**FAST, models=["ridge", "decision_tree"], tuning=None,
                           ensemble=kind).fit(reg_df, "y")
    assert pipe.ensemble_result_["status"] == "ok"


def test_stacking_is_skipped_for_time_series_but_blending_is_chronological():
    n = 200
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"t": np.arange(n), "x": rng.normal(size=n)})
    df["y"] = 2 * df["x"] + rng.normal(0, 0.1, n)
    stack = sx.AutoPipeline(**FAST, models=["ridge", "decision_tree"], tuning=None,
                            ensemble="stacking").fit(df, "y", time_column="t")
    assert any("Stacking omitido" in w for w in stack.warnings_)
    blend = sx.AutoPipeline(**FAST, models=["ridge", "decision_tree"], tuning=None,
                            ensemble="blending").fit(df, "y", time_column="t")
    assert blend.ensemble_result_["status"] == "ok"


def test_blending_members_never_see_the_meta_holdout():
    X, y = make_classification_df(n=200, with_cats=False).pipe(
        lambda d: (d[["a", "b", "c", "d"]].to_numpy(), d["y"].to_numpy()))
    model = BlendingClassifier([("lr", LogisticRegression()), ("dt", DecisionTreeClassifier())],
                               random_state=0).fit(X, y)
    for _, est in model.estimators_:
        assert est.n_features_in_ == 4
    assert model.predict_proba(X[:5]).shape == (5, 2)

    chrono = BlendingRegressor([("a", Ridge()), ("b", Ridge(alpha=10))], shuffle=False)
    train_idx, hold_idx = chrono._split(X, np.arange(200.0), stratify=False)
    assert train_idx.max() < hold_idx.min()


def test_ensemble_needs_two_members():
    with pytest.raises(AutoMLError, match="al menos 2"):
        build_ensemble("voting", "classification", [("lr", LogisticRegression())])


# ---------------------------------------------------------------- selección de modelos
def test_models_by_alias_tuple_and_exclusion():
    cands, _ = select_candidates("classification", models=[
        "logistic_regression", "svc", ("random_forest", {"n_estimators": 7})],
        exclude=["svm"])
    assert [c.name for c in cands] == ["logistic", "random_forest"]
    assert cands[1].estimator.n_estimators == 7
    assert cands[1].estimator.n_jobs == 1


def test_mode_presets_and_exclusion():
    for mode in ("fast", "balanced", "exhaustive"):
        cands, _ = select_candidates("regression", mode=mode)
        assert [c.name for c in cands] == MODE_CANDIDATES[mode]["regression"]
    cands, _ = select_candidates("regression", mode="fast", exclude=["linear", "ridge"])
    assert "linear" not in [c.name for c in cands]


def test_expensive_models_are_skipped_on_large_data_with_note():
    cands, notes = select_candidates("classification", mode="balanced", n_samples=50_000)
    assert "svm" not in [c.name for c in cands]
    assert any("svm" in n for n in notes)


@pytest.mark.parametrize("bad", [["no_existe"], [("random_forest", {"foo": 1})]])
def test_invalid_models_raise(clf_df, bad):
    with pytest.raises(AutoMLError):
        sx.AutoPipeline(**FAST, models=bad).fit(clf_df, "y")


def test_all_models_excluded_raises(clf_df):
    with pytest.raises(AutoMLError, match="No quedan"):
        sx.AutoPipeline(**FAST, models=["logistic"], exclude_models=["logistic"]).fit(clf_df, "y")


@pytest.mark.parametrize("kwargs", [
    {"task": "clustering"}, {"tuning": "genetic"}, {"ensemble": "bagging"},
    {"mode": "turbo"}, {"engine": "spark"},
])
def test_invalid_parameters_raise(clf_df, kwargs):
    with pytest.raises(AutoMLError):
        sx.AutoPipeline(**{**FAST, **kwargs}).fit(clf_df, "y")


def test_explicit_task_overrides_detection():
    df = make_regression_df()
    df["y"] = np.round(df["y"]).astype(int)
    pipe = sx.AutoPipeline(**FAST, task="regression", models=["ridge"]).fit(df, "y")
    assert pipe.task_ == "regression"


# ---------------------------------------------------------------- SHAP
def test_shap_values_when_installed(reg_df):
    pytest.importorskip("shap")
    pipe = sx.AutoPipeline(**FAST, models=["ridge"]).fit(reg_df, "y")
    out = pipe.explain_shap(reg_df.head(30), max_samples=30)
    assert out["values"].shape == (30, len(out["feature_names"]))
    reconstructed = out["values"].sum(axis=1) + np.asarray(out["base_values"])
    assert np.allclose(reconstructed, pipe.pipeline_.predict(reg_df.head(30).drop(columns="y")),
                       atol=1e-6)
