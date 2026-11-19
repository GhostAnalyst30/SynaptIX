"""Feature engineering robusto, selección consultable, explain() y calidad.

Cubre los cuatro huecos que quedaban tras la auditoría:

1. ``fe_handle_div_zero`` tenía un ``except ZeroDivisionError`` inalcanzable
   (NumPy devuelve inf/nan en lugar de lanzar), así que ``"zero"`` y
   ``"error"`` no tenían efecto.
2. No existía API pública para saber qué features se conservaron.
3. No existía ``explain()`` unificado.
4. La calidad de datos estimaba sus parámetros con todos los datos, antes
   del holdout: la mediana de imputación veía el conjunto de prueba.
"""

import numpy as np
import pandas as pd
import pytest

import synaptix as sx
from synaptix.automl.data_quality import analyze_and_solve_data_quality
from synaptix.automl.feature_engineering import FeatureEngineeringTransformer

from .conftest import FAST, make_classification_df


# ------------------------------------------------------- división por cero
@pytest.mark.parametrize("policy,expected", [
    ("nan", np.nan),
    ("inf", np.inf),
    ("zero", 0.0),
])
def test_div_zero_policies_take_effect(policy, expected):
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [1.0, 1.0, 1.0]})
    transformer = FeatureEngineeringTransformer(
        associations={"ratio": "z / x"}, handle_division_by_zero=policy)
    out = transformer.fit_transform(df)
    value = out["ratio"].iloc[0]
    if np.isnan(expected):
        assert np.isnan(value)
    else:
        assert value == expected


def test_div_zero_error_policy_raises():
    df = pd.DataFrame({"x": [0.0, 1.0]})
    transformer = FeatureEngineeringTransformer(
        associations={"ratio": "1 / x"}, handle_division_by_zero="error")
    with pytest.raises(Exception, match="División por cero"):
        transformer.fit_transform(df)


def test_chained_expression_dividing_by_derived_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [1.0, 2.0, 2.0]})
    transformer = FeatureEngineeringTransformer(
        associations={"s": "a + b", "inv": "a / s"},
        handle_division_by_zero="error")
    out = transformer.fit_transform(df)
    assert np.allclose(out["inv"], [0.5, 0.5, 2 / 3])


def test_div_zero_zero_policy_survives_predict():
    df = make_classification_df(n=120, with_cats=False, with_nulls=False)
    df["x"] = np.arange(len(df), dtype=float) + 1.0
    df.loc[0, "x"] = 0.0
    features = df.drop(columns="y")
    pipe = sx.AutoPipeline(
        **FAST, models=["logistic"],
        associations={"inv": "1 / x"}, fe_handle_div_zero="zero",
    ).fit(df, "y")
    out = pipe.fe_transformer_.transform(features.head(3))
    assert out["inv"].iloc[0] == 0.0
    assert len(pipe.predict(features.head(3))) == 3


# ------------------------------------------------- selección de features
def test_selected_and_removed_features_are_exposed(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(clf_df, "y")
    info = pipe.feature_selection_info_
    assert pipe.selected_features_
    assert set(pipe.selected_features_).isdisjoint(pipe.removed_features_)
    assert info["n_selected"] == len(pipe.selected_features_)
    assert info["n_prepared"] >= info["n_selected"]
    assert pipe.feature_importance_ is not None
    assert pipe.feature_importance_source_ in {"model", "permutation_test"}


def test_kbest_records_removed_features_and_reason(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], feature_selection="kbest",
                           feature_selection_k=2).fit(clf_df, "y")
    assert 0 < len(pipe.selected_features_) <= 2
    assert pipe.removed_features_
    assert "test F" in pipe.feature_selection_info_["justification"]


def test_removed_features_are_the_complement_of_selected(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], feature_selection="variance",
                           drop_constant=True).fit(clf_df.assign(const=1.0), "y")
    info = pipe.feature_selection_info_
    assert info["n_prepared"] == len(pipe.selected_features_) + len(pipe.removed_features_)


# ------------------------------------------------------------- explain()
def test_explain_returns_global_and_diagnostics(reg_df):
    pipe = sx.AutoPipeline(**FAST, models=["ridge"]).fit(reg_df, "y")
    out = pipe.explain()
    assert out["task"] == "regression"
    assert out["model"] == pipe.best_name_
    assert out["global"]["model_native"] is not None
    assert out["global"]["coefficients"] is not None
    assert out["global"]["selected_features"] == pipe.selected_features_
    assert "residuals" in out["diagnostics"]
    assert out["calibration"] is None
    assert out["note"] and out["local"] is None


def test_explain_includes_calibration_for_classification(clf_df):
    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(clf_df, "y")
    calibration = pipe.explain()["calibration"]
    assert calibration is not None
    assert 0.0 <= calibration["brier_score"] <= 2.0
    assert calibration["bins"] is not None


def test_explain_prefers_test_permutation_when_available(reg_df):
    pipe = sx.AutoPipeline(mode="fast", cv=3, verbose=False, models=["ridge"],
                           explain=True).fit(reg_df, "y")
    out = pipe.explain()
    assert out["global"]["permutation_test"] is not None
    assert out["global"]["preferred"] == "permutation_test"


# ---------------------------------------------------- calidad de datos
def test_none_strategy_never_modifies_data():
    df = pd.DataFrame({"x": [1.0, np.nan, np.inf, 3.0], "c": ["a"] * 4})
    before = df.copy()
    fixed, report = analyze_and_solve_data_quality(df, strategy="none")
    pd.testing.assert_frame_equal(fixed, before)
    assert report.actions_taken == []
    assert report.issues


def test_data_quality_uses_reference_statistics():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 100.0, 200.0, 300.0]})
    reference = df.iloc[:3]  # mediana de train = 2.0; la global sería 51.5
    fixed, report = analyze_and_solve_data_quality(
        df, strategy="safe", reference=reference)
    assert fixed["x"].iloc[1] == 2.0
    assert any(action.action.value == "impute" for action in report.actions_taken)


def test_safe_strategy_reports_actions_in_pipeline_report():
    df = make_classification_df(n=200, with_nulls=True)
    pipe = sx.AutoPipeline(**FAST, models=["logistic"], warning_strategy="safe",
                           warning_report=True).fit(df, "y")
    content = pipe.report_dict()
    assert "13_calidad_datos" in content
    quality = content["13_calidad_datos"]
    assert quality["strategy"] == "safe"
    assert quality["summary"]["total_issues"] > 0
    assert any(action["action"] != "none" for action in quality["actions_taken"])
    assert pipe.data_quality_report_ is not None


def test_pipeline_fits_data_quality_only_with_train_rows(monkeypatch, clf_df):
    """El solver nunca debe ver las filas del holdout al estimar parámetros."""
    import synaptix.automl.pipeline as automl_pipeline

    captured = {}
    original = automl_pipeline.analyze_and_solve_data_quality

    def spy(*args, **kwargs):
        if kwargs.get("reference") is not None:
            captured["reference"] = kwargs["reference"]
            captured["n_total"] = len(kwargs["df"])
        return original(*args, **kwargs)

    monkeypatch.setattr(automl_pipeline, "analyze_and_solve_data_quality", spy)
    df = clf_df.copy()
    df.loc[df.index[:20], "a"] = np.nan
    pipe = sx.AutoPipeline(**FAST, models=["logistic"],
                           warning_strategy="safe").fit(df, "y")
    assert len(captured["reference"]) == pipe.n_train_
    assert len(captured["reference"]) < captured["n_total"]
