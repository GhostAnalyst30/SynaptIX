"""
Tests de la integración synaptix ↔ viewx ↔ statslibx.

El núcleo (payloads, objetos de resultado, entrada polars) se prueba
siempre; los tests que requieren viewx, statslibx, plotly o polars se
omiten automáticamente si el paquete no está instalado.
"""

import numpy as np
import pandas as pd
import pytest

import synaptix as sx
from synaptix.results import (
    CrossValidationResult,
    EvaluationResult,
    ModelComparisonResult,
)
from synaptix.viewx import _availability
from synaptix.viewx.adapters import to_report_data


@pytest.fixture(scope="module")
def iris_split():
    df = sx.load_dataset("iris")
    X = df.drop(columns="species")
    y = df["species"]
    return X, y


@pytest.fixture(scope="module")
def classification_result(iris_split):
    X, y = iris_split
    model = sx.supervised.RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    return model.evaluate(X, y, verbose=False)


@pytest.fixture(scope="module")
def regression_result():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=80)
    model = sx.supervised.LinearRegression()
    model.fit(X, y)
    return model.evaluate(X, y, verbose=False)


# ----------------------------------------------------------------------
# Objetos de resultado (compatibilidad con el API dict anterior)
# ----------------------------------------------------------------------


def test_evaluate_returns_dict_compatible(classification_result):
    assert isinstance(classification_result, dict)
    assert isinstance(classification_result, EvaluationResult)
    assert "accuracy" in classification_result
    assert classification_result["accuracy"] == pytest.approx(
        dict(classification_result)["accuracy"]
    )
    assert classification_result.task == "classification"
    assert classification_result.model_name == "RandomForestClassifier"
    assert classification_result.y_true is not None
    assert classification_result.y_pred is not None


def test_metrics_functions_return_evaluation_result():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    result = sx.metrics.classification_metrics(y_true, y_pred)
    assert isinstance(result, dict)
    assert isinstance(result, EvaluationResult)

    reg = sx.metrics.regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    assert isinstance(reg, EvaluationResult)
    assert reg.task == "regression"


def test_cross_validate_returns_dict_compatible(iris_split):
    X, y = iris_split
    model = sx.supervised.LogisticRegression(max_iter=500)
    result = sx.model_selection.cross_validate(model, X, y, cv=3, verbose=False)
    assert isinstance(result, dict)
    assert isinstance(result, CrossValidationResult)
    assert set(result) == {"scores", "media", "desviacion"}
    assert result.cv == 3


def test_compare_models_returns_dataframe_compatible(iris_split):
    X, y = iris_split
    table = sx.model_selection.compare_models(X, y, task="classification", cv=2, verbose=False)
    assert isinstance(table, pd.DataFrame)
    assert isinstance(table, ModelComparisonResult)
    assert "modelo" in table.columns
    assert table.scoring_name == "accuracy"
    # to_html sigue siendo el de pandas (devuelve la tabla HTML).
    assert "<table" in table.to_html()


# ----------------------------------------------------------------------
# Payloads (contrato compartido con statslibx/viewx)
# ----------------------------------------------------------------------


def _validate_contract(payload):
    """Estructura mínima que exige viewx.from_report_payload."""
    assert isinstance(payload, dict)
    assert "title" in payload
    assert isinstance(payload.get("sections", []), list)
    assert isinstance(payload.get("tables", []), list)
    for table in payload.get("tables", []):
        assert "name" in table and "data" in table


def test_evaluation_payload(classification_result):
    payload = classification_result.to_report_data()
    _validate_contract(payload)
    assert payload["metadata"]["result_type"] == "EvaluationResult"
    assert payload["valueboxes"]


def test_regression_payload(regression_result):
    payload = regression_result.to_report_data()
    _validate_contract(payload)
    metric_names = payload["tables"][0]["data"][0].keys()
    assert "R2" in metric_names


def test_cross_validation_payload(iris_split):
    X, y = iris_split
    model = sx.supervised.LogisticRegression(max_iter=500)
    result = sx.model_selection.cross_validate(model, X, y, cv=3, verbose=False)
    payload = result.to_report_data()
    _validate_contract(payload)
    assert len(payload["tables"][0]["data"]) == 3


def test_comparison_payload(iris_split):
    X, y = iris_split
    table = sx.model_selection.compare_models(X, y, task="classification", cv=2, verbose=False)
    payload = table.to_report_data()
    _validate_contract(payload)
    assert payload["metadata"]["scoring"] == "accuracy"


def test_dict_passthrough():
    payload = {"title": "Custom", "sections": [], "tables": []}
    assert to_report_data(payload) is payload


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        to_report_data(42)


# ----------------------------------------------------------------------
# Disponibilidad de ViewX (error claro sin instalar)
# ----------------------------------------------------------------------


def test_require_viewx_raises_without_viewx(monkeypatch):
    monkeypatch.setattr(_availability, "VIEWX_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"synaptix\[viewx\]"):
        _availability.require_viewx("test")


def test_to_html_raises_without_viewx(monkeypatch, classification_result):
    monkeypatch.setattr(_availability, "VIEWX_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"synaptix\[viewx\]"):
        classification_result.to_html("out.html")


# ----------------------------------------------------------------------
# Figuras Plotly (requiere plotly)
# ----------------------------------------------------------------------


def test_payload_with_figures(classification_result):
    pytest.importorskip("plotly")
    payload = classification_result.to_report_data(include_figures=True)
    _validate_contract(payload)
    assert payload["figures"]
    fig = payload["figures"][0]["fig"]
    assert hasattr(fig, "to_plotly_json")


def test_regression_payload_with_figures(regression_result):
    pytest.importorskip("plotly")
    payload = regression_result.to_report_data(include_figures=True)
    kinds = {f["kind"] for f in payload["figures"]}
    assert {"predictions", "residuals"} <= kinds


# ----------------------------------------------------------------------
# Round-trip con ViewX (requiere viewx instalado)
# ----------------------------------------------------------------------


def test_roundtrip_html_with_viewx(tmp_path, classification_result):
    pytest.importorskip("viewx")
    out = classification_result.to_html(
        str(tmp_path / "report.html"), include_figures=False
    )
    assert (tmp_path / "report.html").exists() or out


def test_from_report_payload_accepts_synaptix_payload(tmp_path, classification_result):
    viewx = pytest.importorskip("viewx")
    payload = classification_result.to_report_data()
    out = viewx.from_report_payload(
        payload, target="html", filename=str(tmp_path / "payload.html")
    )
    assert out


# ----------------------------------------------------------------------
# Puente statslibx (requiere statslibx instalado)
# ----------------------------------------------------------------------


def test_diagnostics_regression():
    pytest.importorskip("statslibx")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([2.0, -1.0]) + rng.normal(scale=0.05, size=60)
    model = sx.supervised.LinearRegression()
    model.fit(X, y)
    results = model.diagnostics(X, y, verbose=False)
    assert "normalidad_residuos" in results
    assert "media_residuos_cero" in results


def test_diagnostics_classification(iris_split):
    pytest.importorskip("statslibx")
    X, y = iris_split
    model = sx.supervised.RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    results = model.diagnostics(X, y, verbose=False)
    assert "asociacion_real_predicho" in results


def test_diagnostics_raises_without_statslibx(monkeypatch, iris_split):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "statslibx" or name.startswith("statslibx."):
            raise ImportError("statslibx bloqueado para el test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    X, y = iris_split
    model = sx.supervised.RandomForestClassifier(n_estimators=5)
    model.fit(X, y)
    with pytest.raises(ImportError, match=r"synaptix\[stats\]"):
        model.diagnostics(X, y)


# ----------------------------------------------------------------------
# Interoperabilidad de datos
# ----------------------------------------------------------------------


def test_polars_input():
    pl = pytest.importorskip("polars")
    df = sx.load_dataset("iris")
    pl_df = pl.from_pandas(df)

    model = sx.supervised.RandomForestClassifier(n_estimators=5)
    model.fit(pl_df.drop("species"), pl_df["species"])
    assert model.fitted
    assert model.feature_names_ == [c for c in df.columns if c != "species"]
    preds = model.predict(pl_df.drop("species"))
    assert len(preds) == len(df)


def test_load_dataset_polars_backend():
    pl = pytest.importorskip("polars")
    df = sx.load_dataset("iris", backend="polars")
    assert isinstance(df, pl.DataFrame)


def test_load_dataset_return_X_y():
    X, y = sx.load_dataset("iris", return_X_y=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_dataset_return_X_y_custom_target():
    X, y = sx.load_dataset("iris", return_X_y="petal_width")
    assert X.shape == (150, 4)


def test_load_dataset_default_unchanged():
    df = sx.load_dataset("iris")
    assert isinstance(df, pd.DataFrame)
    assert "species" in df.columns
