"""
Pruebas unitarias de Feature Subset Search en SynaptIX AutoML.
"""

import math
import numpy as np
import pandas as pd
import pytest

import synaptix as sx
from synaptix.automl import AutoMLError
from synaptix.automl.selection.feature_subset_search import (
    FeatureSubsetSearch,
    count_combinations,
)
from .conftest import FAST, make_classification_df, make_regression_df


FAST_FS = dict(mode="fast", cv=3, verbose=False, explain=False)


def test_count_combinations():
    assert count_combinations(4, 1, 4) == 15
    assert count_combinations(4, 2, 3) == 6 + 4  # 10
    assert count_combinations(4, 1, 1) == 4
    assert count_combinations(0, 1, 1) == 0
    assert count_combinations(4, 5, 4) == 0


# 1 & 2: Búsqueda exhaustiva con cuatro features y 15 combinaciones
def test_exhaustive_search_15_combinations_and_no_duplicates():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)
    df = pd.DataFrame(X, columns=["edad", "temperatura", "humedad", "presion"])
    df["target"] = y

    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=4,
        models=["logistic"],
    ).fit(df, target="target")

    summary = pipe.feature_search_summary_
    assert summary is not None
    assert summary["n_original_features"] == 4
    assert summary["n_possible_combinations"] == 15
    assert summary["n_evaluated_combinations"] == 15
    assert summary["n_failed_combinations"] == 0

    results = pipe.feature_search_results_
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 15

    # Verificación de distribución de tamaños (4 de 1, 6 de 2, 4 de 3, 1 de 4)
    size_counts = results["n_features"].value_counts().to_dict()
    assert size_counts == {1: 4, 2: 6, 3: 4, 4: 1}

    # Sin duplicados
    subsets_tuples = results["features"].apply(lambda f: tuple(sorted(f)))
    assert len(subsets_tuples) == len(set(subsets_tuples))

    # Ranking ordenado
    ranking = pipe.feature_search_ranking_
    assert isinstance(ranking, pd.DataFrame)
    assert len(ranking) == 15
    assert "rank" in ranking.columns
    assert ranking["rank"].iloc[0] == 1


# 3 & 4: Evaluación independiente por modelo y selección de mejores subconjuntos
def test_independent_evaluation_per_model():
    rng = np.random.default_rng(123)
    n = 100
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = np.where(x0 * 2 + x1 > 0, 1, 0)
    df = pd.DataFrame({"f0": x0, "f1": x1, "f2": x2, "f3": x3, "y": y})

    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        models=["logistic", "decision_tree"],
    ).fit(df, target="y")

    summary = pipe.feature_search_summary_
    best_by_model = summary["best_by_model"]
    assert "logistic" in best_by_model
    assert "decision_tree" in best_by_model

    # Cada modelo tiene su mejor subconjunto registrado
    assert len(best_by_model["logistic"]["features"]) >= 1
    assert len(best_by_model["decision_tree"]["features"]) >= 1

    # Atributos públicos
    assert pipe.best_features_ is not None
    assert pipe.best_model_ is not None
    assert pipe.best_score_ is not None
    assert pipe.best_name_ in ("logistic", "decision_tree")
    assert pipe.best_features_ == best_by_model[pipe.best_name_]["features"]


# 5: Regresión y clasificación
def test_regression_and_classification_feature_search():
    # Clasificación
    clf_df = make_classification_df(n=120)
    pipe_clf = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(clf_df, target="y")
    assert pipe_clf.task_ == "classification"
    assert len(pipe_clf.best_features_) in (1, 2)
    assert pipe_clf.feature_search_ranking_["cv_mean"].is_monotonic_decreasing

    # Regresión
    reg_df = make_regression_df(n=120)
    pipe_reg = sx.AutoPipeline(
        **FAST_FS,
        scoring="rmse",
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=2,
        models=["ridge"],
    ).fit(reg_df, target="y")
    assert pipe_reg.task_ == "regression"
    assert len(pipe_reg.best_features_) in (1, 2)
    # En rmse, menor es mejor -> el ranking se ordena ascendente
    assert pipe_reg.feature_search_ranking_["cv_mean"].is_monotonic_increasing


# 6: Variables numéricas y categóricas
def test_numerical_and_categorical_features():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n),
        "cat1": rng.choice(["A", "B", "C"], size=n),
        "cat2": rng.choice(["X", "Y"], size=n),
        "y": rng.choice(["si", "no"], size=n),
    })

    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df, target="y")

    assert pipe.best_features_ is not None
    # Al menos una combinación con variable categórica fue evaluada exitosamente
    cat_combos = [c for c in pipe.feature_search_results_["features"] if any(k in c for k in ["cat1", "cat2"])]
    assert len(cat_combos) > 0


# 7: Variables derivadas mediante associations
def test_associations_feature_engineering():
    rng = np.random.default_rng(1)
    n = 100
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    y = np.where(a + b > 0, 1, 0)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "y": y})

    pipe = sx.AutoPipeline(
        **FAST_FS,
        associations={"suma_ab": "a + b"},
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df, target="y")

    # La variable derivada 'suma_ab' estuvo disponible y se evaluó
    all_evaluated = [f for combo in pipe.feature_search_results_["features"] for f in combo]
    assert "suma_ab" in all_evaluated

    # Inferencia funciona
    preds = pipe.predict(df.head(5))
    assert len(preds) == 5


# 8: Columnas constantes
def test_constant_columns_handling():
    df = make_classification_df(n=100).assign(const=1.0)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=2,
        drop_constant=True,
        models=["logistic"],
    ).fit(df, target="y")

    assert pipe.best_model_ is not None
    assert "const" not in pipe.best_model_.feature_names_


# 9: Valores faltantes (imputación sin fuga)
def test_missing_values_without_leakage():
    df = make_classification_df(n=120, with_nulls=True)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_min_features=1,
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df, target="y")

    assert pipe.feature_search_summary_["n_failed_combinations"] == 0
    preds = pipe.predict(df.head(5))
    assert len(preds) == 5


# 10: Validación cruzada sin leakage
def test_cross_validation_no_leakage():
    df = make_classification_df(n=150)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        test_size=0.25,
        feature_search="exhaustive",
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df, target="y")

    assert pipe.n_test_ == 38
    assert pipe.test_score_ is not None
    # El test score se calculó una sola vez
    assert "accuracy" in pipe.results_


# 11: Compatibilidad con groups
def test_groups_compatibility():
    df = make_classification_df(n=120)
    df["grp"] = np.repeat(range(30), 4)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="forward",
        models=["logistic"],
    ).fit(df, target="y", groups="grp")

    assert "grp" not in pipe.feature_columns_
    assert "grp" not in pipe.best_features_
    assert pipe.cv_strategy_ is not None


# 12: Compatibilidad con time_column
def test_time_column_compatibility():
    df = make_classification_df(n=120)
    df["fecha"] = pd.date_range("2025-01-01", periods=120)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="forward",
        models=["logistic"],
    ).fit(df, target="y", time_column="fecha")

    assert "fecha" not in pipe.feature_columns_
    assert "fecha" not in pipe.best_features_
    assert pipe.cv_strategy_ == "timeseries"


# 13: Estrategia aleatoria (random)
def test_random_strategy():
    df = make_classification_df(n=100)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="random",
        feature_search_max_combinations=6,
        random_state=42,
        models=["logistic"],
    ).fit(df, target="y")

    assert pipe.feature_search_summary_["strategy_used"] == "random"
    assert len(pipe.feature_search_results_) == 6


# 14: Estrategias forward y backward
def test_forward_and_backward_strategies():
    df = make_classification_df(n=100)
    pipe_fwd = sx.AutoPipeline(
        **FAST_FS,
        feature_search="forward",
        feature_search_max_features=3,
        models=["logistic"],
    ).fit(df, target="y")
    assert pipe_fwd.feature_search_summary_["strategy_used"] == "forward"
    assert len(pipe_fwd.best_features_) <= 3

    pipe_bwd = sx.AutoPipeline(
        **FAST_FS,
        feature_search="backward",
        feature_search_min_features=2,
        models=["logistic"],
    ).fit(df, target="y")
    assert pipe_bwd.feature_search_summary_["strategy_used"] == "backward"
    assert len(pipe_bwd.best_features_) >= 2


# 15: Límites de tiempo y combinaciones
def test_max_combinations_and_time_budget():
    df = make_classification_df(n=100)
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_max_combinations=5,
        models=["logistic"],
    ).fit(df, target="y")
    assert len(pipe.feature_search_results_) == 5
    assert pipe.feature_search_summary_["termination_reason"] == "max_combinations_reached"

    # Time budget agotado
    pipe_budget = sx.AutoPipeline(
        **FAST_FS,
        time_budget=0.0001,
        feature_search="exhaustive",
        models=["logistic"],
    ).fit(df, target="y")
    assert pipe_budget.feature_search_summary_["termination_reason"] == "time_budget_exhausted"


# 16: Compatibilidad con predict sin exigir columnas descartadas
def test_predict_without_discarded_columns():
    rng = np.random.default_rng(99)
    n = 100
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    descartada1 = rng.normal(size=n)
    descartada2 = rng.normal(size=n)
    y = np.where(x0 * 3 + x1 > 0, 1, 0)
    df = pd.DataFrame({"x0": x0, "x1": x1, "desc1": descartada1, "desc2": descartada2, "y": y})

    # Forzar selección de subconjunto de tamaño 2
    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="exhaustive",
        feature_search_min_features=2,
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df, target="y")

    best_feats = pipe.best_features_
    assert len(best_feats) == 2

    # Nuevo DataFrame que contiene ÚNICAMENTE las mejores features
    df_nuevo = df[best_feats].head(5).copy()
    assert set(df_nuevo.columns) == set(best_feats)
    assert not any(c in df_nuevo.columns for c in ["desc1", "desc2"] if c not in best_feats)

    # predict() NO debe arrojar error por la falta de desc1 / desc2
    preds = pipe.predict(df_nuevo)
    assert len(preds) == 5
    proba = pipe.predict_proba(df_nuevo)
    assert proba.shape == (5, 2)


# 17: Modos de AutoPipeline y reportes
def test_autopipeline_modes_and_reports():
    df = make_classification_df(n=80)
    pipe = sx.AutoPipeline(
        mode="fast",
        cv=2,
        verbose=False,
        explain=False,
        feature_search="exhaustive",
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df, target="y")

    assert pipe.best_features_ is not None
    assert pipe.feature_search_summary_ is not None

    from synaptix.automl.reporting import build_report

    full_rep = build_report(pipe)
    assert "feature_search" in full_rep
    assert full_rep["feature_search"]["resumen"]["n_evaluated_combinations"] > 0

    rep = pipe.report(plot=False)
    assert isinstance(rep, dict)


# 18: Compatibilidad con Polars y pandas
def test_polars_compatibility():
    pl = pytest.importorskip("polars")
    df_pandas = make_classification_df(n=100)
    df_polars = pl.from_pandas(df_pandas)

    pipe = sx.AutoPipeline(
        **FAST_FS,
        feature_search="forward",
        feature_search_max_features=2,
        models=["logistic"],
    ).fit(df_polars, target="y")

    assert pipe._engine_used == "polars"
    assert pipe.best_features_ is not None
    assert len(pipe.best_features_) <= 2

    # Predicción con DataFrame de polars que solo contiene las columnas ganadoras
    new_polars = pl.from_pandas(df_pandas[pipe.best_features_].head(5))
    preds = pipe.predict(new_polars)
    assert len(preds) == 5
