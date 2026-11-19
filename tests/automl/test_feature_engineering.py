"""
Tests para Feature Engineering persistente (associations).
"""

import numpy as np
import pandas as pd
import pytest

import synaptix as sx
from tests.automl.conftest import FAST, make_classification_df


def test_associations_basic():
    """Test básico: crear variables derivadas en fit y usarlas en predict."""
    df = make_classification_df(n=200, with_nulls=False, with_cats=False)
    # Agregar columnas numéricas para probar expresiones
    df["ingresos"] = np.abs(df["a"]) * 1000 + 50000
    df["gastos"] = np.abs(df["b"]) * 500 + 20000

    original_cols = [c for c in df.columns if c != "y"]

    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={
            "balance": "ingresos - gastos",
            "ratio_gastos": "gastos / ingresos",
        }
    ).fit(df, target="y")

    # Verificar que las columnas derivadas existen en feature_columns_
    assert "balance" in pipe.feature_columns_
    assert "ratio_gastos" in pipe.feature_columns_

    # Predecir con datos nuevos que SOLO tienen las columnas originales
    nuevo = df[original_cols].head(10)
    preds = pipe.predict(nuevo)
    assert len(preds) == 10

    proba = pipe.predict_proba(nuevo)
    assert proba.shape == (10, 2)


def test_associations_with_powers_and_functions():
    """Test con potencias y funciones numpy."""
    df = make_classification_df(n=200, with_nulls=False, with_cats=False)
    df["edad"] = np.abs(df["a"]) * 50 + 20
    original_cols = [c for c in df.columns if c != "y"]

    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={
            "edad_cuadratica": "edad ** 2",
            "log_edad": "log(edad)",
            "sqrt_edad": "sqrt(edad)",
        }
    ).fit(df, target="y")

    assert "edad_cuadratica" in pipe.feature_columns_
    assert "log_edad" in pipe.feature_columns_
    assert "sqrt_edad" in pipe.feature_columns_

    nuevo = df[original_cols].head(5)
    preds = pipe.predict(nuevo)
    assert len(preds) == 5


def test_associations_division_by_zero_handling():
    """Test manejo de división por cero."""
    df = make_classification_df(n=100, with_nulls=False, with_cats=False)
    df["x"] = np.arange(100, dtype=float)
    df.loc[0, "x"] = 0.0  # división por cero
    df["y"] = (df["x"] > 50).astype(int)
    original_cols = [c for c in df.columns if c != "y"]

    # Con manejo "nan" (default)
    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={"ratio": "1 / x"},
        fe_handle_div_zero="nan",
    ).fit(df, target="y")

    nuevo = df[original_cols].head(10)
    preds = pipe.predict(nuevo)
    assert len(preds) == 10
    # La predicción no debe fallar


def test_associations_chained_expressions():
    """Test expresiones encadenadas (una variable derivada usada en otra)."""
    df = make_classification_df(n=150, with_nulls=False, with_cats=False)
    df["a_raw"] = df["a"]
    df["b_raw"] = df["b"]
    original_cols = [c for c in df.columns if c != "y"]

    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={
            "suma": "a_raw + b_raw",
            "suma_cuadrado": "suma ** 2",  # usa 'suma' que se acaba de crear
        }
    ).fit(df, target="y")

    assert "suma" in pipe.feature_columns_
    assert "suma_cuadrado" in pipe.feature_columns_

    nuevo = df[original_cols].head(5)
    preds = pipe.predict(nuevo)
    assert len(preds) == 5


def test_associations_save_load():
    """Test que associations se guarda y carga correctamente con save/load."""
    import tempfile
    import os

    df = make_classification_df(n=100, with_nulls=False, with_cats=False)
    df["ingresos"] = np.abs(df["a"]) * 1000 + 50000
    df["gastos"] = np.abs(df["b"]) * 500 + 20000
    original_cols = [c for c in df.columns if c != "y"]

    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={"balance": "ingresos - gastos"}
    ).fit(df, target="y")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pipe.pkl")
        pipe.save(path)
        loaded = sx.AutoPipeline.load(path)

        # Verificar que el feature engineering se preserva
        assert loaded.fe_transformer_ is not None
        assert "balance" in loaded.feature_columns_

        # Predecir con datos nuevos
        nuevo = df[original_cols].head(5)
        preds_orig = pipe.predict(nuevo)
        preds_loaded = loaded.predict(nuevo)
        assert np.array_equal(preds_orig, preds_loaded)


def test_associations_invalid_expression():
    """Test que expresiones inválidas lanzan error claro."""
    df = make_classification_df(n=50, with_nulls=False)

    # Expresión con sintaxis inválida
    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={"bad": "a + * b"}  # operador binario seguido de operador
    )
    with pytest.raises(Exception) as exc_info:
        pipe.fit(df, target="y")
    assert "inválida" in str(exc_info.value).lower() or "syntax" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()

    # Expresión con columna inexistente
    pipe2 = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={"bad": "columna_inexistente * 2"}
    )
    with pytest.raises(Exception) as exc_info:
        pipe2.fit(df, target="y")
    assert "no existe" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()


def test_associations_missing_column_at_predict():
    """Test que fit falla si faltan columnas base en los datos de entrenamiento."""
    df = make_classification_df(n=100, with_nulls=False, with_cats=False)
    df["ingresos"] = np.abs(df["a"]) * 1000 + 50000
    # NO agregamos 'gastos'

    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={"balance": "ingresos - gastos"}  # 'gastos' no existe
    )
    with pytest.raises(Exception) as exc_info:
        pipe.fit(df, target="y")
    assert "no está disponible" in str(exc_info.value).lower() or "no existe" in str(exc_info.value).lower() or "disponible" in str(exc_info.value).lower()


def test_associations_with_categorical():
    """Test que associations funciona junto con variables categóricas."""
    df = make_classification_df(n=150, with_nulls=True, with_cats=True)
    df["ingresos"] = np.abs(df["a"]) * 1000 + 50000
    original_cols = [c for c in df.columns if c != "y"]

    pipe = sx.AutoPipeline(
        **FAST,
        models=["logistic"],
        associations={"log_ingresos": "log(ingresos)"}
    ).fit(df, target="y")

    assert "log_ingresos" in pipe.feature_columns_

    nuevo = df[original_cols].head(10)
    preds = pipe.predict(nuevo)
    assert len(preds) == 10


def test_no_associations_works_as_before():
    """Test que sin associations el comportamiento es el original."""
    df = make_classification_df(n=100, with_nulls=False)

    pipe = sx.AutoPipeline(**FAST, models=["logistic"]).fit(df, target="y")

    assert pipe.fe_transformer_ is None
    preds = pipe.predict(df.head(5))
    assert len(preds) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])