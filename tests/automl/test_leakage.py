"""
Pruebas de ausencia de fuga de datos.

1. Un transformador espía registra qué filas ve cada ajuste del
   preprocesamiento: ninguna fila de prueba debe aparecer nunca, y en la
   validación cruzada cada ajuste ve solo su fold de entrenamiento.
2. Dataset de ruido puro con miles de features: si la selección de
   features se ajustara fuera de los folds, la validación cruzada daría
   una precisión muy superior al azar. Dentro de los folds debe quedar
   cerca de 0.5. La misma prueba con el procedimiento con fuga confirma
   que el test es sensible.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import synaptix as sx
import synaptix.automl.pipeline as automl_pipeline

from .conftest import make_classification_df

SEEN = []


class Spy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        SEEN.append(frozenset(X.index))
        return self

    def transform(self, X):
        return X


@pytest.fixture
def spy_preprocessor(monkeypatch):
    original = automl_pipeline.build_preprocessor
    SEEN.clear()

    def spied(*args, **kwargs):
        return Pipeline([("spy", Spy()), ("ct", original(*args, **kwargs))])

    monkeypatch.setattr(automl_pipeline, "build_preprocessor", spied)
    yield SEEN


@pytest.mark.parametrize("tuning", ["grid", "random"])
def test_preprocessing_never_sees_test_rows(spy_preprocessor, tuning):
    df = make_classification_df(n=200, with_nulls=True)
    pipe = sx.AutoPipeline(models=["logistic", "random_forest"], cv=4, tuning=tuning,
                           n_iter=3, verbose=False, explain=False, random_state=0)
    pipe.fit(df, target="y")
    test_index = set(pipe._eval["index"])
    assert len(test_index) == 40
    assert spy_preprocessor, "el espía no registró ajustes"
    for seen in spy_preprocessor:
        assert not seen & test_index, "el preprocesamiento vio filas de prueba"
    sizes = sorted({len(s) for s in spy_preprocessor})
    # folds de CV (3/4 de train) y el reentrenamiento final (train completo)
    assert sizes == [120, 160]


@pytest.mark.parametrize("kwargs, n", [
    (dict(tuning="halving", n_iter=6), 1300),
    (dict(tuning="optuna", n_iter=4), 200),
    (dict(tuning=None, ensemble="stacking"), 200),
    (dict(tuning=None, ensemble="blending"), 200),
])
def test_advanced_strategies_never_see_test_rows(spy_preprocessor, kwargs, n):
    if kwargs.get("tuning") == "optuna":
        pytest.importorskip("optuna")
    df = make_classification_df(n=n, with_nulls=True)
    pipe = sx.AutoPipeline(models=["logistic", "naive_bayes"], cv=3, verbose=False,
                           explain=False, random_state=0, **kwargs)
    pipe.fit(df, target="y")
    test_index = set(pipe._eval["index"])
    assert len(spy_preprocessor) > 3
    for seen in spy_preprocessor:
        assert not seen & test_index, "el preprocesamiento vio filas de prueba"
        assert len(seen) <= pipe.n_train_


def test_holdout_and_cv_folds_are_disjoint(spy_preprocessor):
    df = make_classification_df(n=120)
    pipe = sx.AutoPipeline(models=["logistic"], cv=3, tune=False, verbose=False,
                           explain=False, random_state=1)
    pipe.fit(df, target="y")
    fold_fits = [s for s in spy_preprocessor if len(s) < pipe.n_train_]
    assert len(fold_fits) == 3
    union = frozenset().union(*fold_fits)
    # cada fila de train se queda fuera de exactamente un fold
    for row in union:
        assert sum(row not in s for s in fold_fits) == 1


def _noise(n=80, p=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = np.array(["a", "b"] * (n // 2))
    rng.shuffle(y)
    return X, y


def test_leaky_procedure_is_detected_by_this_dataset():
    X, y = _noise()
    selected = SelectKBest(f_classif, k=20).fit(X, y).transform(X)  # fuga: usa todos los datos
    leaky = cross_val_score(LogisticRegression(max_iter=1000), selected, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    assert leaky > 0.8


def test_feature_selection_inside_folds_has_no_leakage():
    X, y = _noise()
    df = X.assign(y=y)
    pipe = sx.AutoPipeline(models=["logistic"], feature_selection="kbest", feature_selection_k=20,
                           cv=5, tune=False, test_size=None, verbose=False, explain=False,
                           random_state=0)
    pipe.fit(df, target="y")
    assert 0.25 <= pipe.cv_score_["mean"] <= 0.75, pipe.cv_score_
