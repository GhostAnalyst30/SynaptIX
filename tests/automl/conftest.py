import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def make_classification_df(n=300, n_classes=2, seed=0, with_nulls=False, with_cats=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    logits = X[:, 0] * 2 - X[:, 1]
    if n_classes == 2:
        y = np.where(logits + rng.normal(0, 0.5, n) > 0, "si", "no")
    else:
        cuts = np.quantile(logits, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.array([f"c{k}" for k in np.digitize(logits, cuts)])
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    if with_cats:
        df["color"] = rng.choice(["rojo", "verde", "azul"], n)
        df["flag"] = rng.random(n) > 0.5
    df["y"] = y
    if with_nulls:
        df.loc[rng.choice(n, n // 10, replace=False), "a"] = np.nan
        if with_cats:
            df.loc[rng.choice(n, n // 10, replace=False), "color"] = None
    return df


def make_regression_df(n=300, seed=0, with_nulls=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 0.3, n) + 10
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["zona"] = rng.choice(["n", "s", "e", "o"], n)
    df["y"] = y
    if with_nulls:
        df.loc[rng.choice(n, n // 10, replace=False), "b"] = np.nan
    return df


@pytest.fixture
def clf_df():
    return make_classification_df()


@pytest.fixture
def reg_df():
    return make_regression_df()


FAST = dict(mode="fast", cv=3, verbose=False, explain=False)
