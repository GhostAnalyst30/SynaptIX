"""SynaptIX AutoPipeline frente a scikit-learn y AutoML externos.

Todas las herramientas reciben exactamente el mismo split train/test (80/20,
estratificado en clasificación) y se evalúan una sola vez en test. Se repite
con 3 semillas: la estabilidad es la desviación del score de test entre
semillas. Cada ejecución corre en un proceso nuevo (ver ``_common.py``).

Herramientas:

- ``sklearn_baseline``: ColumnTransformer + HistGradientBoosting por defecto.
- ``sklearn_random_search``: lo mismo + ``RandomizedSearchCV`` (10 combinaciones, 3 folds).
- ``synaptix_fast`` / ``synaptix_balanced``: ``AutoPipeline(mode=...)``.
- ``flaml``: ``flaml.AutoML`` con el mismo presupuesto de tiempo que
  ``synaptix_balanced`` (si está instalado).
- ``pycaret`` / ``autogluon``: se registran como no disponibles si no se
  pueden importar en el entorno (no se inventan resultados).

Uso::

    python benchmarks/bench_automl.py
    python benchmarks/bench_automl.py --sizes 5000 --seeds 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _common import environment, measure, save  # noqa: E402

TARGET = "target"


# ----------------------------------------------------------------- datos
def make_data(name: str, n: int, seed: int):
    import pandas as pd

    if name == "penguins":
        import synaptix as sx

        return sx.load_dataset("penguins").rename(columns={"species": TARGET}), "classification"
    rng = np.random.default_rng(1000 + seed)
    X = rng.normal(size=(n, 10))
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(10)])
    df["cat_a"] = np.char.add("a", rng.integers(0, 5, n).astype(str))
    df["cat_b"] = np.char.add("b", rng.integers(0, 40, n).astype(str))
    signal = X[:, 0] * 1.5 - X[:, 1] + np.sin(2 * X[:, 2]) + (df["cat_a"] == "a1") * 1.0
    if name == "classification":
        df[TARGET] = np.where(signal + rng.normal(0, 1, n) > 0.3, "pos", "neg")
    else:
        df[TARGET] = signal * 3 + rng.normal(0, 1, n)
    for col in ("num_0", "num_4", "cat_b"):
        df.loc[rng.random(n) < 0.05, col] = None
    return df, name


def split(df, task: str, seed: int):
    from sklearn.model_selection import train_test_split

    strat = df[TARGET] if task == "classification" else None
    return train_test_split(df, test_size=0.2, random_state=seed, stratify=strat)


def score(task: str, y_true, y_pred, proba=None):
    from sklearn import metrics

    if task == "classification":
        out = {"accuracy": metrics.accuracy_score(y_true, y_pred),
               "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
               "f1_weighted": metrics.f1_score(y_true, y_pred, average="weighted")}
        return out
    return {"r2": metrics.r2_score(y_true, y_pred),
            "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
            "mae": metrics.mean_absolute_error(y_true, y_pred)}


def _sklearn_pipeline(task: str, X):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat),
    ])
    model = (HistGradientBoostingClassifier(random_state=0) if task == "classification"
             else HistGradientBoostingRegressor(random_state=0))
    return Pipeline([("prep", prep), ("model", model)])


def _timed_predict(model, X):
    start = time.perf_counter()
    pred = model.predict(X)
    return pred, time.perf_counter() - start


# ----------------------------------------------------------------- tareas (proceso hijo)
def run_tool(tool: str, dataset: str, n: int, seed: int, budget: float = 60.0):
    from sklearn.base import clone

    df, task = make_data(dataset, n, seed)
    df = df[df[TARGET].notna()]
    train, test = split(df, task, seed)
    X_train, y_train = train.drop(columns=TARGET), train[TARGET].to_numpy()
    X_test, y_test = test.drop(columns=TARGET), test[TARGET].to_numpy()
    if task == "regression":
        y_train, y_test = y_train.astype(float), y_test.astype(float)
    out = {"tool": tool, "dataset": dataset, "rows": len(df), "seed": seed}
    start = time.perf_counter()

    if tool.startswith("sklearn"):
        pipe = _sklearn_pipeline(task, X_train)
        t0 = time.perf_counter()
        clone(pipe[:-1]).fit_transform(X_train, y_train)
        out["preprocess_s"] = time.perf_counter() - t0
        if tool == "sklearn_random_search":
            from scipy.stats import loguniform, randint
            from sklearn.model_selection import RandomizedSearchCV

            search = RandomizedSearchCV(
                pipe, {"model__learning_rate": loguniform(0.01, 0.3),
                       "model__max_leaf_nodes": randint(15, 64),
                       "model__min_samples_leaf": randint(5, 50)},
                n_iter=10, cv=3, random_state=seed, n_jobs=1)
            search.fit(X_train, y_train)
            pipe = clone(pipe).set_params(**search.best_params_)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        out["final_fit_s"] = time.perf_counter() - t0
        model = pipe
        out["selected"] = "HistGradientBoosting"
    elif tool.startswith("synaptix"):
        import synaptix as sx

        mode = tool.split("_", 1)[1]
        pipe = sx.AutoPipeline(mode=mode, test_size=None, explain=False, verbose=False,
                               random_state=seed)
        pipe.fit(train, target=TARGET)
        t0 = time.perf_counter()
        clone(pipe.pipeline_[:-1]).fit_transform(X_train, y_train)
        out["preprocess_s"] = time.perf_counter() - t0
        out["final_fit_s"] = pipe.timings_.get("entrenamiento_final")
        out["selection_s"] = pipe.timings_.get("seleccion_modelos")
        out["tuning_s"] = pipe.timings_.get("optimizacion", 0.0)
        out["selected"] = pipe.best_name_
        model = pipe
    elif tool == "flaml":
        from flaml import AutoML

        automl = AutoML()
        automl.fit(X_train, y_train, task=task, time_budget=budget, seed=seed, verbose=0,
                   metric="accuracy" if task == "classification" else "r2")
        out["selected"] = automl.best_estimator
        out["preprocess_s"] = None
        out["final_fit_s"] = None
        model = automl
    else:
        raise ValueError(tool)

    out["train_total_s"] = time.perf_counter() - start
    y_pred, infer = _timed_predict(model, X_test)
    out["inference_s"] = infer
    out["inference_ms_per_1k"] = 1000 * infer / max(len(X_test), 1) * 1000
    out.update({f"test_{k}": float(v) for k, v in score(task, y_test, y_pred).items()})
    out["seconds"] = out["train_total_s"]
    return out


def availability():
    status = {}
    for tool, module in (("flaml", "flaml.automl"), ("pycaret", "pycaret"),
                         ("autogluon", "autogluon.tabular")):
        try:
            __import__(module)
            status[tool] = "disponible"
        except Exception as exc:  # noqa: BLE001
            status[tool] = f"no disponible: {type(exc).__name__}: {str(exc)[:120]}"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[5_000, 50_000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--balanced-max", type=int, default=10_000,
                        help="Tamaño máximo para synaptix_balanced (y FLAML con su presupuesto).")
    args = parser.parse_args()

    status = availability()
    print("Disponibilidad:", status)
    jobs = [("penguins", 0)] + [(d, n) for d in ("classification", "regression")
                                for n in args.sizes]
    results = []
    for dataset, n in jobs:
        tools = ["sklearn_baseline", "sklearn_random_search", "synaptix_fast"]
        if n <= args.balanced_max:
            tools.append("synaptix_balanced")
        for seed in args.seeds:
            budget = None
            for tool in tools + (["flaml"] if status["flaml"] == "disponible" else []):
                if tool == "flaml" and budget is None:
                    continue
                out = measure("bench_automl", "run_tool", repeats=1, tool=tool,
                              dataset=dataset, n=n, seed=seed, budget=budget or 60.0)
                out.pop("runs", None)
                results.append(out)
                if tool == "synaptix_balanced" or (tool == "synaptix_fast"
                                                   and "synaptix_balanced" not in tools):
                    budget = max(5.0, round(out["train_total_s"]))
                metric = next(k for k in out if k.startswith("test_"))
                print(f"{dataset:<15} n={out['rows']:<6} seed={seed} {tool:<22} "
                      f"{metric}={out[metric]:.4f} total={out['train_total_s']:.1f}s "
                      f"pico={out['peak_mb']:.0f}MB modelo={out['selected']}")

    path = save("automl", {"environment": environment(), "availability": status,
                           "results": results})
    print(f"\nResultados: {path}")


if __name__ == "__main__":
    main()
