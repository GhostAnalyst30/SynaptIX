"""Benchmark pandas vs polars por etapa dentro de SynaptIX.

Etapas medidas por separado (mediana de ``--repeats`` procesos nuevos):

- ``carga``: leer CSV / Parquet y dejar un DataFrame de pandas listo para
  scikit-learn (con polars incluye ``to_pandas``).
- ``perfilado``: ``profile_data`` (nulos, cardinalidad, duplicados...).
  ``polars_desde_pandas`` incluye el costo de ``pl.from_pandas``.
- ``preprocesamiento``: ``ColumnTransformer`` (idéntico en ambos motores:
  scikit-learn siempre recibe pandas).
- ``flujo_completo``: ``AutoPipeline.fit`` desde el archivo.

Uso::

    python benchmarks/bench_engines.py                 # 10k, 100k y 1M filas
    python benchmarks/bench_engines.py --sizes 10000 --repeats 1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _common import environment, measure, save  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), ".data")
TARGET = "target"


def make_dataset(n: int, seed: int = 0):
    """12 numéricas, 4 categóricas, 1 booleana, ~5 % de nulos; target binario."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(12)])
    for j, k in enumerate((3, 10, 50, 500)):
        df[f"cat_{j}"] = np.char.add("v", rng.integers(0, k, n).astype(str))
    df["flag"] = rng.random(n) > 0.5
    logits = X[:, 0] - 0.5 * X[:, 1] + (df["cat_0"] == "v1").to_numpy() * 0.8
    df[TARGET] = np.where(logits + rng.normal(0, 1, n) > 0, "si", "no")
    for col in ("num_0", "num_5", "cat_1", "cat_3"):
        df.loc[rng.random(n) < 0.05, col] = None
    return df


def dataset_paths(n: int):
    os.makedirs(DATA_DIR, exist_ok=True)
    csv = os.path.join(DATA_DIR, f"engines_{n}.csv")
    parquet = os.path.join(DATA_DIR, f"engines_{n}.parquet")
    if not (os.path.exists(csv) and os.path.exists(parquet)):
        df = make_dataset(n)
        df.to_csv(csv, index=False)
        df.to_parquet(parquet, index=False)
    return csv, parquet


# ----------------------------------------------------------------- tareas (proceso hijo)
def task_load(path: str, engine: str):
    from synaptix.automl.data.adapters import load_data

    loaded, _ = load_data(path, TARGET, engine)
    return {"rows": len(loaded.frame), "seconds": loaded.load_seconds + loaded.convert_seconds,
            "convert_seconds": loaded.convert_seconds}


def task_profile(path: str, engine: str):
    import time

    import pandas as pd

    from synaptix.automl.data.intelligence import profile_data

    df = pd.read_parquet(path)
    if engine == "pandas":
        start = time.perf_counter()
        profile_data(df, TARGET, "classification")
        return {"seconds": time.perf_counter() - start}
    import polars as pl

    start = time.perf_counter()
    if engine == "polars_desde_pandas":
        frame = pl.from_pandas(df)
    else:
        frame = pl.read_parquet(path)
        start = time.perf_counter()
    profile_data(df, TARGET, "classification", polars_frame=frame)
    return {"seconds": time.perf_counter() - start}


def task_preprocess(path: str):
    import time

    import pandas as pd

    from synaptix.automl.data.detection import infer_feature_types
    from synaptix.automl.features import build_preprocessor

    df = pd.read_parquet(path)
    X = df.drop(columns=TARGET)
    start = time.perf_counter()
    build_preprocessor(infer_feature_types(X)).fit_transform(X)
    return {"seconds": time.perf_counter() - start}


def task_automl(path: str, engine: str):
    import synaptix as sx

    pipe = sx.AutoPipeline(engine=engine, mode="fast", models=["hist_gradient_boosting"],
                           tune=False, explain=False, cv=3, verbose=False, random_state=0)
    pipe.fit(path, target=TARGET)
    out = {f"etapa_{k}": v for k, v in pipe.timings_.items()}
    out["test_accuracy"] = pipe.test_score_
    return out


# ----------------------------------------------------------------- orquestación
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--automl-max", type=int, default=100_000,
                        help="Tamaño máximo para el flujo completo.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stages", nargs="+",
                        default=["carga", "perfilado", "preprocesamiento", "flujo_completo"])
    parser.add_argument("--output", default="engines")
    args = parser.parse_args()

    results = []
    for n in args.sizes:
        csv, parquet = dataset_paths(n)
        size_mb = {"csv": os.path.getsize(csv) / 1e6, "parquet": os.path.getsize(parquet) / 1e6}
        print(f"\n== {n:,} filas (CSV {size_mb['csv']:.1f} MB, Parquet {size_mb['parquet']:.1f} MB)")

        def record(stage, variant, fmt, out):
            row = {"rows": n, "stage": stage, "variant": variant, "format": fmt,
                   "file_mb": round(size_mb.get(fmt, 0.0), 2),
                   "seconds": out["seconds"], "seconds_stdev": out.get("seconds_stdev"),
                   "peak_mb": out["peak_mb"],
                   "extra": {k: v for k, v in out.items()
                             if k not in ("runs", "seconds", "seconds_stdev", "peak_mb",
                                          "peak_mb_stdev")}}
            results.append(row)
            print(f"  {stage:<16} {variant:<20} {fmt:<8} {out['seconds']:8.3f} s  "
                  f"pico {out['peak_mb']:8.1f} MB")

        if "carga" in args.stages:
            for fmt, path in (("csv", csv), ("parquet", parquet)):
                for engine in ("pandas", "polars"):
                    record("carga", engine, fmt,
                           measure("bench_engines", "task_load", args.repeats, path=path,
                                   engine=engine))
        if "perfilado" in args.stages:
            for engine in ("pandas", "polars", "polars_desde_pandas"):
                record("perfilado", engine, "memoria",
                       measure("bench_engines", "task_profile", args.repeats, path=parquet,
                               engine=engine))
        if "preprocesamiento" in args.stages:
            record("preprocesamiento", "sklearn", "memoria",
                   measure("bench_engines", "task_preprocess", args.repeats, path=parquet))
        if "flujo_completo" in args.stages and n <= args.automl_max:
            for engine in ("pandas", "polars"):
                record("flujo_completo", engine, "csv",
                       measure("bench_engines", "task_automl", args.repeats, path=csv,
                               engine=engine))

    path = save(args.output, {"environment": environment(), "repeats": args.repeats,
                            "results": results})
    print(f"\nResultados: {path}")


if __name__ == "__main__":
    main()
