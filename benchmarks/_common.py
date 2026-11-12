"""Utilidades compartidas por los benchmarks.

Cada medición corre en un proceso nuevo (``spawn``) para que la memoria
máxima no arrastre lo asignado por mediciones anteriores. Dentro del
proceso, un hilo muestrea el RSS cada 5 ms; el pico reportado es
``max(RSS) - RSS al empezar la tarea`` (incluye memoria nativa de polars y
NumPy, que ``tracemalloc`` no ve). El tiempo y la memoria excluyen el
arranque del proceso y la importación de librerías (``PRELOAD``).
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from multiprocessing import get_context
from typing import Any, Callable, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(os.path.dirname(__file__), "results")
# Se importan antes de medir para no contar la importación como tiempo ni memoria.
PRELOAD = ("numpy", "pandas", "pyarrow", "pyarrow.parquet", "pyarrow.csv", "polars",
           "sklearn", "sklearn.ensemble", "sklearn.model_selection", "synaptix",
           "synaptix.automl")


def environment() -> Dict[str, Any]:
    import psutil

    sys.path.insert(0, ROOT)
    import synaptix

    # La versión de synaptix sale del paquete importado (el del repositorio),
    # no de los metadatos de una instalación previa en site-packages.
    packages = {"synaptix": f"{synaptix.__version__} ({os.path.dirname(synaptix.__file__)})"}
    for name in ("numpy", "pandas", "scikit-learn", "polars", "pyarrow",
                 "optuna", "flaml", "pycaret", "autogluon.tabular", "psutil"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "packages": packages,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _child(module: str, func: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import psutil

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.dirname(__file__))
    for name in PRELOAD:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    target: Callable = getattr(importlib.import_module(module), func)
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        out = target(**kwargs) or {}
    finally:
        seconds = time.perf_counter() - start
        stop.set()
        thread.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    out = dict(out)
    out.setdefault("seconds", seconds)
    out["wall_seconds"] = seconds
    out["peak_mb"] = round((peak[0] - base) / 1e6, 1)
    return out


def measure(module: str, func: str, repeats: int = 3, **kwargs) -> Dict[str, Any]:
    """Ejecuta ``module.func(**kwargs)`` ``repeats`` veces en procesos nuevos.

    Devuelve la mediana de cada campo numérico y los valores crudos.
    """
    runs: List[Dict[str, Any]] = []
    ctx = get_context("spawn")
    for _ in range(repeats):
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            runs.append(pool.submit(_child, module, func, kwargs).result())
    summary: Dict[str, Any] = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            summary[key] = statistics.median(values)
            if len(values) > 1:
                summary[f"{key}_stdev"] = statistics.stdev(values)
        else:
            summary[key] = values[0]
    summary["runs"] = runs
    return summary


def save(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return path
