"""
Política de paralelismo.

Reglas:

1. Un solo nivel de paralelismo: ``n_jobs`` se aplica a la validación
   cruzada o a la búsqueda de hiperparámetros; los estimadores internos
   (RandomForest, KNN...) usan ``n_jobs=1``. Anidar ambos niveles con
   ``-1`` multiplica procesos e hilos y satura CPU y memoria.
2. En los procesos de joblib (backend loky), joblib ya limita los hilos de
   BLAS/OpenMP a ``cpu_count // n_jobs``. En el proceso principal se
   aplica el mismo límite con ``threadpoolctl`` mientras dura la sección
   paralela, para que los hilos nativos del proceso padre no compitan con
   los workers.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from joblib import effective_n_jobs


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Número real de procesos (``-1`` = todos los núcleos)."""
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs no puede ser 0.")
    return int(effective_n_jobs(n_jobs))


def threads_per_worker(n_jobs: Optional[int]) -> int:
    """Hilos nativos por proceso para no sobrepasar los núcleos disponibles."""
    workers = resolve_n_jobs(n_jobs)
    return max(1, (os.cpu_count() or 1) // workers)


@contextmanager
def native_thread_limits(n_jobs: Optional[int]) -> Iterator[None]:
    """Limita los hilos de BLAS/OpenMP del proceso actual si ``n_jobs > 1``."""
    if resolve_n_jobs(n_jobs) <= 1:
        yield
        return
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads_per_worker(n_jobs)):
        yield
