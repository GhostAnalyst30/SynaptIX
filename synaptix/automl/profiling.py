"""
Profiling por etapas: tiempo de reloj y memoria del proceso.

La memoria se mide como memoria residente (RSS) del proceso con ``psutil``
si está instalado; es una medida fiable del proceso principal pero no
incluye los workers de joblib cuando ``n_jobs > 1``. Sin ``psutil`` solo
se registran tiempos.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional

import pandas as pd


def _rss_mb() -> Optional[float]:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1e6


@dataclass
class StageTiming:
    """Medición de una etapa."""

    stage: str
    seconds: float
    rss_start_mb: Optional[float] = None
    rss_end_mb: Optional[float] = None

    @property
    def rss_delta_mb(self) -> Optional[float]:
        if self.rss_start_mb is None or self.rss_end_mb is None:
            return None
        return self.rss_end_mb - self.rss_start_mb


class Profiler:
    """Registra la duración y la memoria de cada etapa del AutoML.

    Ejemplo
    -------
    >>> profiler = Profiler()
    >>> with profiler.stage("preprocesamiento"):
    ...     ...
    >>> profiler.to_frame()
    """

    def __init__(self) -> None:
        self.stages: List[StageTiming] = []
        self._peak: Optional[float] = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        before = _rss_mb()
        start = time.perf_counter()
        try:
            yield
        finally:
            after = _rss_mb()
            self.stages.append(StageTiming(name, time.perf_counter() - start, before, after))
            for value in (before, after):
                if value is not None:
                    self._peak = value if self._peak is None else max(self._peak, value)

    def add(self, name: str, seconds: float) -> None:
        """Registra una etapa medida por fuera (por ejemplo, la carga)."""
        self.stages.append(StageTiming(name, seconds))

    @property
    def total_seconds(self) -> float:
        return float(sum(s.seconds for s in self.stages))

    @property
    def peak_rss_mb(self) -> Optional[float]:
        """Máximo RSS observado en los límites de las etapas."""
        return self._peak

    def to_dict(self) -> Dict[str, float]:
        """``{etapa: segundos}``."""
        out: Dict[str, float] = {}
        for s in self.stages:
            out[s.stage] = out.get(s.stage, 0.0) + s.seconds
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.stages:
            row = asdict(s)
            row["rss_delta_mb"] = s.rss_delta_mb
            rows.append(row)
        return pd.DataFrame(rows, columns=["stage", "seconds", "rss_start_mb",
                                           "rss_end_mb", "rss_delta_mb"])
