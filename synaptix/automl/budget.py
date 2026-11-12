"""Presupuesto de tiempo compartido por las etapas del AutoML."""

from __future__ import annotations

import time
from typing import Optional


class TimeBudget:
    """Cuenta regresiva de un presupuesto en segundos (``None`` = sin límite)."""

    def __init__(self, seconds: Optional[float] = None):
        if seconds is not None and seconds <= 0:
            raise ValueError(f"time_budget debe ser > 0 (recibido: {seconds}).")
        self.seconds = seconds
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def remaining(self) -> Optional[float]:
        """Segundos restantes, o ``None`` si no hay límite."""
        if self.seconds is None:
            return None
        return max(self.seconds - self.elapsed, 0.0)

    @property
    def exhausted(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0
