"""
Presets de ejecución: ``fast``, ``balanced`` (por defecto) y ``exhaustive``.

Fijan los valores que el usuario no indique explícitamente: folds, número
de combinaciones de la búsqueda y cuántos modelos del ranking se ajustan.
Los candidatos de cada modo están en
:data:`synaptix.automl.selection.MODE_CANDIDATES`.
"""

from __future__ import annotations

from typing import Any, Dict

from .exceptions import AutoMLError

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {"cv": 3, "n_iter": 10, "tune_top_k": 1},
    "balanced": {"cv": 5, "n_iter": 20, "tune_top_k": 1},
    "exhaustive": {"cv": 5, "n_iter": 60, "tune_top_k": 3},
}


def resolve_preset(mode: str, **overrides: Any) -> Dict[str, Any]:
    """Valores del preset, sustituyendo los que no sean ``None``."""
    if mode not in PRESETS:
        raise AutoMLError(f"mode debe ser uno de {list(PRESETS)} (recibido: {mode!r}).")
    values = dict(PRESETS[mode])
    values.update({k: v for k, v in overrides.items() if v is not None})
    return values
