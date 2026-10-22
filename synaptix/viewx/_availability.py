"""
Detección de disponibilidad de ViewX para la integración opcional.
"""

from __future__ import annotations

import importlib.util
from typing import NoReturn

VIEWX_AVAILABLE: bool = importlib.util.find_spec("viewx") is not None

_VIEWX_INSTALL_MSG = (
    "ViewX no está instalado. Instala con: pip install synaptix[viewx]"
)


def require_viewx(feature: str = "visualización") -> None:
    """
    Lanza un error claro cuando se requiere ViewX y no está instalado.

    Parameters
    ----------
    feature : str
        Nombre legible de la funcionalidad, mostrado en el mensaje de error.

    Raises
    ------
    ImportError
        Cuando el paquete ``viewx`` no está disponible.
    """
    if not VIEWX_AVAILABLE:
        raise ImportError(f"{_VIEWX_INSTALL_MSG} (requerido para {feature}).")


def viewx_import_error() -> NoReturn:
    """Lanza el error estándar de instalación de ViewX."""
    raise ImportError(_VIEWX_INSTALL_MSG)
