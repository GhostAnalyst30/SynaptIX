"""
Puente SynaptIX–ViewX — integración opcional de visualización.

ViewX no es requerido para el núcleo de machine learning. Instala con:
``pip install synaptix[viewx]``
"""

from __future__ import annotations

from ._availability import VIEWX_AVAILABLE

__all__ = [
    "VIEWX_AVAILABLE",
]


def __getattr__(name: str):
    """Exports perezosos para evitar imports circulares al cargar el núcleo."""
    if name in ("HTML", "Report", "DataMatrix", "Presentation", "Slide", "to_report_data", "VIEWX_CLASSES_AVAILABLE"):
        from . import adapters
        return getattr(adapters, name)
    if name in ("ViewXExportMixin",):
        from .export import ViewXExportMixin
        return ViewXExportMixin
    if name in ("render_html", "render_report", "render_presentation", "render"):
        from . import renderers
        return getattr(renderers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
