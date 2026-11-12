"""
AutoPipeline: machine learning de principio a fin en una llamada.

Implementado en :mod:`synaptix.automl`; este módulo se mantiene para no
romper ``from synaptix.pipeline.auto import AutoPipeline``.
"""

from __future__ import annotations

from ..automl.pipeline import AutoPipeline, detect_task

__all__ = ["AutoPipeline", "detect_task"]
