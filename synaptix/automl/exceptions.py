"""Excepciones del motor AutoML."""

from __future__ import annotations

from typing import Dict


class AutoMLError(ValueError):
    """Error de configuración o de datos detectado por el motor AutoML."""


class AllModelsFailedError(AutoMLError):
    """Ningún modelo candidato pudo entrenarse ni evaluarse.

    Attributes
    ----------
    errors : dict
        Mensaje de error por nombre de modelo.
    """

    def __init__(self, errors: Dict[str, str]):
        self.errors = dict(errors)
        detail = "\n".join(f"  - {name}: {msg}" for name, msg in self.errors.items())
        super().__init__(
            "Todos los modelos candidatos fallaron; no hay un ganador válido.\n" + detail
        )


class MissingDependencyError(ImportError):
    """Falta una dependencia opcional necesaria para la funcionalidad pedida."""

    def __init__(self, package: str, extra: str, feature: str):
        super().__init__(
            f"{feature} requiere '{package}', que no está instalado. "
            f"Instala con: pip install synaptix[{extra}]"
        )
