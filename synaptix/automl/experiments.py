"""
Experiment Management: registro reproducible de cada ejecución.

Cada ``fit`` genera un :class:`ExperimentRecord` con la configuración,
versiones, semilla, datos, modelos evaluados, métricas, tiempos y
advertencias. Se exporta a JSON (registro completo) y a CSV (ranking).
"""

from __future__ import annotations

import json
import math
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

TRACKED_PACKAGES = ("synaptix", "numpy", "pandas", "scikit-learn", "scipy",
                    "joblib", "threadpoolctl", "polars", "pyarrow", "optuna", "shap")


def package_versions() -> Dict[str, Optional[str]]:
    """Versiones instaladas de las dependencias relevantes (None si falta)."""
    from .. import __version__

    # synaptix se toma del paquete importado: los metadatos pueden ser de
    # otra instalación (p. ej. una copia en site-packages y el repo en sys.path).
    versions: Dict[str, Optional[str]] = {"synaptix": __version__}
    for name in TRACKED_PACKAGES:
        if name == "synaptix":
            continue
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def new_experiment_id() -> str:
    """Identificador corto y único: fecha UTC + sufijo aleatorio."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def to_jsonable(value: Any) -> Any:
    """Convierte tipos de numpy/pandas a tipos serializables en JSON."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


@dataclass
class ExperimentRecord:
    """Registro de un experimento AutoML.

    Attributes
    ----------
    experiment_id : str
        Identificador único.
    created_at : str
        Fecha y hora UTC (ISO 8601).
    config : dict
        Parámetros completos de ``AutoPipeline``.
    environment : dict
        Python, sistema operativo, núcleos y versiones de dependencias.
    random_state : int o None
        Semilla usada.
    data : dict
        Forma, target, tarea, tipos de variables y estrategia de validación.
    ranking : list de dict
        Ranking de modelos (validación cruzada).
    tuning : dict
        Estrategia, mejores parámetros y combinaciones evaluadas.
    ensemble : dict
        Resultado del ensemble, si se pidió.
    selected_model : str
        Modelo final.
    validation : dict
        Métrica principal del modelo final en validación cruzada.
    test_metrics : dict
        Métricas en el conjunto de prueba (evaluado una sola vez).
    timings : dict
        Segundos por etapa.
    warnings : list de str
        Advertencias metodológicas y de datos.
    """

    experiment_id: str = field(default_factory=new_experiment_id)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    random_state: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    tuning: Dict[str, Any] = field(default_factory=dict)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    selected_model: str = ""
    validation: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @staticmethod
    def environment_info() -> Dict[str, Any]:
        import os

        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": package_versions(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(asdict(self))

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Serializa el registro; si se pasa ``path`` también lo guarda."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
        return text

    def to_csv(self, path: Optional[str] = None) -> str:
        """Ranking en CSV con el id del experimento en cada fila."""
        table = pd.DataFrame(self.ranking)
        table.insert(0, "experiment_id", self.experiment_id)
        if "parametros" in table.columns:
            table["parametros"] = table["parametros"].map(
                lambda p: json.dumps(to_jsonable(p), ensure_ascii=False))
        text = table.to_csv(index=False)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as file:
                file.write(text)
        return text

    @classmethod
    def from_json(cls, path: str) -> "ExperimentRecord":
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return cls(**data)
