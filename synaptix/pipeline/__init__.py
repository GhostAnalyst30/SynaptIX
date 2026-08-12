"""
synaptix.pipeline - Pipelines automáticos para usuarios de todos los niveles.

- ``AutoPipeline`` : fit(df, target) y el sistema hace todo — detecta la
  tarea, limpia los datos, compara modelos, ajusta hiperparámetros y
  evalúa. Cada etapa es configurable por parámetros.
- ``Pipeline``     : declarativo por pasos — el usuario lista el proceso
  y el sistema rellena los parámetros de cada paso.

Ejemplo (principiante)
----------------------
>>> import synaptix as sx
>>> pipe = sx.AutoPipeline()
>>> pipe.fit(df, target="species")
>>> pipe.report()
>>> pipe.predict(df_nuevo)

Ejemplo (intermedio)
--------------------
>>> from synaptix.pipeline import Pipeline
>>> pipe = Pipeline([
...     "clean",
...     ("model", "random_forest"),
...     "tune",
...     ("evaluate", {"plot": True}),
... ])
>>> pipe.fit(df, target="precio")
"""

from .auto import AutoPipeline, detect_task
from .declarative import Pipeline
from .grids import DEFAULT_CANDIDATES, DEFAULT_GRIDS, MODEL_REGISTRY

__all__ = [
    "AutoPipeline",
    "Pipeline",
    "detect_task",
    "MODEL_REGISTRY",
    "DEFAULT_CANDIDATES",
    "DEFAULT_GRIDS",
]
