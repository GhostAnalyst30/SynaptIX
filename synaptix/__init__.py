"""
SynaptIX - Librería integral de Machine Learning para Python.
Autor: Emmanuel Ascendra

Submódulos:
- ``synaptix.pipeline``        : AutoPipeline y Pipeline declarativo (ML automático).
- ``synaptix.preprocessing``   : limpieza, escalado, codificación, outliers.
- ``synaptix.supervised``      : regresión, clasificación y modelos bayesianos.
- ``synaptix.unsupervised``    : clustering y reducción de dimensionalidad.
- ``synaptix.reinforcement``   : Q-Learning, SARSA, DQN y entorno GridWorld.
- ``synaptix.neural``          : redes MLP, CNN y LSTM (requiere tensorflow).
- ``synaptix.metrics``         : métricas de regresión/clasificación/clustering.
- ``synaptix.model_selection`` : validación cruzada, GridSearch, compare_models.
- ``synaptix.visualization``   : gráficas EDA y de modelos.
- ``synaptix.datasets``        : datasets de ejemplo incluidos.
- ``synaptix.legacy``          : API clásica v0.x (compatibilidad).
- ``synaptix.viewx``           : puente opcional a ViewX (``pip install synaptix[viewx]``).
- ``synaptix.viz``             : figuras Plotly para exportación (``pip install synaptix[viz]``).

Integración opcional (cada una funciona de forma independiente):
- ViewX     : ``model.evaluate(...).to_html("reporte.html")``
- StatsLibX : ``model.diagnostics(X_test, y_test)`` (``pip install synaptix[stats]``)

Ejemplo rápido
--------------
>>> import synaptix as sx
>>> df = sx.load_dataset("iris")
>>> X, y = df.drop(columns="species"), df["species"]
>>> X_train, X_test, y_train, y_test = sx.preprocessing.train_test_split(
...     X, y, test_size=0.2, random_state=42
... )
>>> model = sx.supervised.RandomForestClassifier()
>>> model.fit(X_train, y_train)
>>> model.evaluate(X_test, y_test)
"""

__version__ = "0.2.0"
__author__ = "Emmanuel Ascendra"

from . import (
    metrics,
    model_selection,
    pipeline,
    preprocessing,
    supervised,
    unsupervised,
    reinforcement,
    visualization,
)
from .base import SynaptixModel
from .datasets import list_datasets, load_dataset
from .pipeline import AutoPipeline, Pipeline
from .results import (
    CrossValidationResult,
    EvaluationResult,
    ModelComparisonResult,
    TrainTestResult,
)

# Clases legacy sin dependencias pesadas
from .legacy import (
    DeepLearning,
    IntelligenceArtificial,
    NaturalLanguageProcessing,
    NaturalLanguajeProcessing,
)

__all__ = [
    # Submódulos
    "pipeline",
    "preprocessing",
    "supervised",
    "unsupervised",
    "reinforcement",
    "neural",
    "metrics",
    "model_selection",
    "visualization",
    "datasets",
    "legacy",
    # Utilidades
    "SynaptixModel",
    "AutoPipeline",
    "Pipeline",
    "load_dataset",
    "list_datasets",
    # Resultados exportables
    "EvaluationResult",
    "CrossValidationResult",
    "ModelComparisonResult",
    "TrainTestResult",
    # Integración ViewX (opcional)
    "viewx",
    "viz",
    "VIEWX_AVAILABLE",
    # Legacy
    "MachineLearning",
    "DeepLearning",
    "IntelligenceArtificial",
    "NaturalLanguageProcessing",
    "NaturalLanguajeProcessing",
]


def __getattr__(name):
    # Importaciones perezosas de módulos que requieren tensorflow.
    if name == "neural":
        from . import neural

        return neural
    if name == "MachineLearning":
        from .legacy.machineLearning import MachineLearning

        return MachineLearning
    if name == "datasets":
        from . import datasets

        return datasets
    if name == "legacy":
        from . import legacy

        return legacy
    if name == "viewx":
        from . import viewx

        return viewx
    if name == "viz":
        from . import viz

        return viz
    if name == "VIEWX_AVAILABLE":
        from .viewx import VIEWX_AVAILABLE

        return VIEWX_AVAILABLE
    raise AttributeError(f"module 'synaptix' has no attribute '{name}'")


def welcome():
    """Imprime información sobre la librería SynaptIX."""
    print(f"SynaptIX v{__version__}")
    print("Librería integral de Machine Learning para Python")
    print(f"Autor: {__author__}\n")
    print("Submódulos disponibles:")
    print("  - synaptix.pipeline        : AutoPipeline (ML automático)")
    print("  - synaptix.preprocessing   : limpieza y transformación de datos")
    print("  - synaptix.supervised      : regresión, clasificación y bayesianos")
    print("  - synaptix.unsupervised    : clustering y reducción de dimensión")
    print("  - synaptix.reinforcement   : aprendizaje por refuerzo")
    print("  - synaptix.neural          : redes neuronales (MLP, CNN, LSTM)")
    print("  - synaptix.metrics         : métricas de evaluación")
    print("  - synaptix.model_selection : validación y comparación de modelos")
    print("  - synaptix.visualization   : gráficos de análisis")
    print("  - synaptix.datasets        : datasets de ejemplo\n")
    print("Repositorio: https://github.com/GhostAnalyst30/SynaptIX")
