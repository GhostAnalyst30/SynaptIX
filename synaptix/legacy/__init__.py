"""
synaptix.legacy - API clásica de SynaptIX (v0.x).

Se mantiene por compatibilidad. Para código nuevo, usa los submódulos
modernos: synaptix.supervised, synaptix.unsupervised, synaptix.neural,
synaptix.reinforcement, synaptix.preprocessing, etc.

Nota: ``MachineLearning`` requiere tensorflow instalado; por eso su
importación es perezosa.
"""

from .intelligenceartificial import IntelligenceArtificial
from .deepLearning import DeepLearning
from .naturalLanguageProcessing import (
    NaturalLanguageProcessing,
    NaturalLanguajeProcessing,  # alias por typo histórico
)

__all__ = [
    "IntelligenceArtificial",
    "DeepLearning",
    "NaturalLanguageProcessing",
    "NaturalLanguajeProcessing",
    "MachineLearning",
]


def __getattr__(name):
    # MachineLearning importa keras al cargar el módulo; se difiere para
    # no exigir tensorflow al importar synaptix.
    if name == "MachineLearning":
        from .machineLearning import MachineLearning

        return MachineLearning
    raise AttributeError(f"module 'synaptix.legacy' has no attribute '{name}'")
