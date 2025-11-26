"""
SynaptIX - Librería de Inteligencia Artificial para Python
Autor: Emmanuel Ascendra
Versión: 0.1.0

Incluye:
- Machine Learning
- Deep Learning
- Visión Artificial
- Procesamiento del Lenguaje Natural
- IA Clásica
"""

__version__ = "0.1.0"
__author__ = "Emmanuel Ascendra"

# Importar clases principales
from .intelligenceartificial import Intelligence

# Qué exponer con: from synaptix import *
__all__ = [
    "Intelligence",
]

# Mensaje de bienvenida
def welcome():
    """Imprime información sobre la librería SynaptIX."""
    header = f"\033[1;96mSynaptIX v{__version__}\033[0m"
    print(header)
    print("Librería integral de Inteligencia Artificial para Python")
    print(f"Autor: {__author__}\n")

    print("Clases disponibles:")
    print("  • Intelligence : IA completa (ML, DL, Visión, NLP, IA clásica)\n")

    print("Usa help(Intelligence) para ver todos los métodos.")
    print("Repositorio: https://github.com/Immanuel3008/SynaptIX")
