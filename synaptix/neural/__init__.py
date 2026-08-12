"""
synaptix.neural - Redes neuronales (requiere tensorflow: pip install synaptix[dl]).

- ``MLP``     : perceptrón multicapa para datos tabulares.
- ``CNN``     : red convolucional para imágenes.
- ``LSTMNet`` : red recurrente para series de tiempo.

Ejemplo
-------
>>> from synaptix.neural import MLP
>>> net = MLP(task="classification", hidden_layers=(64, 32), dropout=0.2)
>>> net.fit(X_train, y_train, epochs=50)
>>> net.evaluate(X_test, y_test)
>>> net.plot_history()
"""

from .networks import CNN, LSTMNet, MLP

__all__ = ["MLP", "CNN", "LSTMNet"]
