"""
Redes neuronales de alto nivel sobre Keras.

- ``MLP``     : perceptrón multicapa para tabulares (regresión/clasificación).
- ``CNN``     : red convolucional para imágenes.
- ``LSTMNet`` : red recurrente para series de tiempo.

Requiere tensorflow: ``pip install synaptix[dl]``.

Ejemplo
-------
>>> from synaptix.neural import MLP
>>> net = MLP(task="classification", hidden_layers=(64, 32))
>>> net.fit(X_train, y_train, epochs=50)
>>> net.evaluate(X_test, y_test)
>>> net.plot_history()
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..base import ArrayLike, to_matrix, to_vector


def _require_keras():
    """Importa Keras con un mensaje de error útil si falta tensorflow."""
    try:
        import keras
        from keras import callbacks, layers, models
    except ImportError as error:
        raise ImportError(
            "Este módulo requiere tensorflow/keras. "
            "Instala con: pip install synaptix[dl]"
        ) from error
    return keras, models, layers, callbacks


class _KerasNetwork:
    """Base común: entrenamiento, evaluación, historia y persistencia."""

    def __init__(self):
        self.model = None
        self.history = None
        self.task: str = "regression"
        self._label_map: Optional[dict] = None

    # ------------------------------------------------------------------

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Mapea etiquetas arbitrarias a enteros 0..n_clases-1."""
        classes = np.unique(y)
        self._label_map = {label: index for index, label in enumerate(classes)}
        return np.array([self._label_map[value] for value in y])

    def _decode_labels(self, indices: np.ndarray) -> np.ndarray:
        if self._label_map is None:
            return indices
        inverse = {index: label for label, index in self._label_map.items()}
        return np.array([inverse[int(i)] for i in indices])

    def _fit_keras(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_split: float,
        early_stopping: bool,
        verbose: int,
    ):
        _, _, _, callbacks_module = _require_keras()
        callback_list = []
        if early_stopping and validation_split > 0:
            callback_list.append(
                callbacks_module.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            )
        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=verbose,
        )
        return self

    # ------------------------------------------------------------------

    def plot_history(self) -> None:
        """Grafica las curvas de pérdida (y métrica) del entrenamiento."""
        if self.history is None:
            raise RuntimeError("Entrena la red primero con fit().")

        history = self.history.history
        metric_keys = [k for k in history if k not in ("loss", "val_loss")]

        n_plots = 1 + (1 if metric_keys else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        axes = np.atleast_1d(axes)

        axes[0].plot(history["loss"], label="Train")
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Validación")
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Pérdida")
        axes[0].set_title("Pérdida")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if metric_keys:
            metric = [k for k in metric_keys if not k.startswith("val_")][0]
            axes[1].plot(history[metric], label="Train")
            if f"val_{metric}" in history:
                axes[1].plot(history[f"val_{metric}"], label="Validación")
            axes[1].set_xlabel("Época")
            axes[1].set_ylabel(metric)
            axes[1].set_title(metric.capitalize())
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        """Imprime la arquitectura de la red."""
        if self.model is None:
            raise RuntimeError("Construye la red primero con fit().")
        self.model.summary()

    def save(self, path: str = "model.keras") -> None:
        """Guarda la red en formato Keras."""
        self.model.save(path)

    def load(self, path: str = "model.keras") -> "_KerasNetwork":
        """Carga una red guardada previamente."""
        _, models_module, _, _ = _require_keras()
        self.model = models_module.load_model(path)
        return self


class MLP(_KerasNetwork):
    """Perceptrón multicapa para datos tabulares.

    Parameters
    ----------
    task : {"regression", "classification"}, default="regression"
        Tipo de problema. En clasificación detecta automáticamente si es
        binaria (sigmoid) o multiclase (softmax).
    hidden_layers : tuple, default=(64, 32)
        Neuronas por capa oculta.
    activation : str, default="relu"
        Activación de las capas ocultas.
    dropout : float, default=0.0
        Tasa de dropout tras cada capa oculta (0 = sin dropout).
    scale : bool, default=True
        Escala las features internamente con StandardScaler.
    optimizer : str, default="adam"
        Optimizador de Keras.

    Ejemplo
    -------
    >>> net = MLP(task="classification", hidden_layers=(64, 32), dropout=0.2)
    >>> net.fit(X_train, y_train, epochs=50)
    >>> predicciones = net.predict(X_test)
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        hidden_layers: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
        scale: bool = True,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.task = task
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.scale = scale
        self.optimizer = optimizer
        self.scaler = None
        self.n_classes: Optional[int] = None

    def _build(self, input_dim: int) -> None:
        _, models, layers, _ = _require_keras()

        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation))
            if self.dropout > 0:
                model.add(layers.Dropout(self.dropout))

        if self.task == "regression":
            model.add(layers.Dense(1))
            model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])
        elif self.n_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=self.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
            )
        else:
            model.add(layers.Dense(self.n_classes, activation="softmax"))
            model.compile(
                optimizer=self.optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        self.model = model

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        verbose: int = 1,
    ) -> "MLP":
        """Entrena la red.

        Parameters
        ----------
        X : DataFrame o ndarray
            Features ``(n_muestras, n_features)``.
        y : Series o ndarray
            Variable objetivo.
        epochs : int, default=100
            Épocas máximas (con early stopping por defecto).
        batch_size : int, default=32
            Tamaño de lote.
        validation_split : float, default=0.2
            Fracción de datos para validación.
        early_stopping : bool, default=True
            Detiene el entrenamiento si la validación deja de mejorar.
        """
        X = to_matrix(X).astype(float)
        y = to_vector(y)

        if self.scale:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        if self.task == "classification":
            y = self._encode_labels(y)
            self.n_classes = len(self._label_map)
        else:
            y = y.astype(float)

        self._build(input_dim=X.shape[1])
        return self._fit_keras(
            X, y, epochs, batch_size, validation_split, early_stopping, verbose
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predice valores (regresión) o etiquetas (clasificación)."""
        X = to_matrix(X).astype(float)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        raw = self.model.predict(X, verbose=0)

        if self.task == "regression":
            return raw.ravel()
        if self.n_classes == 2:
            indices = (raw.ravel() > 0.5).astype(int)
        else:
            indices = np.argmax(raw, axis=1)
        return self._decode_labels(indices)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Probabilidades por clase (solo clasificación)."""
        if self.task != "classification":
            raise RuntimeError("predict_proba solo aplica a clasificación.")
        X = to_matrix(X).astype(float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        raw = self.model.predict(X, verbose=0)
        if self.n_classes == 2:
            positive = raw.ravel()
            return np.column_stack([1 - positive, positive])
        return raw

    def evaluate(self, X: ArrayLike, y: ArrayLike, verbose: bool = True) -> dict:
        """Evalúa con las métricas de :mod:`synaptix.metrics`."""
        from .. import metrics as sx_metrics

        y_true = to_vector(y)
        y_pred = self.predict(X)

        if self.task == "classification":
            results = sx_metrics.classification_metrics(y_true, y_pred)
        else:
            results = sx_metrics.regression_metrics(y_true, y_pred)

        if verbose:
            print("\n=== Evaluación: MLP ===")
            for key, value in results.items():
                print(f"  {key:<12}: {value:.4f}")
        return results


class CNN(_KerasNetwork):
    """Red convolucional para clasificación de imágenes.

    Parameters
    ----------
    input_shape : tuple
        Forma de cada imagen, por ejemplo ``(28, 28, 1)`` o ``(32, 32, 3)``.
    n_classes : int
        Número de clases.
    conv_blocks : tuple, default=(32, 64)
        Filtros por bloque convolucional (Conv2D + MaxPooling).
    dense_units : int, default=128
        Neuronas de la capa densa final.
    dropout : float, default=0.3
        Dropout antes de la capa de salida.

    Ejemplo
    -------
    >>> net = CNN(input_shape=(28, 28, 1), n_classes=10)
    >>> net.fit(X_train, y_train, epochs=10)
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int,
        conv_blocks: Tuple[int, ...] = (32, 64),
        dense_units: int = 128,
        dropout: float = 0.3,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.task = "classification"
        self.input_shape = input_shape
        self.n_classes = n_classes

        _, models, layers, _ = _require_keras()

        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        for filters in conv_blocks:
            model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

        if n_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(layers.Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        self.model = model

    def fit(
        self,
        X: np.ndarray,
        y: ArrayLike,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        verbose: int = 1,
    ) -> "CNN":
        """Entrena la red con imágenes ``(n, alto, ancho, canales)``."""
        X = np.asarray(X, dtype=float)
        y = self._encode_labels(to_vector(y))
        return self._fit_keras(
            X, y, epochs, batch_size, validation_split, early_stopping, verbose
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice la clase de cada imagen."""
        raw = self.model.predict(np.asarray(X, dtype=float), verbose=0)
        if self.n_classes == 2:
            indices = (raw.ravel() > 0.5).astype(int)
        else:
            indices = np.argmax(raw, axis=1)
        return self._decode_labels(indices)


class LSTMNet(_KerasNetwork):
    """Red LSTM para pronóstico de series de tiempo.

    Convierte automáticamente una serie 1D en ventanas deslizantes de
    tamaño ``window`` y entrena una LSTM para predecir el siguiente valor.

    Parameters
    ----------
    window : int, default=10
        Número de pasos previos usados para predecir el siguiente.
    units : tuple, default=(50,)
        Unidades por capa LSTM.
    scale : bool, default=True
        Escala la serie internamente a [0, 1].

    Ejemplo
    -------
    >>> net = LSTMNet(window=12, units=(64,))
    >>> net.fit(serie, epochs=50)          # serie: array 1D
    >>> futuro = net.forecast(serie, steps=6)
    """

    def __init__(
        self,
        window: int = 10,
        units: Tuple[int, ...] = (50,),
        scale: bool = True,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.task = "regression"
        self.window = window
        self.units = units
        self.scale = scale
        self.optimizer = optimizer
        self.scaler = None

    @staticmethod
    def make_windows(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convierte una serie 1D en pares (ventana, siguiente_valor).

        Returns
        -------
        tuple
            ``(X, y)`` con formas ``(n, window, 1)`` y ``(n,)``.
        """
        X, y = [], []
        for i in range(window, len(series)):
            X.append(series[i - window : i])
            y.append(series[i])
        return np.array(X).reshape(-1, window, 1), np.array(y)

    def fit(
        self,
        series: ArrayLike,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping: bool = True,
        verbose: int = 1,
    ) -> "LSTMNet":
        """Entrena la red con una serie de tiempo 1D."""
        _, models, layers, _ = _require_keras()

        series = to_vector(series).astype(float)

        if self.scale:
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
            series = self.scaler.fit_transform(series.reshape(-1, 1)).ravel()

        X, y = self.make_windows(series, self.window)

        model = models.Sequential()
        model.add(layers.Input(shape=(self.window, 1)))
        for index, units in enumerate(self.units):
            return_sequences = index < len(self.units) - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dense(1))
        model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])
        self.model = model

        return self._fit_keras(
            X, y, epochs, batch_size, validation_split, early_stopping, verbose
        )

    def forecast(self, series: ArrayLike, steps: int = 1) -> np.ndarray:
        """Pronostica los próximos ``steps`` valores de la serie.

        Parameters
        ----------
        series : ndarray o Series
            Serie histórica (usa los últimos ``window`` valores como semilla).
        steps : int, default=1
            Número de pasos futuros a predecir.

        Returns
        -------
        ndarray
            Valores pronosticados en la escala original.
        """
        series = to_vector(series).astype(float)
        if self.scaler is not None:
            series = self.scaler.transform(series.reshape(-1, 1)).ravel()

        window_values = list(series[-self.window :])
        predictions = []
        for _ in range(steps):
            X = np.array(window_values[-self.window :]).reshape(1, self.window, 1)
            next_value = float(self.model.predict(X, verbose=0).ravel()[0])
            predictions.append(next_value)
            window_values.append(next_value)

        predictions = np.array(predictions)
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).ravel()
        return predictions
