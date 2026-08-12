import numpy as np
import pandas as pd
from typing import Union, Literal, List, Any, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import os


class MachineLearning:
    """
    Clase general para Machine Learning
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray, list]):

        if isinstance(data, str) and os.path.exists(data):
            data = MachineLearning.from_file(data).data

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        if isinstance(data, list):
            data = pd.DataFrame({"var": data})

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Los datos deben ser DataFrame, ndarray, lista o ruta")

        self.data = data

    @classmethod
    def from_file(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        return cls(df)

    def neural_network(self):
        return NeuralNetwork(self)

    def linear_regression(self):
        pass

    def decision_tree(self):
        pass

    def logistic_regression(self):
        pass

    def random_forest(self):
        pass

    def gradient_boosting(self):
        pass

    def clasification_report(self):
        pass


class NeuralNetwork:
    """
    API de Redes Neuronales
    """

    def __init__(self, parent: MachineLearning):
        self.parent = parent
        self.df = parent.data.copy()

        self.model = None
        self.history = None

        self.scaler = None
        self.y_scaler = None
        self.scale_y = False

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.task = "regression"
        self.window = None
        self.date_col = None

    # =========================
    # DATA
    # =========================

    def data(
        self,
        X: Union[list[str], str],
        y: str,
        task: Literal["regression", "classification", "time_series"] = "regression",
        test_size: float = 0.2,
        random_state: int = 42,
        scale: bool = True,
        scale_y: bool = False,
        window: int = 5,
        date_col: Optional[str] = None
    ):
        self.task = task
        self.window = window
        self.date_col = date_col
        self.scale_y = scale_y

        df = self.df.copy()

        # ordenar por fecha si es serie temporal
        if task == "time_series":
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            else:
                for col in df.columns:
                    if np.issubdtype(df[col].dtype, np.datetime64):
                        df = df.sort_values(col)
                        self.date_col = col
                        break

        X_df = df[X] if isinstance(X, list) else df[[X]]
        y_df = df[y]

        X_values = X_df.values
        y_values = y_df.values.reshape(-1, 1)

        if scale:
            self.scaler = StandardScaler()
            X_values = self.scaler.fit_transform(X_values)

        if task == "regression" and scale_y:
            self.y_scaler = StandardScaler()
            y_values = self.y_scaler.fit_transform(y_values)

        y_values = y_values.ravel()

        # ===== TIME SERIES =====
        if task == "time_series":
            X_seq, y_seq = [], []

            for i in range(window, len(X_values)):
                X_seq.append(X_values[i - window:i])
                y_seq.append(y_values[i])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            split = int(len(X_seq) * (1 - test_size))

            self.X_train = X_seq[:split]
            self.X_test = X_seq[split:]
            self.y_train = y_seq[:split]
            self.y_test = y_seq[split:]

            # flatten automático
            self.X_train = self.X_train.reshape(len(self.X_train), -1)
            self.X_test = self.X_test.reshape(len(self.X_test), -1)

        # ===== REGRESSION / CLASSIFICATION =====
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_values,
                y_values,
                test_size=test_size,
                random_state=random_state
            )

        return self.X_train, self.X_test, self.y_train, self.y_test

    # =========================
    # BUILD
    # =========================

    def build(
        self,
        input_dim: Optional[int] = None,
        layers: List[Any] = [],
        optimizer: str = "adam",
    ):
        if input_dim is None:
            input_dim = self.X_train.shape[1]

        self.model = Sequential()
        first_dense = True

        for layer in layers:
            if isinstance(layer, tuple):
                layer_type, units, activation = layer
                if layer_type.lower() == "dense":
                    if first_dense:
                        self.model.add(Dense(units, activation=activation, input_dim=input_dim))
                        first_dense = False
                    else:
                        self.model.add(Dense(units, activation=activation))

            elif isinstance(layer, (Dropout, BatchNormalization)):
                self.model.add(layer)

            else:
                raise ValueError(f"Capa no soportada: {layer}")

        if self.task in ["regression", "time_series"]:
            self.model.add(Dense(1))
            loss = "mse"
            metrics = ["mae"]
        else:
            self.model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = ["accuracy"]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

    # =========================
    # TRAIN
    # =========================

    def fit(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        verbose: int = 1
    ):
        callbacks = []

        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True
                )
            )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

    # =========================
    # EVALUATION & PLOTS
    # =========================

    def loss_plot(self):
        plt.plot(self.history.history["loss"], label="Train")
        plt.plot(self.history.history["val_loss"], label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=1)

    def plot(self, bins: int = 20):
        y_pred = self.model.predict(self.X_test).flatten()

        if self.task == "regression" and self.scale_y:
            y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_real = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        else:
            y_real = self.y_test

        if self.task == "time_series":
            plt.plot(y_real, label="Real")
            plt.plot(y_pred, label="Predicho")
            plt.legend()
            plt.title("Serie de Tiempo")
            plt.show()

        elif self.task == "regression":
            plt.scatter(y_real, y_pred, alpha=0.7)
            plt.xlabel("Real")
            plt.ylabel("Predicho")

            min_val = min(y_real.min(), y_pred.min())
            max_val = max(y_real.max(), y_pred.max())

            plt.plot([min_val, max_val], [min_val, max_val], "--", color="black")
            plt.title("Regresión: Real vs Predicho")
            plt.show()

        else:
            plt.hist(y_pred, bins=bins)
            plt.title("Distribución de probabilidades")
            plt.show()

    # =========================
    # UTILS
    # =========================

    def predict(self, X_new):
        if self.scaler:
            X_new = self.scaler.transform(X_new)

        y_pred = self.model.predict(X_new)

        if self.task == "regression" and self.scale_y:
            y_pred = self.y_scaler.inverse_transform(y_pred)

        return y_pred

    def save(self, path="model.keras"):
        self.model.save(path)

    def load(self, path="model.keras"):
        self.model = load_model(path)
