
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, List

import os


class DeepLearning:
    """
    Clase general para Inteligencia Artificial + ML + DL + Visión + NLP + IA clásica.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray, list], 
                backend: Literal['pandas'] = 'pandas'):
        
        if isinstance(data, str) and os.path.exists(data):
            data = DeepLearning.from_file(data).data

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'var': data})
            else:
                data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])

        if isinstance(data, list):
            data = pd.DataFrame({'var': data})

        self.data = data
        self.backend = backend
        self._numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def from_file(path: str):
        """
        Carga automática de archivos y devuelve instancia de Intelligence.
        Soporta CSV, Excel, TXT, JSON, Parquet, Feather, TSV.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(path)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        elif ext in [".txt", ".tsv"]:
            df = pd.read_table(path)

        elif ext == ".json":
            df = pd.read_json(path)

        elif ext == ".parquet":
            df = pd.read_parquet(path)

        elif ext == ".feather":
            df = pd.read_feather(path)

        else:
            raise ValueError(f"Formato no soportado: {ext}")

        return DeepLearning(df)
    

    # =========================
    # Redes neuronales
    # =========================
    def dense_network(
        self,
        input_dim: int,
        layers: list,
        activation: str = "relu",
        output_activation: str = "linear",
        loss: str = "mse",
        optimizer: str = "sgd",
        lr: float = 0.01
    ):
        pass

    # =========================
    # Entrenamiento
    # =========================
    def fit(self, X, y, epochs: int = 100, batch_size: int = 32):
        pass

    def predict(self, X):
        pass

    def evaluate(self, X, y):
        pass

    def summary(self):
        pass

    # =========================
    # Activaciones
    # =========================
    def relu(self, x):
        pass

    def sigmoid(self, x):
        pass

    def tanh(self, x):
        pass

    # =========================
    # Funciones de pérdida
    # =========================
    def mse(self, y_true, y_pred):
        pass

    def cross_entropy(self, y_true, y_pred):
        pass

    # =========================
    # Optimizadores
    # =========================
    def sgd(self, grads):
        pass

    def momentum(self, grads, beta: float = 0.9):
        pass

    # =========================
    # Utilidades
    # =========================
    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass




    def help(self):
        """
        Muestra ayuda completa de la clase Intelligence
        """
        help_text = """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                   🤖 CLASE Intelligence - AYUDA COMPLETA                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝

    📝 DESCRIPCIÓN:
    Clase para Machine Learning, Deep Learning, Visión Artificial,
    Procesamiento de Lenguaje Natural e Inteligencia Artificial clásica.
    Compatible con estilo StatsLibX.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📋 MÉTODOS PRINCIPALES:

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 1. ⭐ MACHINE LEARNING                                                     │
    └────────────────────────────────────────────────────────────────────────────┘
    • linear_regression(y, X)      → Regresión lineal
    • logistic_regression(y, X)    → Regresión logística (clasificación)
    • svm_classifier(y, X, kernel='rbf') → SVM classifier
    • kmeans(n_clusters=3, columns=None) → K-Means clustering
    """
        print(help_text)