import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, List
from datetime import datetime

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import cv2  # opcional si se usa visión
from collections import deque
import heapq

class Intelligence:
    """
    Clase general para Inteligencia Artificial + ML + DL + Visión + NLP + IA clásica.
    Compatible con estilo StatsLibX.
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray, list], 
                 backend: Literal['pandas'] = 'pandas'):
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
        
    # ============================================================================
    #                                MACHINE LEARNING
    # ============================================================================

    def linear_regression(self, y: str, X: List[str]):
        """
        Regresión Lineal (scikit-learn)
        """
        model = LinearRegression()
        model.fit(self.data[X], self.data[y])

        return {
            "type": "linear_regression",
            "coef": model.coef_,
            "intercept": model.intercept_,
            "predict": model.predict
        }

    def logistic_regression(self, y: str, X: List[str]):
        """
        Regresión Logística para clasificación
        """
        model = LogisticRegression()
        model.fit(self.data[X], self.data[y])

        return {
            "type": "logistic_regression",
            "classes": model.classes_,
            "coef": model.coef_,
            "predict": model.predict
        }

    def svm_classifier(self, y: str, X: List[str], kernel="rbf"):
        """
        SVM Classifier
        """
        model = SVC(kernel=kernel, probability=True)
        model.fit(self.data[X], self.data[y])

        return {
            "type": "svm_classifier",
            "kernel": kernel,
            "predict": model.predict,
            "predict_proba": model.predict_proba
        }

    def kmeans(self, n_clusters=3, columns=None):
        """
        K-Means Clustering
        """
        cols = columns or self._numeric_cols
        model = KMeans(n_clusters=n_clusters)
        model.fit(self.data[cols])

        return {
            "type": "kmeans",
            "centroids": model.cluster_centers_,
            "labels": model.labels_,
            "predict": model.predict
        }

    # ============================================================================
    #                                DEEP LEARNING
    # ============================================================================

    def dense_network(self, input_dim, layers=[32,16,8], activation="relu"):
        """
        Red neuronal densa simple implementada 'a mano' (sin frameworks).
        Solo forward.
        """

        weights = []
        biases = []

        prev = input_dim
        for layer in layers:
            weights.append(np.random.randn(prev, layer) * 0.1)
            biases.append(np.zeros((layer,)))
            prev = layer

        def forward(x):
            out = x
            for W, b in zip(weights, biases):
                out = np.dot(out, W) + b
                out = np.maximum(out, 0) if activation == "relu" else np.tanh(out)
            return out

        return {
            "type": "dense_network",
            "forward": forward,
            "weights": weights
        }

    def cnn(self):
        """
        Placeholder CNN simple; retornas estructura.
        """
        return {
            "type": "cnn",
            "note": "Implementación ligera; aquí puedes usar TensorFlow/PyTorch luego."
        }

    # ============================================================================
    #                                 VISIÓN ARTIFICIAL
    # ============================================================================

    def detect_edges(self, image, method="canny"):
        """
        Detección de bordes con Canny u Sobel
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "canny":
            edges = cv2.Canny(gray, 100, 200)
        else:
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            edges = np.sqrt(sx**2 + sy**2)

        return edges

    def image_features(self, image):
        """
        Histogramas y características simples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()

        return {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "histogram": hist
        }

    # ============================================================================
    #                       PROCESAMIENTO DEL LENGUAJE NATURAL (NLP)
    # ============================================================================

    def tokenize(self, text: str):
        """
        Tokenización básica
        """
        return text.lower().replace(",", "").replace(".", "").split()

    def vectorize_tfidf(self, documents: List[str]):
        """
        TF-IDF vectorization
        """
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(documents)

        return {
            "type": "tfidf",
            "matrix": X,
            "vocab": tfidf.vocabulary_,
            "transform": tfidf.transform
        }

    # ============================================================================
    #                                IA CLÁSICA
    # ============================================================================

    def a_star(self, start, goal, graph: dict):
        """
        Algoritmo A*
        Grafo como dict: {nodo: {vecino: costo}}
        """

        def heuristic(a, b):
            return abs(a - b)

        pq = [(0, start)]
        visited = set()
        parent = {start: None}
        g = {start: 0}

        while pq:
            cost, node = heapq.heappop(pq)
            if node == goal:
                break
            
            if node in visited:
                continue

            visited.add(node)

            for nxt, w in graph[node].items():
                new_cost = g[node] + w
                if nxt not in g or new_cost < g[nxt]:
                    g[nxt] = new_cost
                    f = new_cost + heuristic(nxt, goal)
                    heapq.heappush(pq, (f, nxt))
                    parent[nxt] = node

        # reconstruir camino
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()

        return path

    def genetic_algorithm(self, fitness_fn, population_size=10, generations=20):
        """
        Algoritmo genético muy simple
        """

        population = [np.random.rand() for _ in range(population_size)]

        for _ in range(generations):
            scores = np.array([fitness_fn(ind) for ind in population])
            probs = scores / np.sum(scores)
            selected = np.random.choice(population, size=population_size, p=probs)

            new_pop = []
            for _ in range(population_size):
                a, b = np.random.choice(selected, 2)
                child = (a + b) / 2
                child += np.random.randn() * 0.01
                new_pop.append(child)

            population = new_pop

        best = max(population, key=fitness_fn)
        return best
    
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

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 2. ⭐ DEEP LEARNING                                                        │
    └────────────────────────────────────────────────────────────────────────────┘
    • dense_network(input_dim, layers=[32,16], activation='relu') → Red densa
    • cnn()                             → Red convolucional placeholder

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 3. ⭐ VISIÓN ARTIFICIAL                                                      │
    └────────────────────────────────────────────────────────────────────────────┘
    • detect_edges(image, method='canny') → Detección de bordes
    • image_features(image)               → Histogramas y características básicas

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 4. ⭐ NLP (Procesamiento del Lenguaje)                                      │
    └────────────────────────────────────────────────────────────────────────────┘
    • tokenize(text)                   → Tokenización básica
    • vectorize_tfidf(documents)       → Vectorización TF-IDF

    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 5. ⭐ IA CLÁSICA                                                             │
    └────────────────────────────────────────────────────────────────────────────┘
    • a_star(start, goal, graph)        → Búsqueda A* en grafos
    • genetic_algorithm(fitness_fn, population_size=10, generations=20)
                                        → Algoritmo genético simple

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 EJEMPLOS DE USO:

    ┌─ Inicialización ─────────────────────────────────────────────────────────┐
    │ import pandas as pd                                                      │
    │ from intelligence import Intelligence                                    │
    │ df = pd.read_csv('datos.csv')                                            │
    │ ai = Intelligence(df)                                                    │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ Machine Learning ───────────────────────────────────────────────────────┐
    │ lr = ai.linear_regression(y='target', X=['feat1','feat2'])               │
    │ print(lr['coef_'], lr['intercept'])                                      │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ Deep Learning ──────────────────────────────────────────────────────────┐
    │ net = ai.dense_network(input_dim=3, layers=[8,4])                        │
    │ out = net['forward'](np.array([[1,2,3]]))                                 │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ Visión ─────────────────────────────────────────────────────────────────┐
    │ edges = ai.detect_edges(image)                                           │
    │ feats = ai.image_features(image)                                         │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ NLP ────────────────────────────────────────────────────────────────────┐
    │ tokens = ai.tokenize("Hola mundo")                                       │
    │ tfidf = ai.vectorize_tfidf(["texto 1","texto 2"])                        │
    └──────────────────────────────────────────────────────────────────────────┘

    ┌─ IA Clásica ─────────────────────────────────────────────────────────────┐
    │ path = ai.a_star(start='A', goal='B', graph=graph_dict)                  │
    │ best = ai.genetic_algorithm(fitness_fn=lambda x: -x**2)                  │
    └──────────────────────────────────────────────────────────────────────────┘

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    🎯 CARACTERÍSTICAS CLAVE:

    ✓ ML, DL, visión, NLP e IA clásica integrados
    ✓ Métodos compatibles con estilo StatsLibX
    ✓ Salidas uniformes y fáciles de encadenar
    ✓ Implementación ligera y escalable

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📚 DOCUMENTACIÓN ADICIONAL:
    Para más información sobre métodos específicos, use:
    help(Intelligence.nombre_metodo)

    ╚════════════════════════════════════════════════════════════════════════════╝
        """
        print(help_text)
