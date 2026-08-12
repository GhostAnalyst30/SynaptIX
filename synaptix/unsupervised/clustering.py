"""
Algoritmos de clustering (aprendizaje no supervisado).

Todos los modelos exponen ``fit / fit_predict / plot`` y aceptan
DataFrames de pandas o arrays de NumPy.

Ejemplo
-------
>>> from synaptix.unsupervised import KMeans
>>> km = KMeans(n_clusters=3)
>>> labels = km.fit_predict(X)
>>> km.plot(X)          # scatter 2D coloreado por cluster
>>> KMeans.elbow(X, k_max=10)  # método del codo para elegir k
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering as _SkAgglo,
    DBSCAN as _SkDBSCAN,
    KMeans as _SkKMeans,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as _SkGMM

from ..base import ArrayLike, to_matrix


class _ClusteringModel:
    """Base para modelos de clustering."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike) -> "_ClusteringModel":
        """Ajusta el modelo a los datos."""
        X = to_matrix(X)
        self.model.fit(X)
        self.labels_ = getattr(self.model, "labels_", None)
        if self.labels_ is None and hasattr(self.model, "predict"):
            self.labels_ = self.model.predict(X)
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """Ajusta el modelo y devuelve la etiqueta de cluster de cada muestra."""
        self.fit(X)
        return self.labels_

    def silhouette(self, X: ArrayLike) -> float:
        """Coeficiente de silhouette del clustering actual (mayor = mejor)."""
        if self.labels_ is None:
            raise RuntimeError("Llama a fit(X) primero.")
        return float(silhouette_score(to_matrix(X), self.labels_))

    def plot(self, X: ArrayLike, features: tuple = (0, 1)) -> None:
        """Grafica los clusters en 2D usando dos features.

        Parameters
        ----------
        X : DataFrame o ndarray
            Datos usados en el clustering.
        features : tuple, default=(0, 1)
            Índices de las dos columnas a graficar.
        """
        from ..visualization import plot_clusters

        if self.labels_ is None:
            raise RuntimeError("Llama a fit(X) primero.")
        plot_clusters(X, self.labels_, features=features, title=self.name)

    def __repr__(self) -> str:
        status = "ajustado" if self.labels_ is not None else "sin ajustar"
        return f"<{self.name} ({status})>"


class KMeans(_ClusteringModel):
    """Clustering K-Means.

    Parameters
    ----------
    n_clusters : int, default=3
        Número de clusters ``k``.

    Notas
    -----
    Usa :meth:`elbow` o :meth:`silhouette_scores` para elegir ``k``.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42, **kwargs):
        super().__init__(
            _SkKMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto", **kwargs),
            "KMeans",
        )
        self.n_clusters = n_clusters

    @property
    def centroids_(self) -> np.ndarray:
        """Coordenadas de los centroides tras el ajuste."""
        return self.model.cluster_centers_

    @staticmethod
    def elbow(X: ArrayLike, k_max: int = 10, plot: bool = True) -> dict:
        """Método del codo: inercia para k = 1..k_max.

        Parameters
        ----------
        X : DataFrame o ndarray
            Datos a agrupar.
        k_max : int, default=10
            Máximo número de clusters a probar.
        plot : bool, default=True
            Mostrar el gráfico de inercia vs. k.

        Returns
        -------
        dict
            ``{k: inercia}`` para cada k probado.
        """
        X = to_matrix(X)
        inertias = {}
        for k in range(1, k_max + 1):
            km = _SkKMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
            inertias[k] = float(km.inertia_)

        if plot:
            plt.figure(figsize=(7, 4))
            plt.plot(list(inertias.keys()), list(inertias.values()), "o-")
            plt.xlabel("Número de clusters (k)")
            plt.ylabel("Inercia")
            plt.title("Método del codo")
            plt.grid(alpha=0.3)
            plt.show()

        return inertias

    @staticmethod
    def silhouette_scores(X: ArrayLike, k_max: int = 10, plot: bool = True) -> dict:
        """Coeficiente de silhouette para k = 2..k_max (mayor = mejor)."""
        X = to_matrix(X)
        scores = {}
        for k in range(2, k_max + 1):
            labels = _SkKMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
            scores[k] = float(silhouette_score(X, labels))

        if plot:
            plt.figure(figsize=(7, 4))
            plt.plot(list(scores.keys()), list(scores.values()), "o-")
            plt.xlabel("Número de clusters (k)")
            plt.ylabel("Silhouette")
            plt.title("Análisis de silhouette")
            plt.grid(alpha=0.3)
            plt.show()

        return scores


class DBSCAN(_ClusteringModel):
    """Clustering por densidad (detecta ruido y formas arbitrarias).

    Parameters
    ----------
    eps : float, default=0.5
        Radio máximo de vecindad.
    min_samples : int, default=5
        Muestras mínimas para formar un núcleo. Los puntos etiquetados
        como ``-1`` son ruido.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        super().__init__(_SkDBSCAN(eps=eps, min_samples=min_samples, **kwargs), "DBSCAN")


class HierarchicalClustering(_ClusteringModel):
    """Clustering jerárquico aglomerativo.

    Parameters
    ----------
    n_clusters : int, default=3
        Número de clusters finales.
    linkage : str, default="ward"
        Criterio de enlace: "ward", "complete", "average" o "single".
    """

    def __init__(self, n_clusters: int = 3, linkage: str = "ward", **kwargs):
        super().__init__(
            _SkAgglo(n_clusters=n_clusters, linkage=linkage, **kwargs),
            "HierarchicalClustering",
        )


class GaussianMixture(_ClusteringModel):
    """Mezcla de gaussianas (clustering probabilístico).

    Parameters
    ----------
    n_components : int, default=3
        Número de componentes gaussianas.
    """

    def __init__(self, n_components: int = 3, random_state: int = 42, **kwargs):
        super().__init__(
            _SkGMM(n_components=n_components, random_state=random_state, **kwargs),
            "GaussianMixture",
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Probabilidad de pertenencia de cada muestra a cada componente."""
        return self.model.predict_proba(to_matrix(X))
