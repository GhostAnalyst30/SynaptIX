"""
synaptix.unsupervised - Aprendizaje no supervisado.

Clustering:
    KMeans (con método del codo y silhouette), DBSCAN,
    HierarchicalClustering, GaussianMixture

Reducción de dimensionalidad:
    PCA (con varianza explicada), TSNE

Ejemplo
-------
>>> from synaptix.unsupervised import KMeans, PCA
>>> KMeans.elbow(X, k_max=10)      # elegir k
>>> labels = KMeans(n_clusters=3).fit_predict(X)
>>> X_2d = PCA(n_components=2).fit_transform(X)
"""

from .clustering import DBSCAN, GaussianMixture, HierarchicalClustering, KMeans
from .decomposition import PCA, TSNE

__all__ = [
    "KMeans",
    "DBSCAN",
    "HierarchicalClustering",
    "GaussianMixture",
    "PCA",
    "TSNE",
]
