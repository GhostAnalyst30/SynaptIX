"""
Reducción de dimensionalidad.

Ejemplo
-------
>>> from synaptix.unsupervised import PCA
>>> pca = PCA(n_components=2)
>>> X_2d = pca.fit_transform(X)
>>> pca.plot_variance()  # varianza explicada por componente
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as _SkPCA
from sklearn.manifold import TSNE as _SkTSNE

from ..base import ArrayLike, to_matrix


class PCA:
    """Análisis de componentes principales.

    Proyecta los datos a un espacio de menor dimensión maximizando la
    varianza retenida.

    Parameters
    ----------
    n_components : int o float, default=2
        Número de componentes, o fracción de varianza a retener
        (por ejemplo ``0.95``).

    Ejemplo
    -------
    >>> pca = PCA(n_components=2)
    >>> X_reducido = pca.fit_transform(X)
    >>> pca.explained_variance()
    """

    def __init__(self, n_components: Union[int, float] = 2, **kwargs):
        self.model = _SkPCA(n_components=n_components, **kwargs)
        self.fitted = False

    def fit(self, X: ArrayLike) -> "PCA":
        """Aprende los componentes principales."""
        self.model.fit(to_matrix(X))
        self.fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Proyecta los datos al espacio reducido."""
        return self.model.transform(to_matrix(X))

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Equivale a ``fit`` seguido de ``transform``."""
        self.fitted = True
        return self.model.fit_transform(to_matrix(X))

    def inverse_transform(self, X_reduced: ArrayLike) -> np.ndarray:
        """Reconstruye (aproximadamente) los datos originales."""
        return self.model.inverse_transform(np.asarray(X_reduced))

    def explained_variance(self) -> pd.Series:
        """Fracción de varianza explicada por cada componente."""
        if not self.fitted:
            raise RuntimeError("Llama a fit(X) primero.")
        ratios = self.model.explained_variance_ratio_
        return pd.Series(
            ratios, index=[f"PC{i+1}" for i in range(len(ratios))], name="varianza"
        )

    def plot_variance(self) -> None:
        """Grafica la varianza explicada acumulada."""
        variance = self.explained_variance()
        cumulative = variance.cumsum()

        plt.figure(figsize=(7, 4))
        plt.bar(variance.index, variance.values, alpha=0.6, label="Individual")
        plt.plot(variance.index, cumulative.values, "o-", color="crimson", label="Acumulada")
        plt.ylabel("Varianza explicada")
        plt.title("PCA: varianza explicada")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


class TSNE:
    """t-SNE para visualización de datos de alta dimensión en 2D/3D.

    Parameters
    ----------
    n_components : int, default=2
        Dimensión del espacio de salida (2 o 3).
    perplexity : float, default=30.0
        Balance entre estructura local y global (típico: 5-50).

    Notas
    -----
    t-SNE no tiene ``transform`` para datos nuevos; usa
    :meth:`fit_transform` sobre todo el conjunto.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42,
        **kwargs,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self.kwargs = kwargs

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Proyecta los datos al espacio de baja dimensión."""
        X = to_matrix(X)
        perplexity = min(self.perplexity, max(5.0, (len(X) - 1) / 3))
        model = _SkTSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            **self.kwargs,
        )
        return model.fit_transform(X)

    def plot(self, X: ArrayLike, labels: Optional[ArrayLike] = None) -> None:
        """Proyecta y grafica los datos, coloreados por ``labels`` si se dan."""
        embedding = self.fit_transform(X)

        plt.figure(figsize=(7, 6))
        if labels is not None:
            labels = np.asarray(labels).ravel()
            for value in np.unique(labels):
                points = embedding[labels == value]
                plt.scatter(points[:, 0], points[:, 1], label=str(value), alpha=0.7)
            plt.legend()
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        plt.title("Proyección t-SNE")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.show()
