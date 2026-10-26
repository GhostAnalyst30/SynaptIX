"""
Reducción de dimensionalidad.

Ejemplo
-------
>>> from synaptix.unsupervised import PCA
>>> pca = PCA(n_components=2)
>>> X_2d = pca.fit_transform(X)
>>> pca.plot_variance()  # varianza explicada por componente
>>> pca.fit_evaluate(X, test_size=0.2)  # reconstrucción en train vs. test
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as _SkPCA
from sklearn.manifold import TSNE as _SkTSNE

from ..base import ArrayLike, TestSize, _print_train_test, split_data, to_matrix

_PCA_METRICS = ["varianza_explicada", "mse_reconstruccion", "rmse_reconstruccion"]


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

    def evaluate(
        self,
        X: ArrayLike,
        metrics: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """Calidad de la proyección sobre ``X`` (proyectar y reconstruir).

        Parameters
        ----------
        X : array-like
            Datos a evaluar (pueden ser distintos a los de ajuste).
        metrics : sequence of str, optional
            Subconjunto de ``varianza_explicada`` (1 - error de
            reconstrucción / varianza total; mayor es mejor),
            ``mse_reconstruccion`` y ``rmse_reconstruccion``.
        verbose : bool, default=True
            Imprime las métricas.

        Returns
        -------
        EvaluationResult
        """
        from ..metrics import _pick
        from ..results import EvaluationResult

        if not self.fitted:
            raise RuntimeError("Llama a fit(X) primero.")
        names = list(_PCA_METRICS) if metrics is None else _pick(metrics, _PCA_METRICS)

        X = to_matrix(X).astype(float)
        reconstructed = self.model.inverse_transform(self.model.transform(X))
        mse = float(np.mean((X - reconstructed) ** 2))
        total = float(np.mean((X - self.model.mean_) ** 2))
        values = {
            "varianza_explicada": 1 - mse / total if total > 0 else np.nan,
            "mse_reconstruccion": mse,
            "rmse_reconstruccion": float(np.sqrt(mse)),
        }
        results = EvaluationResult(
            {n: values[n] for n in names}, model_name="PCA", task="decomposition"
        )
        if verbose:
            print("\n=== Evaluación: PCA ===")
            for key, value in results.items():
                print(f"  {key:<20}: {value:.4f}")
        return results

    def fit_evaluate(
        self,
        X: ArrayLike,
        test_size: TestSize = 0.2,
        metrics: Optional[Sequence[str]] = None,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        """Ajusta en train y compara la reconstrucción en train y test.

        Parameters
        ----------
        X : array-like
            Datos completos.
        test_size : float, int o None, default=0.2
            Proporción (float) o cantidad (int) de muestras de prueba.
            Con ``None`` ajusta con todo y solo reporta train.
        metrics : sequence of str, optional
            Ver :meth:`evaluate`.
        random_state : int, optional, default=42
            Semilla del split.
        verbose : bool, default=True
            Imprime la tabla train vs. test.

        Returns
        -------
        TrainTestResult
        """
        from ..results import TrainTestResult

        X_train, X_test = X, None
        if test_size is not None:
            X_train, X_test = split_data(X, test_size=test_size, random_state=random_state)

        start = time.perf_counter()
        self.fit(X_train)
        fit_time = time.perf_counter() - start

        train = self.evaluate(X_train, metrics=metrics, verbose=False)
        test = None if X_test is None else self.evaluate(X_test, metrics=metrics, verbose=False)

        result = TrainTestResult(
            train,
            test,
            model_name="PCA",
            task="decomposition",
            test_size=test_size,
            n_train=len(to_matrix(X_train)),
            n_test=None if X_test is None else len(to_matrix(X_test)),
            fit_time=round(fit_time, 4),
            info={
                "n_componentes": int(self.model.n_components_),
                "varianza_por_componente": self.explained_variance().to_dict(),
            },
        )
        if verbose:
            _print_train_test(result)
        return result

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
