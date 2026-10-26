"""
Algoritmos de clustering (aprendizaje no supervisado).

Todos los modelos exponen ``fit / fit_predict / predict / evaluate /
fit_evaluate / plot`` y aceptan DataFrames de pandas o arrays de NumPy.

Ejemplo
-------
>>> from synaptix.unsupervised import KMeans
>>> km = KMeans(n_clusters=3)
>>> labels = km.fit_predict(X)
>>> km.plot(X)          # scatter 2D coloreado por cluster
>>> KMeans.elbow(X, k_max=10)  # método del codo para elegir k
>>> km.fit_evaluate(X, test_size=0.3, metrics=["silhouette", "davies_bouldin"])
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering as _SkAgglo,
    DBSCAN as _SkDBSCAN,
    KMeans as _SkKMeans,
)
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as _SkGMM
from sklearn.neighbors import KNeighborsClassifier

from ..base import ArrayLike, TestSize, _print_train_test, split_data, to_matrix


class _ClusteringModel:
    """Base para modelos de clustering."""

    task = "clustering"

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.labels_: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None
        self._assigner: Optional[KNeighborsClassifier] = None

    def fit(self, X: ArrayLike) -> "_ClusteringModel":
        """Ajusta el modelo a los datos."""
        X = to_matrix(X)
        self.model.fit(X)
        self.labels_ = getattr(self.model, "labels_", None)
        if self.labels_ is None and hasattr(self.model, "predict"):
            self.labels_ = self.model.predict(X)
        self._X_fit = X
        self._assigner = None
        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """Ajusta el modelo y devuelve la etiqueta de cluster de cada muestra."""
        self.fit(X)
        return self.labels_

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Asigna un cluster a muestras nuevas.

        Usa el ``predict`` nativo si existe (KMeans, GaussianMixture). En
        DBSCAN y jerárquico, que no generalizan a datos nuevos, asigna el
        cluster del vecino más cercano del conjunto de entrenamiento.
        """
        if self.labels_ is None:
            raise RuntimeError("Llama a fit(X) primero.")
        X = to_matrix(X)
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        if X.shape == self._X_fit.shape and np.array_equal(X, self._X_fit):
            return self.labels_
        if self._assigner is None:
            self._assigner = KNeighborsClassifier(n_neighbors=1).fit(self._X_fit, self.labels_)
        return self._assigner.predict(X)

    def evaluate(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        metrics: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """Métricas de clustering sobre ``X`` (ver ``synaptix.metrics.clustering_metrics``).

        Parameters
        ----------
        X : array-like
            Datos a evaluar (se les asigna cluster con :meth:`predict`).
        y : array-like, optional
            Etiquetas reales; habilita ari, nmi, homogeneity, completeness
            y v_measure.
        metrics : sequence of str, optional
            Métricas a calcular. Por defecto silhouette, calinski_harabasz,
            davies_bouldin (+ externas si se pasa ``y``).
        verbose : bool, default=True
            Imprime las métricas.

        Returns
        -------
        EvaluationResult
            Las métricas que no se pueden calcular (p. ej. menos de 2
            clusters en ``X``) se devuelven como NaN.
        """
        from ..metrics import clustering_metrics
        from ..results import EvaluationResult

        labels = self.predict(X)
        try:
            results = clustering_metrics(X, labels, y_true=y, metrics=metrics)
        except ValueError as error:
            if "requieren y_true" in str(error):
                raise
            warnings.warn(f"{self.name}: {error} Se devuelven NaN.", stacklevel=2)
            names = metrics or ["silhouette", "calinski_harabasz", "davies_bouldin"]
            results = EvaluationResult({name: np.nan for name in names}, task="clustering")

        n_clusters = len(np.unique(labels[labels != -1]))
        results["n_clusters"] = float(n_clusters)
        if np.any(labels == -1):
            results["ruido_pct"] = float(100 * np.mean(labels == -1))
        results.model_name = self.name

        if verbose:
            print(f"\n=== Evaluación: {self.name} ===")
            for key, value in results.items():
                print(f"  {key:<18}: {value:.4f}")
        return results

    def fit_evaluate(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        test_size: TestSize = 0.2,
        metrics: Optional[Sequence[str]] = None,
        random_state: Optional[int] = 42,
        verbose: bool = True,
    ):
        """Ajusta en train y compara métricas de clustering en train y test.

        Parameters
        ----------
        X : array-like
            Datos completos.
        y : array-like, optional
            Etiquetas reales (habilita métricas externas como ari y nmi).
        test_size : float, int o None, default=0.2
            Proporción (float) o cantidad (int) de muestras de prueba.
            Con ``None`` ajusta con todo y solo reporta train.
        metrics : sequence of str, optional
            Métricas a calcular.
        random_state : int, optional, default=42
            Semilla del split.
        verbose : bool, default=True
            Imprime la tabla train vs. test.

        Returns
        -------
        TrainTestResult
        """
        from ..results import TrainTestResult

        X_train, X_test, y_train, y_test = X, None, y, None
        if test_size is not None:
            if y is None:
                X_train, X_test = split_data(X, test_size=test_size, random_state=random_state)
            else:
                X_train, X_test, y_train, y_test = split_data(
                    X, y, test_size=test_size, random_state=random_state
                )

        start = time.perf_counter()
        self.fit(X_train)
        fit_time = time.perf_counter() - start

        train = self.evaluate(X_train, y_train, metrics=metrics, verbose=False)
        test = None
        if X_test is not None:
            test = self.evaluate(X_test, y_test, metrics=metrics, verbose=False)

        info = {"params": self.model.get_params(), **self._extra_info()}
        if not hasattr(self.model, "predict"):
            info["asignacion_test"] = "vecino_mas_cercano"

        result = TrainTestResult(
            train,
            test,
            model_name=self.name,
            task="clustering",
            test_size=test_size,
            n_train=len(to_matrix(X_train)),
            n_test=None if X_test is None else len(to_matrix(X_test)),
            fit_time=round(fit_time, 4),
            info=info,
        )
        if verbose:
            _print_train_test(result)
        return result

    def _extra_info(self) -> dict:
        return {}

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

    def _extra_info(self) -> dict:
        return {"inercia": float(self.model.inertia_)}

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

    def _extra_info(self) -> dict:
        X = self._X_fit
        return {"bic": float(self.model.bic(X)), "aic": float(self.model.aic(X))}
