"""
Modelos bayesianos supervisados.

Wrappers sobre scikit-learn (sin dependencias nuevas):
- ``BayesianRidgeRegression`` : regresión lineal bayesiana con incertidumbre.
- ``ARDRegression``           : regresión bayesiana con selección automática
  de relevancia (poda features irrelevantes).
- ``GaussianProcessRegressor``: proceso gaussiano con intervalos de confianza.
- ``GaussianProcessClassifier``: clasificación probabilística no paramétrica.
- ``MultinomialNB`` / ``BernoulliNB`` / ``ComplementNB``: variantes de
  Naive Bayes para conteos, binarias y clases desbalanceadas.

Inferencia bayesiana completa con PyMC (``pip install synaptix[bayes]``):
- ``PyMCLinearRegression``   : regresión lineal con posteriores MCMC.
- ``PyMCLogisticRegression`` : clasificación binaria con posteriores MCMC.

Ejemplo
-------
>>> from synaptix.supervised import BayesianRidgeRegression
>>> model = BayesianRidgeRegression()
>>> model.fit(X_train, y_train)
>>> media, inferior, superior = model.predict_interval(X_test, std=2)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sklearn.gaussian_process import (
    GaussianProcessClassifier as _SkGPC,
    GaussianProcessRegressor as _SkGPR,
)
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import (
    ARDRegression as _SkARD,
    BayesianRidge as _SkBayesianRidge,
)
from sklearn.naive_bayes import (
    BernoulliNB as _SkBernoulliNB,
    ComplementNB as _SkComplementNB,
    MultinomialNB as _SkMultinomialNB,
)

from ..base import ArrayLike, SynaptixModel, to_matrix, to_vector


# ======================================================================
# Wrappers de scikit-learn
# ======================================================================


class _UncertainRegression(SynaptixModel):
    """Base para regresores bayesianos que exponen desviación predictiva."""

    task = "regression"

    def predict_interval(
        self, X: ArrayLike, std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicción con intervalo de incertidumbre.

        Parameters
        ----------
        X : array-like
            Features.
        std : float, default=2.0
            Número de desviaciones estándar del intervalo
            (2.0 equivale a ~95% de confianza).

        Returns
        -------
        tuple de ndarray
            ``(media, inferior, superior)``.
        """
        self._check_fitted()
        mean, deviation = self.model.predict(to_matrix(X), return_std=True)
        return mean, mean - std * deviation, mean + std * deviation


class BayesianRidgeRegression(_UncertainRegression):
    """Regresión lineal bayesiana (Bayesian Ridge).

    Estima distribuciones sobre los coeficientes en lugar de valores
    puntuales; la regularización se aprende automáticamente de los datos.

    Parameters
    ----------
    max_iter : int, default=300
        Iteraciones máximas del ajuste.
    **kwargs
        Parámetros de ``sklearn.linear_model.BayesianRidge``
        (``alpha_1``, ``alpha_2``, ``lambda_1``, ``lambda_2``...).

    Ejemplo
    -------
    >>> model = BayesianRidgeRegression()
    >>> model.fit(X_train, y_train)
    >>> media, low, high = model.predict_interval(X_test)
    """

    def __init__(self, max_iter: int = 300, **kwargs):
        super().__init__(_SkBayesianRidge(max_iter=max_iter, **kwargs))


class ARDRegression(_UncertainRegression):
    """Regresión bayesiana con determinación automática de relevancia.

    Similar a :class:`BayesianRidgeRegression` pero con una precisión
    por coeficiente: las features irrelevantes tienden a cero
    (selección de variables automática).

    Parameters
    ----------
    max_iter : int, default=300
        Iteraciones máximas del ajuste.
    """

    def __init__(self, max_iter: int = 300, **kwargs):
        super().__init__(_SkARD(max_iter=max_iter, **kwargs))


class GaussianProcessRegressor(_UncertainRegression):
    """Proceso gaussiano para regresión (bayesiano no paramétrico).

    Ideal para datasets pequeños donde importa la incertidumbre:
    el intervalo se ensancha lejos de los datos de entrenamiento.

    Parameters
    ----------
    kernel : kernel de sklearn, optional
        Por defecto ``ConstantKernel * RBF + WhiteKernel`` (suavidad
        con ruido aprendido de los datos).
    normalize_y : bool, default=True
        Normaliza el objetivo internamente.

    Notas
    -----
    Escala O(n³) con el número de muestras; no recomendado para
    datasets con más de unas ~10 000 filas.
    """

    def __init__(self, kernel=None, normalize_y: bool = True, random_state: int = 42, **kwargs):
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
                noise_level=1e-2
            )
        super().__init__(
            _SkGPR(
                kernel=kernel,
                normalize_y=normalize_y,
                random_state=random_state,
                **kwargs,
            )
        )


class GaussianProcessClassifier(SynaptixModel):
    """Proceso gaussiano para clasificación probabilística.

    Parameters
    ----------
    kernel : kernel de sklearn, optional
        Por defecto ``ConstantKernel * RBF``.
    """

    task = "classification"

    def __init__(self, kernel=None, random_state: int = 42, **kwargs):
        if kernel is None:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        super().__init__(_SkGPC(kernel=kernel, random_state=random_state, **kwargs))


class MultinomialNB(SynaptixModel):
    """Naive Bayes multinomial (conteos: texto, frecuencias).

    Requiere features no negativas.

    Parameters
    ----------
    alpha : float, default=1.0
        Suavizado de Laplace/Lidstone.
    """

    task = "classification"

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(_SkMultinomialNB(alpha=alpha, **kwargs))


class BernoulliNB(SynaptixModel):
    """Naive Bayes para features binarias (presencia/ausencia).

    Parameters
    ----------
    alpha : float, default=1.0
        Suavizado.
    binarize : float, default=0.0
        Umbral para binarizar las features (None si ya son binarias).
    """

    task = "classification"

    def __init__(self, alpha: float = 1.0, binarize: float = 0.0, **kwargs):
        super().__init__(_SkBernoulliNB(alpha=alpha, binarize=binarize, **kwargs))


class ComplementNB(SynaptixModel):
    """Naive Bayes complementario (robusto a clases desbalanceadas).

    Requiere features no negativas.

    Parameters
    ----------
    alpha : float, default=1.0
        Suavizado.
    """

    task = "classification"

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(_SkComplementNB(alpha=alpha, **kwargs))


# ======================================================================
# Inferencia bayesiana completa con PyMC (opcional)
# ======================================================================


def _require_pymc():
    """Importa PyMC y ArviZ con un error útil si no están instalados."""
    try:
        import arviz as az
        import pymc as pm
    except ImportError as error:
        raise ImportError(
            "Este modelo requiere PyMC. Instala con: pip install synaptix[bayes]"
        ) from error
    return pm, az


class _PyMCModel:
    """Base común para modelos PyMC: estandarización, resumen y gráficos."""

    def __init__(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        random_seed: int = 42,
    ):
        _require_pymc()  # falla temprano si no está instalado
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.random_seed = random_seed

        self.idata = None
        self.fitted = False
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        return (X - self._x_mean) / self._x_std

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def _posterior(self, name: str) -> np.ndarray:
        """Muestras aplanadas (draws*chains, ...) de un parámetro."""
        values = self.idata.posterior[name].values
        return values.reshape(-1, *values.shape[2:])

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("Llama a fit(X, y) primero.")

    def summary(self):
        """Tabla resumen de las distribuciones posteriores (media, HDI, r_hat)."""
        _, az = _require_pymc()
        self._check_fitted()
        return az.summary(self.idata)

    def plot_posterior(self):
        """Grafica las distribuciones posteriores de los parámetros."""
        import matplotlib.pyplot as plt

        _, az = _require_pymc()
        self._check_fitted()
        az.plot_posterior(self.idata)
        plt.tight_layout()
        plt.show()


class PyMCLinearRegression(_PyMCModel):
    """Regresión lineal bayesiana con inferencia MCMC (PyMC).

    Modelo::

        y ~ Normal(intercepto + X @ beta, sigma)

    con priors débilmente informativos. A diferencia de los wrappers de
    sklearn, aquí se obtiene la distribución posterior completa de cada
    parámetro.

    Parameters
    ----------
    draws : int, default=1000
        Muestras MCMC por cadena.
    tune : int, default=1000
        Muestras de calentamiento (descartadas).
    chains : int, default=2
        Cadenas independientes.
    random_seed : int, default=42
        Semilla de reproducibilidad.

    Ejemplo
    -------
    >>> model = PyMCLinearRegression(draws=1000)
    >>> model.fit(X_train, y_train)
    >>> media = model.predict(X_test)
    >>> media, low, high = model.predict_interval(X_test, hdi=0.94)
    >>> model.summary()
    """

    def fit(self, X: ArrayLike, y: ArrayLike, progressbar: bool = False):
        """Ajusta el modelo muestreando la distribución posterior."""
        pm, _ = _require_pymc()

        X = to_matrix(X).astype(float)
        y = to_vector(y).astype(float)
        X_std = self._standardize_fit(X)

        with pm.Model():
            intercept = pm.Normal("intercepto", mu=float(y.mean()), sigma=10.0)
            betas = pm.Normal("beta", mu=0.0, sigma=10.0, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", sigma=float(y.std() or 1.0))

            mu = intercept + pm.math.dot(X_std, betas)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            self.idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_seed,
                progressbar=progressbar,
            )

        self.fitted = True
        return self

    def _mu_samples(self, X: ArrayLike) -> np.ndarray:
        """Muestras posteriores de la media predicha: (n_muestras_mcmc, n_filas)."""
        X_std = self._standardize(to_matrix(X).astype(float))
        intercept = self._posterior("intercepto")          # (s,)
        betas = self._posterior("beta")                    # (s, p)
        return intercept[:, None] + betas @ X_std.T        # (s, n)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Media posterior de la predicción."""
        self._check_fitted()
        return self._mu_samples(X).mean(axis=0)

    def predict_interval(
        self, X: ArrayLike, hdi: float = 0.94
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicción con intervalo creíble.

        Parameters
        ----------
        X : array-like
            Features.
        hdi : float, default=0.94
            Masa de probabilidad del intervalo (0.94 = 94%).

        Returns
        -------
        tuple de ndarray
            ``(media, inferior, superior)``.
        """
        self._check_fitted()
        samples = self._mu_samples(X)
        tail = (1 - hdi) / 2
        lower = np.quantile(samples, tail, axis=0)
        upper = np.quantile(samples, 1 - tail, axis=0)
        return samples.mean(axis=0), lower, upper

    def evaluate(self, X: ArrayLike, y: ArrayLike, verbose: bool = True) -> dict:
        """Métricas de regresión sobre la media posterior."""
        from .. import metrics as sx_metrics

        results = sx_metrics.regression_metrics(to_vector(y), self.predict(X))
        if verbose:
            print("\n=== Evaluación: PyMCLinearRegression ===")
            for key, value in results.items():
                print(f"  {key:<12}: {value:.4f}")
        return results


class PyMCLogisticRegression(_PyMCModel):
    """Regresión logística bayesiana binaria con MCMC (PyMC).

    Modelo::

        y ~ Bernoulli(sigmoid(intercepto + X @ beta))

    Parameters
    ----------
    draws, tune, chains, random_seed
        Igual que :class:`PyMCLinearRegression`.

    Ejemplo
    -------
    >>> model = PyMCLogisticRegression()
    >>> model.fit(X_train, y_train)          # etiquetas binarias
    >>> probas = model.predict_proba(X_test)
    >>> clases = model.predict(X_test)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike, progressbar: bool = False):
        """Ajusta el clasificador (solo problemas binarios)."""
        pm, _ = _require_pymc()

        X = to_matrix(X).astype(float)
        y_raw = to_vector(y)
        self._classes = np.unique(y_raw)
        if len(self._classes) != 2:
            raise ValueError(
                f"PyMCLogisticRegression es binaria; se recibieron "
                f"{len(self._classes)} clases."
            )
        y_bin = (y_raw == self._classes[1]).astype(int)
        X_std = self._standardize_fit(X)

        with pm.Model():
            intercept = pm.Normal("intercepto", mu=0.0, sigma=5.0)
            betas = pm.Normal("beta", mu=0.0, sigma=5.0, shape=X.shape[1])

            logits = intercept + pm.math.dot(X_std, betas)
            pm.Bernoulli("y_obs", logit_p=logits, observed=y_bin)

            self.idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_seed,
                progressbar=progressbar,
            )

        self.fitted = True
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Probabilidad posterior media de cada clase: (n, 2)."""
        self._check_fitted()
        X_std = self._standardize(to_matrix(X).astype(float))
        intercept = self._posterior("intercepto")
        betas = self._posterior("beta")
        logits = intercept[:, None] + betas @ X_std.T
        positive = (1 / (1 + np.exp(-logits))).mean(axis=0)
        return np.column_stack([1 - positive, positive])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Clase más probable (en las etiquetas originales)."""
        positive = self.predict_proba(X)[:, 1]
        return np.where(positive > 0.5, self._classes[1], self._classes[0])

    def evaluate(self, X: ArrayLike, y: ArrayLike, verbose: bool = True) -> dict:
        """Métricas de clasificación sobre la clase más probable."""
        from .. import metrics as sx_metrics

        results = sx_metrics.classification_metrics(to_vector(y), self.predict(X))
        if verbose:
            print("\n=== Evaluación: PyMCLogisticRegression ===")
            for key, value in results.items():
                print(f"  {key:<12}: {value:.4f}")
        return results
