"""
Ensemble Engine: votación, stacking y blending de los mejores modelos.

El ensemble se evalúa con exactamente los mismos splits y la misma métrica
que los modelos individuales, y solo se elige si su media de validación
supera a la del mejor modelo individual. No se asume que sea superior.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted

from .exceptions import AutoMLError

KINDS = ("voting", "stacking", "blending")


class _BlendingMixin:
    """Lógica común del blending.

    Los miembros se entrenan con una parte del conjunto recibido y el
    meta-modelo aprende de sus predicciones sobre la parte reservada
    (``holdout_size``). A diferencia del stacking no hay validación interna
    con k folds: es más barato, pero el meta-modelo ve menos filas.
    """

    def _split(self, X, y, stratify):
        if not 0 < self.holdout_size < 1:
            raise AutoMLError("holdout_size debe estar entre 0 y 1.")
        if not self.shuffle:
            cut = int(round(len(y) * (1 - self.holdout_size)))
            idx = np.arange(len(y))
            return idx[:cut], idx[cut:]
        idx = np.arange(len(y))
        try:
            return train_test_split(idx, test_size=self.holdout_size,
                                    random_state=self.random_state,
                                    stratify=y if stratify else None)
        except ValueError:
            return train_test_split(idx, test_size=self.holdout_size,
                                    random_state=self.random_state)

    @staticmethod
    def _rows(X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    def _fit_members(self, X, y, train_idx):
        if len(self.estimators) < 2:
            raise AutoMLError("Un ensemble necesita al menos 2 modelos válidos.")
        X_fit, y_fit = self._rows(X, train_idx), np.asarray(y)[train_idx]
        self.estimators_ = [(name, clone(est).fit(X_fit, y_fit))
                            for name, est in self.estimators]


class BlendingClassifier(_BlendingMixin, ClassifierMixin, BaseEstimator):
    """Blending para clasificación.

    Parameters
    ----------
    estimators : list de (str, estimator)
        Modelos base (al menos dos).
    final_estimator : estimator, optional
        Meta-modelo; por defecto ``LogisticRegression(max_iter=1000)``.
    holdout_size : float, default=0.2
        Proporción reservada para entrenar el meta-modelo.
    shuffle : bool, default=True
        Si es False la reserva son las últimas filas (datos temporales).
    random_state : int, optional
    """

    def __init__(self, estimators, final_estimator=None, holdout_size=0.2,
                 shuffle=True, random_state=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_size = holdout_size
        self.shuffle = shuffle
        self.random_state = random_state

    def _meta_features(self, X):
        columns = []
        for _, est in self.estimators_:
            if hasattr(est, "predict_proba"):
                columns.append(est.predict_proba(X))
            elif hasattr(est, "decision_function"):
                scores = np.asarray(est.decision_function(X))
                columns.append(scores.reshape(len(scores), -1))
            else:
                pred = np.asarray(est.predict(X))
                columns.append((pred[:, None] == self.classes_[None, :]).astype(float))
        return np.hstack(columns)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        train_idx, hold_idx = self._split(X, y, stratify=True)
        self._fit_members(X, y, train_idx)
        meta = clone(self.final_estimator or LogisticRegression(max_iter=1000))
        meta.fit(self._meta_features(self._rows(X, hold_idx)), y[hold_idx])
        missing = set(self.classes_) - set(meta.classes_)
        if missing:
            raise AutoMLError(f"La reserva del blending no contiene las clases {missing}; "
                              "aumenta holdout_size o usa stacking.")
        self.final_estimator_ = meta
        return self

    def predict(self, X):
        check_is_fitted(self, "final_estimator_")
        return self.final_estimator_.predict(self._meta_features(X))

    def predict_proba(self, X):
        check_is_fitted(self, "final_estimator_")
        return self.final_estimator_.predict_proba(self._meta_features(X))


class BlendingRegressor(_BlendingMixin, RegressorMixin, BaseEstimator):
    """Blending para regresión (meta-modelo por defecto: ``RidgeCV``).

    Mismos parámetros que :class:`BlendingClassifier`.
    """

    def __init__(self, estimators, final_estimator=None, holdout_size=0.2,
                 shuffle=True, random_state=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_size = holdout_size
        self.shuffle = shuffle
        self.random_state = random_state

    def _meta_features(self, X):
        return np.column_stack([est.predict(X) for _, est in self.estimators_])

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        train_idx, hold_idx = self._split(X, y, stratify=False)
        self._fit_members(X, y, train_idx)
        meta = clone(self.final_estimator or RidgeCV())
        self.final_estimator_ = meta.fit(self._meta_features(self._rows(X, hold_idx)),
                                         y[hold_idx])
        return self

    def predict(self, X):
        check_is_fitted(self, "final_estimator_")
        return self.final_estimator_.predict(self._meta_features(X))


def build_ensemble(
    kind: str,
    task: str,
    members: Sequence[Tuple[str, Any]],
    random_state: Optional[int] = None,
    inner_cv: int = 5,
    shuffle: bool = True,
) -> Any:
    """Construye un ensemble (sin entrenar) a partir de ``(nombre, estimador)``.

    Parameters
    ----------
    kind : {"voting", "stacking", "blending"}
        - ``voting``: promedio de probabilidades (soft) si todos los
          miembros las exponen; si no, voto mayoritario (hard). En regresión,
          promedio de predicciones.
        - ``stacking``: un meta-modelo (LogisticRegression / RidgeCV)
          aprende de predicciones fuera de fold de los miembros.
        - ``blending``: el meta-modelo aprende de las predicciones sobre una
          reserva interna del 20 % de cada conjunto de entrenamiento.
    members : sequence of (str, estimator)
        Al menos dos modelos.
    inner_cv : int
        Folds internos del stacking (dentro de cada fold externo).
    shuffle : bool
        Con False (datos temporales) la reserva del blending son las
        últimas filas.
    """
    if kind not in KINDS:
        raise AutoMLError(f"ensemble debe ser uno de {KINDS} o None (recibido: {kind!r}).")
    if len(members) < 2:
        raise AutoMLError("Un ensemble necesita al menos 2 modelos válidos.")
    estimators: List[Tuple[str, Any]] = [(name, clone(est)) for name, est in members]

    if kind == "blending":
        cls = BlendingClassifier if task == "classification" else BlendingRegressor
        return cls(estimators, shuffle=shuffle, random_state=random_state)
    if task == "classification":
        if kind == "voting":
            soft = all(hasattr(est, "predict_proba") for _, est in estimators)
            return VotingClassifier(estimators, voting="soft" if soft else "hard")
        return StackingClassifier(
            estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(inner_cv, shuffle=True, random_state=random_state),
        )
    if kind == "voting":
        return VotingRegressor(estimators)
    return StackingRegressor(
        estimators,
        final_estimator=RidgeCV(),
        cv=KFold(inner_cv, shuffle=True, random_state=random_state),
    )
