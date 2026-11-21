"""Baselines: similar-day naive and a LEAR-style LASSO model."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LassoLarsIC
from threadpoolctl import threadpool_limits

from gridwise.features import Features
from gridwise.frames import SLOTS


class SimilarDayNaive:
    """The standard naive benchmark of the price-forecasting literature.

    Tuesday to Friday repeat yesterday's price for the same hour; Monday, Saturday and Sunday repeat
    the price of the same hour a week earlier.
    """

    name = "naive"
    kind = "point"
    refit_every = 1  # nothing is fitted; a block of one day keeps the backtest loop simple
    train_days: int | None = None

    def fit(self, feats: Features, train_idx: np.ndarray) -> None:
        pass

    def predict(self, feats: Features, idx: np.ndarray) -> np.ndarray:
        d1, d7 = feats.lags[idx, 0], feats.lags[idx, 3]
        week_ago = np.isin(feats.dow[idx], (0, 5, 6))
        return np.where(week_ago[:, None], d7, d1)


class Lear:
    """LASSO-estimated autoregressive model in the style of LEAR (Lago et al., 2021).

    One model per hour. Inputs: all 24 prices of D-1, D-2, D-3 and D-7, weekday dummies and, if given,
    exogenous columns for the hour on day D. Prices pass through asinh to tame spikes. The penalty is
    chosen with AIC and the model is re-estimated on a rolling window. Differences from the published
    recipe: prices are normalised with the previous week's median and spread instead of the whole
    window's, and one window length is used instead of an average over four.
    """

    kind = "point"

    def __init__(
        self, exog: list[str] | None = None, train_days: int = 728, refit_every: int = 1, name: str = "lear"
    ):
        self.exog = exog or []
        self.train_days = train_days
        self.refit_every = refit_every
        self.name = name
        self._models: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    def _design(self, feats: Features, idx: np.ndarray, slot: int) -> np.ndarray:
        lags = np.arcsinh(feats.lags[idx].reshape(len(idx), -1))  # (k, 96)
        dow = np.eye(7)[feats.dow[idx]]
        parts = [lags, dow]
        if self.exog:
            cols = [feats.names.index(c) for c in self.exog]
            parts.append(feats.X[idx][:, slot, cols].astype(float))
        return np.concatenate(parts, axis=1)

    def fit(self, feats: Features, train_idx: np.ndarray) -> None:
        # The problems are tiny; BLAS threads only add overhead (2.5 s -> 1.2 s per refit on 8 cores).
        with threadpool_limits(limits=1):
            self._fit(feats, train_idx)

    def _fit(self, feats: Features, train_idx: np.ndarray) -> None:
        self._models = []
        for h in range(SLOTS):
            X = self._design(feats, train_idx, h)
            y = np.arcsinh(feats.y[train_idx, h])
            ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            X, y = X[ok], y[ok]
            mu, sd = X.mean(axis=0), X.std(axis=0)
            sd[sd < 1e-9] = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = LassoLarsIC(criterion="aic", max_iter=2500).fit((X - mu) / sd, y)
            self._models.append((mu, sd, m.coef_.copy(), float(m.intercept_)))

    def predict(self, feats: Features, idx: np.ndarray) -> np.ndarray:
        out = np.empty((len(idx), SLOTS))
        for h, (mu, sd, coef, b) in enumerate(self._models):
            X = np.nan_to_num((self._design(feats, idx, h) - mu) / sd)
            out[:, h] = np.sinh(X @ coef + b)
        return out
