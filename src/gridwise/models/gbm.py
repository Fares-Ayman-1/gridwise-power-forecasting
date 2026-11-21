"""Quantile gradient boosting: one LightGBM model per quantile level, all hours pooled."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from gridwise.features import Features
from gridwise.frames import SLOTS

DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 6,
    "seed": 42,
    "verbose": -1,
}


class QuantileGBM:
    kind = "quantile"

    def __init__(
        self,
        taus: list[float],
        params: dict[str, Any] | None = None,
        num_boost_round: int = 200,
        train_days: int | None = 1095,
        refit_every: int = 28,
        name: str = "qgbm",
    ):
        self.taus = list(taus)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.train_days = train_days
        self.refit_every = refit_every
        self.name = name
        self._boosters: list[lgb.Booster] = []
        self.feature_names: list[str] = []

    @staticmethod
    def _flat(feats: Features, idx: np.ndarray) -> np.ndarray:
        return feats.X[idx].reshape(len(idx) * SLOTS, -1)

    def fit(self, feats: Features, train_idx: np.ndarray) -> None:
        X = self._flat(feats, train_idx)
        y = feats.y[train_idx].reshape(-1)
        ok = ~np.isnan(y) & feats.real[train_idx].reshape(-1)
        self.feature_names = list(feats.names)
        self._boosters = []
        for tau in self.taus:
            # hour and weekday stay numeric: ordered splits are enough and much faster than categorical ones
            data = lgb.Dataset(X[ok], y[ok], feature_name=self.feature_names, categorical_feature=[])
            params = {**self.params, "objective": "quantile", "alpha": tau}
            self._boosters.append(lgb.train(params, data, num_boost_round=self.num_boost_round))

    def predict(self, feats: Features, idx: np.ndarray) -> np.ndarray:
        X = self._flat(feats, idx)
        q = np.stack([b.predict(X) for b in self._boosters], axis=-1)
        return np.sort(q, axis=-1).reshape(len(idx), SLOTS, len(self.taus))

    def importance(self) -> dict[str, float]:
        """Mean gain share per feature across the quantile models."""
        total = np.zeros(len(self.feature_names))
        for b in self._boosters:
            g = b.feature_importance(importance_type="gain")
            total += g / max(g.sum(), 1e-12)
        return dict(zip(self.feature_names, (total / len(self._boosters)).tolist(), strict=True))
