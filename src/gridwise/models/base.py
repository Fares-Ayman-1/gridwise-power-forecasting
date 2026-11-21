"""Model interface shared by every forecaster.

Models work in normalised price units on the day x slot grid:

    fit(feats, train_idx)            train on the given day indices
    predict(feats, idx) -> array     (len(idx), 24) for point models, (len(idx), 24, Q) for quantile models
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from gridwise.features import Features


class Forecaster(Protocol):
    name: str
    kind: str  # "point" or "quantile"
    refit_every: int  # days between refits in a backtest
    train_days: int | None  # length of the rolling training window; None = all history

    def fit(self, feats: Features, train_idx: np.ndarray) -> None: ...

    def predict(self, feats: Features, idx: np.ndarray) -> np.ndarray: ...
