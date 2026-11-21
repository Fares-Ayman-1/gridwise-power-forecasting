"""Conformal calibration of quantile forecasts, run online one delivery day at a time.

For each quantile level tau the calibrator keeps the recent errors e = y - q_tau and shifts the next
forecast by the empirical tau-quantile of those errors, so that the shifted quantile would have had
coverage tau on the window. A point forecast is handled the same way: every level starts from the
point forecast, and the shifts are then simply the empirical quantiles of its recent errors.

Two modes:

- "rolling": the level used is tau itself (split-conformal on a sliding window).
- "adaptive": adaptive conformal inference (Gibbs and Candes, 2021). After each day the level moves
  towards whatever would have given the right coverage: tau_eff += gamma * (tau - observed share
  below the quantile). When the market shifts and quantiles start to miss, the level moves further
  out without waiting for the window to fill with new errors.

Everything is done in normalised price units, so errors from calm and volatile weeks are comparable.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class OnlineCalibrator:
    def __init__(self, taus: np.ndarray, window_days: int = 90, mode: str = "rolling", gamma: float = 0.01):
        if mode not in ("none", "rolling", "adaptive"):
            raise ValueError(f"unknown mode {mode!r}")
        self.taus = np.asarray(taus, dtype=float)
        self.window_days = window_days
        self.mode = mode
        self.gamma = gamma
        self.tau_eff = self.taus.copy()
        self._errors: deque[np.ndarray] = deque(maxlen=window_days)  # each (hours, Q)
        self.min_days = 14

    def shifts(self) -> np.ndarray:
        """Shift to add to each raw quantile; zeros until there is enough history."""
        if self.mode == "none" or len(self._errors) < self.min_days:
            return np.zeros_like(self.taus)
        e = np.concatenate(list(self._errors), axis=0)  # (n, Q)
        n = len(e)
        out = np.empty_like(self.taus)
        for j, (tau, eff) in enumerate(zip(self.taus, self.tau_eff, strict=True)):
            # finite-sample correction pushes both tails outwards (conservative)
            level = np.ceil((n + 1) * eff) / n if tau > 0.5 else np.floor((n + 1) * eff) / n
            if tau == 0.5:
                level = eff
            out[j] = np.quantile(e[:, j], float(np.clip(level, 0.0, 1.0)))
        return out

    def apply(self, q_raw: np.ndarray) -> np.ndarray:
        """Calibrate one day's raw quantiles, shape (24, Q). The result is sorted so levels cannot cross."""
        return np.sort(q_raw + self.shifts(), axis=-1)

    def update(
        self, q_raw: np.ndarray, q_cal: np.ndarray, y: np.ndarray, real: np.ndarray | None = None
    ) -> None:
        """Feed the outcome of a day. `q_cal` is what `apply` returned for it."""
        ok = ~np.isnan(y) if real is None else (~np.isnan(y) & real)
        if not ok.any():
            return
        self._errors.append((y[ok][:, None] - q_raw[ok]).astype(float))
        if self.mode == "adaptive":
            below = np.mean(y[ok][:, None] < q_cal[ok], axis=0)  # should equal tau
            self.tau_eff = np.clip(self.tau_eff + self.gamma * (self.taus - below), 0.001, 0.999)


def calibrate_sequence(
    q_raw: np.ndarray,
    y: np.ndarray,
    taus: np.ndarray,
    real: np.ndarray | None = None,
    window_days: int = 90,
    mode: str = "rolling",
    gamma: float = 0.01,
) -> np.ndarray:
    """Calibrate consecutive days in order; day i only uses days before i. Shapes (n, 24, Q) and (n, 24)."""
    cal = OnlineCalibrator(taus, window_days, mode, gamma)
    out = np.empty_like(q_raw, dtype=float)
    for i in range(len(q_raw)):
        out[i] = cal.apply(q_raw[i])
        cal.update(q_raw[i], out[i], y[i], None if real is None else real[i])
    return out


def point_to_quantiles(point: np.ndarray, n_levels: int) -> np.ndarray:
    """(n, 24) -> (n, 24, Q) with every level equal to the point forecast, ready for calibration."""
    return np.repeat(point[..., None], n_levels, axis=-1)
