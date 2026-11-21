"""From 24 marginal quantile forecasts to joint price scenarios for a whole day.

Battery profit depends on price differences between hours, so the scenarios must keep the dependence
between hours: when the morning turns out more expensive than forecast, so does the evening, usually.
A Gaussian copula does that. For each past day the outcome of every hour is turned into its
probability level under that day's forecast (the PIT), then into a normal score; the correlation of
those scores across hours is estimated on a recent window. New scenarios are correlated normal
scores mapped back through tomorrow's quantile forecasts.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

U_MIN = 1e-3


def quantile_function(q: np.ndarray, taus: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Value at probability level u for each hour. q: (H, Q), u: (S, H) -> (S, H).

    Between the forecast levels the quantile function is linear; outside them it continues with the
    slope of the outermost pair of levels.
    """
    out = np.empty_like(u, dtype=float)
    for h in range(q.shape[0]):
        qh = q[h]
        v = np.interp(u[:, h], taus, qh)
        lo, hi = u[:, h] < taus[0], u[:, h] > taus[-1]
        slope_lo = (qh[1] - qh[0]) / (taus[1] - taus[0])
        slope_hi = (qh[-1] - qh[-2]) / (taus[-1] - taus[-2])
        v[lo] = qh[0] - slope_lo * (taus[0] - u[lo, h])
        v[hi] = qh[-1] + slope_hi * (u[hi, h] - taus[-1])
        out[:, h] = v
    return out


def pit(q: np.ndarray, taus: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability level of each outcome under its forecast. q: (..., Q), y: (...) -> (...) in (0, 1)."""
    flat_q, flat_y = q.reshape(-1, q.shape[-1]), y.reshape(-1)
    u = np.full(flat_y.shape, np.nan)
    for i, (qi, yi) in enumerate(zip(flat_q, flat_y, strict=True)):
        if np.isnan(yi):
            continue
        if yi < qi[0]:
            slope = (qi[1] - qi[0]) / (taus[1] - taus[0])
            u[i] = taus[0] - (qi[0] - yi) / slope if slope > 0 else U_MIN
        elif yi > qi[-1]:
            slope = (qi[-1] - qi[-2]) / (taus[-1] - taus[-2])
            u[i] = taus[-1] + (yi - qi[-1]) / slope if slope > 0 else 1 - U_MIN
        else:
            # strictly increasing grid for interp; ties resolve to the middle of the flat stretch
            u[i] = np.interp(yi, qi + np.arange(len(qi)) * 1e-9, taus)
    return np.clip(u, U_MIN, 1 - U_MIN).reshape(y.shape)


def copula_correlation(u_hist: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Correlation of normal scores across hours from past PITs (n_days, H), shrunk towards identity."""
    z = stats.norm.ppf(u_hist[~np.isnan(u_hist).any(axis=1)])
    r = np.corrcoef(z, rowvar=False)
    r = np.nan_to_num(r, nan=0.0)
    np.fill_diagonal(r, 1.0)
    return (1 - shrink) * r + shrink * np.eye(r.shape[0])


def sample(q: np.ndarray, taus: np.ndarray, corr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n joint scenarios for one day. q: (H, Q) -> (n, H). Antithetic pairs halve the sampling noise."""
    half = (n + 1) // 2
    z = rng.multivariate_normal(np.zeros(len(corr)), corr, size=half, method="cholesky")
    z = np.concatenate([z, -z])[:n]
    return quantile_function(q, taus, stats.norm.cdf(z))


def forecast_mean(q: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Expected value implied by the quantiles: integral of the quantile function over (0, 1).

    Uses the same piecewise-linear quantile function as `sample`, so the two agree.
    """
    grid = np.linspace(0.0025, 0.9975, 200)  # midpoints of 200 equal-probability bins
    flat = q.reshape(-1, q.shape[-1])
    u = np.broadcast_to(grid[:, None], (len(grid), len(flat)))
    return quantile_function(flat, taus, u).mean(axis=0).reshape(q.shape[:-1])
