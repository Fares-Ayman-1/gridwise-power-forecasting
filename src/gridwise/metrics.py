"""Forecast metrics. All take arrays of shape (n_days, 24[, Q]) and an optional boolean mask."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _mask(y: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    ok = ~np.isnan(y)
    return ok if mask is None else ok & mask


def pinball(q: np.ndarray, y: np.ndarray, taus: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean pinball loss for each quantile level; shape (Q,)."""
    ok = _mask(y, mask)
    diff = y[ok][:, None] - q[ok]
    return np.mean(np.maximum(taus * diff, (taus - 1) * diff), axis=0)


def tau_weights(taus: np.ndarray) -> np.ndarray:
    """Trapezoid weights that turn values on the tau grid into an integral over (0, 1); they sum to 1.

    The mass outside the outermost levels is given to those levels.
    """
    edges = np.concatenate([[0.0], (taus[:-1] + taus[1:]) / 2, [1.0]])
    return np.diff(edges)


def crps(q: np.ndarray, y: np.ndarray, taus: np.ndarray, mask: np.ndarray | None = None) -> float:
    """CRPS approximated on the quantile grid: 2 x the tau-integral of the pinball loss."""
    return float(2.0 * np.sum(tau_weights(taus) * pinball(q, y, taus, mask)))


def coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    ok = _mask(y, mask)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


def width(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    ok = _mask(y, mask)
    return float(np.mean(hi[ok] - lo[ok]))


def mae(point: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    ok = _mask(y, mask)
    return float(np.mean(np.abs(y[ok] - point[ok])))


def rmse(point: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    ok = _mask(y, mask)
    return float(np.sqrt(np.mean((y[ok] - point[ok]) ** 2)))


def daily_abs_error(point: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean absolute error of each day; shape (n_days,)."""
    err = np.abs(y - point)
    ok = ~np.isnan(err) if mask is None else (~np.isnan(err) & mask)
    return np.where(ok, err, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)


def daily_pinball(
    q: np.ndarray, y: np.ndarray, taus: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Tau-weighted pinball loss of each day (half the day's CRPS); shape (n_days,)."""
    diff = y[..., None] - q
    loss = np.sum(tau_weights(taus) * np.maximum(taus * diff, (taus - 1) * diff), axis=-1)
    ok = ~np.isnan(loss) if mask is None else (~np.isnan(loss) & mask)
    return np.where(ok, loss, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """One-sided Diebold-Mariano test on daily losses.

    Returns (statistic, p-value) for the null "model A is not more accurate than model B". A small
    p-value means A's daily loss is significantly lower. Day-ahead forecasts are one step ahead, so the
    loss differences are treated as serially uncorrelated, as in Lago et al. (2021).
    """
    d = loss_a - loss_b
    d = d[~np.isnan(d)]
    stat = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))
    return stat, float(stats.norm.cdf(stat))


def paired_bootstrap(
    diff: np.ndarray, n_boot: int = 10_000, seed: int = 0, block: int = 7
) -> tuple[float, float]:
    """95% interval for the mean of a daily difference, by moving-block bootstrap (blocks of `block` days)."""
    rng = np.random.default_rng(seed)
    d = diff[~np.isnan(diff)]
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
