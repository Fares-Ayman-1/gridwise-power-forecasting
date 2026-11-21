"""Leak-safe features for a delivery day.

Everything a row for delivery day D uses was known at the decision time, 07:30 UTC on D-1:

- prices of day D-1 and earlier (D-1 prices were published at about 13:00 local on D-2);
- weather forecasts for day D at a 48-hour lead (their runs started by 18:00 UTC on D-2);
- the calendar.

Prices move through regimes (in 2022 they were several times higher than in 2020), and tree models
cannot extrapolate. So prices are expressed relative to the recent market: for day D,
m = median and s = robust scale of the prices of days D-7..D-1, and every price p becomes (p - m) / s.
The target is normalised the same way and forecasts are mapped back with q * s + m.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import holidays
import numpy as np

from gridwise.frames import SLOTS, Daily

LEVEL_DAYS = 7
MIN_SCALE = 5.0  # EUR/MWh; keeps the scale away from zero in very flat weeks
MAX_LAG = 7

WEATHER = ["wx_wind_on", "wx_wind_on_pow", "wx_wind_off", "wx_wind_off_pow", "wx_solar", "wx_temp"]
OPERATOR = ["load_fc", "wind_onshore_fc", "wind_offshore_fc", "solar_fc"]

FEATURE_SETS = ("price", "weather", "operator_leaky")


@dataclass
class Features:
    days: np.ndarray  # datetime64[D], (n,)
    X: np.ndarray  # float32, (n, 24, F); NaN where a value is unknown
    y: np.ndarray  # float64, (n, 24): normalised target, NaN where the price is unknown
    price: np.ndarray  # float64, (n, 24): raw price
    level: np.ndarray  # float64, (n,): m
    scale: np.ndarray  # float64, (n,): s
    real: np.ndarray  # bool, (n, 24)
    ready: np.ndarray  # bool, (n,): every feature of the day is known
    names: list[str]
    lags: np.ndarray  # float64, (n, 4, 24): normalised prices of D-1, D-2, D-3, D-7 (for linear models)
    dow: np.ndarray  # int, (n,): 0 = Monday

    def to_price(self, q_norm: np.ndarray, rows: np.ndarray | slice) -> np.ndarray:
        """Normalised values of shape (k, 24, ...) -> EUR/MWh."""
        s = self.scale[rows].reshape(-1, *([1] * (q_norm.ndim - 1)))
        m = self.level[rows].reshape(-1, *([1] * (q_norm.ndim - 1)))
        return q_norm * s + m


def level_and_scale(price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and robust scale of the previous LEVEL_DAYS days, for every day. NaN until enough history."""
    n = len(price)
    m, s = np.full(n, np.nan), np.full(n, np.nan)
    for d in range(LEVEL_DAYS, n):
        w = price[d - LEVEL_DAYS : d].ravel()
        if np.isnan(w).any():
            continue
        m[d] = np.median(w)
        s[d] = max(1.4826 * np.median(np.abs(w - m[d])), MIN_SCALE)
    return m, s


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Row d of the result is row d-k of `a` (NaN for the first k rows)."""
    out = np.full_like(a, np.nan, dtype=float)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def german_holidays(days: np.ndarray) -> np.ndarray:
    years = sorted({int(str(d)[:4]) for d in days})
    cal = holidays.Germany(years=years)  # nationwide public holidays
    return np.array([d.astype(object) in cal for d in days])


def build(daily: Daily, feature_set: str = "weather") -> Features:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose one of {FEATURE_SETS}")
    P = daily["price"]
    n = len(daily)
    m, s = level_and_scale(P)

    def norm(a: np.ndarray) -> np.ndarray:
        """Normalise rows of `a` (already aligned to target day d) with day d's level and scale."""
        return (a - m[:, None]) / s[:, None]

    d1, d2, d3, d7 = (norm(_shift(P, k)) for k in (1, 2, 3, 7))
    week = np.nanmean(np.stack([norm(_shift(P, k)) for k in range(1, 8)]), axis=0)
    slot = np.broadcast_to(np.arange(SLOTS, dtype=float), (n, SLOTS))

    def per_day(v: np.ndarray) -> np.ndarray:
        return np.broadcast_to(v[:, None], (n, SLOTS))

    cols: dict[str, np.ndarray] = {
        "p_d1": d1,
        "p_d2": d2,
        "p_d3": d3,
        "p_d7": d7,
        "p_d1_prev_hour": np.concatenate([d1[:, :1], d1[:, :-1]], axis=1),
        "p_d1_next_hour": np.concatenate([d1[:, 1:], d1[:, -1:]], axis=1),
        "p_week_same_hour": week,
        "p_d1_mean": per_day(d1.mean(axis=1)),
        "p_d1_min": per_day(d1.min(axis=1)),
        "p_d1_max": per_day(d1.max(axis=1)),
        "p_d1_std": per_day(d1.std(axis=1)),
        "p_d1_last": per_day(d1[:, -1]),
        "p_d2_mean": per_day(d2.mean(axis=1)),
        "p_d7_mean": per_day(d7.mean(axis=1)),
        "level": per_day(m),
        "scale": per_day(s),
    }

    dow = (daily.days.astype("datetime64[D]").astype("int64") + 3) % 7  # 1970-01-01 was a Thursday
    hol = german_holidays(daily.days)
    doy = np.array([d.astype(object).timetuple().tm_yday for d in daily.days], dtype=float)
    cols |= {
        "slot": slot,
        "dow": per_day(dow.astype(float)),
        "is_weekend": per_day((dow >= 5).astype(float)),
        "is_holiday": per_day(hol.astype(float)),
        "holiday_yesterday": per_day(_shift(hol.astype(float)[:, None], 1)[:, 0]),
        "doy_sin": per_day(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": per_day(np.cos(2 * np.pi * doy / 365.25)),
    }

    need_all = ["p_d1", "p_d2", "p_d3", "p_d7"]
    if feature_set == "weather":
        for c in WEATHER:
            w = daily[c]
            cols[c] = w
            cols[f"{c}_chg_d1"] = w - _shift(w, 1)
        for c in ("wx_wind_on_pow", "wx_wind_off_pow", "wx_solar", "wx_temp"):
            cols[f"{c}_day_mean"] = per_day(daily[c].mean(axis=1))
        need_all += WEATHER
    elif feature_set == "operator_leaky":
        load, won, woff, sol = (daily[c] for c in OPERATOR)
        resid = load - won - woff - sol
        cols |= {"op_load": load, "op_wind": won + woff, "op_solar": sol, "op_residual": resid}
        cols["op_residual_chg_d1"] = resid - _shift(resid, 1)
        cols["op_residual_day_mean"] = per_day(resid.mean(axis=1))
        need_all += ["op_load", "op_wind", "op_solar"]

    names = list(cols)
    X = np.stack([cols[k] for k in names], axis=-1).astype(np.float32)
    ready = ~np.isnan(m)
    for k in need_all:
        ready &= ~np.isnan(cols[k]).any(axis=1)
    return Features(
        days=daily.days,
        X=X,
        y=norm(P),
        price=P,
        level=m,
        scale=s,
        real=daily.real,
        ready=ready,
        names=names,
        lags=np.stack([d1, d2, d3, d7], axis=1),
        dow=dow,
    )


def feature_days(feats: Features, start: date, end: date) -> np.ndarray:
    """Indices of ready days in [start, end]."""
    lo, hi = np.datetime64(start), np.datetime64(end)
    return np.where(feats.ready & (feats.days >= lo) & (feats.days <= hi))[0]
