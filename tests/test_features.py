import numpy as np
import pytest

from gridwise.features import LEVEL_DAYS, MIN_SCALE, build, level_and_scale
from gridwise.frames import Daily


def _copy(d: Daily) -> Daily:
    return Daily(d.days.copy(), {k: v.copy() for k, v in d.values.items()}, d.real.copy(), d.tz)


def test_features_of_a_day_ignore_that_day_and_later_prices(daily):
    """The core no-leak guarantee: rewriting prices from day D on must not change day D's features."""
    d = 200
    other = _copy(daily)
    other.values["price"][d:] = np.random.default_rng(1).normal(500, 100, other.values["price"][d:].shape)
    a, b = build(daily, "weather"), build(other, "weather")
    assert np.array_equal(a.X[d], b.X[d], equal_nan=True)
    assert np.array_equal(a.lags[d], b.lags[d], equal_nan=True)
    assert a.level[d] == b.level[d] and a.scale[d] == b.scale[d]
    assert not np.array_equal(a.X[d + 1], b.X[d + 1])  # the next day does see day D
    assert not np.array_equal(a.y[d], b.y[d])


def test_weather_features_ignore_later_days(daily):
    d = 200
    other = _copy(daily)
    for c in ("wx_wind_on_pow", "wx_solar", "wx_temp"):
        other.values[c][d + 1 :] += 5.0
    a, b = build(daily, "weather"), build(other, "weather")
    assert np.array_equal(a.X[d], b.X[d], equal_nan=True)


def test_level_and_scale_use_only_the_previous_week():
    p = np.arange(20 * 24, dtype=float).reshape(20, 24)
    m, s = level_and_scale(p)
    assert np.isnan(m[:LEVEL_DAYS]).all()
    window = p[3:10].ravel()
    assert m[10] == np.median(window)
    assert s[10] == pytest.approx(1.4826 * np.median(np.abs(window - np.median(window))))
    flat, _ = np.full((20, 24), 50.0), None
    assert level_and_scale(flat)[1][10] == MIN_SCALE


def test_normalisation_round_trip(feats):
    rows = np.where(feats.ready)[0][:5]
    assert np.allclose(feats.to_price(feats.y[rows], rows), feats.price[rows])
    q = np.repeat(feats.y[rows][..., None], 3, axis=-1)
    assert np.allclose(feats.to_price(q, rows)[..., 1], feats.price[rows])


def test_ready_days_have_no_missing_features(feats):
    assert feats.ready.sum() > 300
    assert not np.isnan(feats.X[feats.ready]).any()
    assert not feats.ready[:7].any()  # a week of history is needed first


def test_feature_sets(daily):
    price, leaky = build(daily, "price"), build(daily, "operator_leaky")
    assert not any(n.startswith(("wx_", "op_")) for n in price.names)
    assert "op_residual" in leaky.names and not any(n.startswith("wx_") for n in leaky.names)
    with pytest.raises(ValueError):
        build(daily, "nope")


def test_weekday_index(daily, feats):
    for i in (0, 50, 123):
        assert feats.dow[i] == daily.day(i).weekday()
