import numpy as np
import pytest

from gridwise.conformal import OnlineCalibrator, calibrate_sequence, point_to_quantiles

TAUS = np.array([0.05, 0.1, 0.5, 0.9, 0.95])


def _coverage(q, y, lo, hi):
    return float(np.mean((y >= q[..., lo]) & (y <= q[..., hi])))


def test_too_narrow_quantiles_are_widened_to_nominal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, (400, 24))
    narrow = np.broadcast_to(np.array([-0.5, -0.4, 0.0, 0.4, 0.5]), (400, 24, 5)).copy()
    assert _coverage(narrow[100:], y[100:], 0, 4) < 0.45
    for mode in ("rolling", "adaptive"):
        cal = calibrate_sequence(narrow, y, TAUS, mode=mode)
        assert _coverage(cal[100:], y[100:], 0, 4) == pytest.approx(0.90, abs=0.02)
        assert _coverage(cal[100:], y[100:], 1, 3) == pytest.approx(0.80, abs=0.02)


def test_point_forecast_gets_quantiles_from_its_own_errors():
    rng = np.random.default_rng(1)
    y = 3.0 + rng.normal(0, 2, (300, 24))  # the point forecast 0 is biased by 3
    cal = calibrate_sequence(point_to_quantiles(np.zeros((300, 24)), 5), y, TAUS)
    assert np.median(cal[100:, :, 2]) == pytest.approx(3.0, abs=0.3)  # bias removed
    assert _coverage(cal[100:], y[100:], 0, 4) == pytest.approx(0.90, abs=0.03)


def test_adaptive_mode_recovers_coverage_faster_after_a_volatility_jump():
    rng = np.random.default_rng(2)
    sd = np.r_[np.ones(200), np.full(40, 3.0)]
    y = rng.normal(0, 1, (240, 24)) * sd[:, None]
    q = np.broadcast_to(np.array([-1.645, -1.2816, 0.0, 1.2816, 1.645]), (240, 24, 5)).copy()
    after = slice(205, 240)
    rolling = _coverage(calibrate_sequence(q, y, TAUS, mode="rolling")[after], y[after], 0, 4)
    adaptive = _coverage(calibrate_sequence(q, y, TAUS, mode="adaptive", gamma=0.05)[after], y[after], 0, 4)
    assert adaptive > rolling


def test_calibration_never_looks_ahead():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, (120, 24))
    q = np.broadcast_to(np.array([-1.0, -0.8, 0.0, 0.8, 1.0]), (120, 24, 5)).copy()
    a = calibrate_sequence(q, y, TAUS, mode="adaptive")
    y2 = y.copy()
    y2[80:] += 100.0
    b = calibrate_sequence(q, y2, TAUS, mode="adaptive")
    assert np.array_equal(a[:81], b[:81])  # day 80 is calibrated before its outcome is seen
    assert not np.array_equal(a[81:], b[81:])


def test_quantiles_never_cross_and_warm_up_leaves_forecasts_unchanged():
    cal = OnlineCalibrator(TAUS)
    q = np.tile(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), (24, 1))
    assert np.array_equal(cal.apply(q), q)  # no history yet
    rng = np.random.default_rng(4)
    for _ in range(40):
        out = cal.apply(q)
        cal.update(q, out, rng.standard_t(3, 24))
        assert (np.diff(out, axis=-1) >= 0).all()


def test_skipped_hours_are_ignored():
    cal = OnlineCalibrator(TAUS)
    q = np.zeros((24, 5))
    y = np.zeros(24)
    real = np.ones(24, dtype=bool)
    real[2] = False
    y[2] = 1e6
    for _ in range(20):
        cal.update(q, q, y, real)
    assert np.allclose(cal.shifts(), 0.0)


def test_unknown_mode():
    with pytest.raises(ValueError):
        OnlineCalibrator(TAUS, mode="magic")
