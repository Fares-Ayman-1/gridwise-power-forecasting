import numpy as np
import pytest
from scipy import stats

from gridwise import metrics as M
from gridwise.scenarios import copula_correlation, forecast_mean, pit, quantile_function, sample

TAUS = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975])


def _normal_q(mu, sd):
    return mu[:, None] + sd[:, None] * stats.norm.ppf(TAUS)[None, :]


def test_quantile_function_hits_the_forecast_levels_and_extends_the_tails():
    q = _normal_q(np.array([50.0, 80.0]), np.array([10.0, 20.0]))
    u = np.tile(TAUS[:, None], (1, 2))
    assert np.allclose(quantile_function(q, TAUS, u), q.T)
    tails = quantile_function(q, TAUS, np.array([[0.001, 0.999]]))
    assert tails[0, 0] < q[0, 0] and tails[0, 1] > q[1, -1]


def test_pit_inverts_the_quantile_function():
    q = _normal_q(np.array([50.0, 80.0]), np.array([10.0, 20.0]))
    u = np.array([[0.01, 0.33], [0.5, 0.77], [0.9, 0.99]])
    y = quantile_function(q, TAUS, u)  # (3, 2)
    back = pit(np.broadcast_to(q, (3, 2, len(TAUS))), TAUS, y)
    assert np.allclose(back, u, atol=1e-6)


def test_scenarios_keep_the_marginals_and_the_dependence():
    rng = np.random.default_rng(0)
    q = _normal_q(np.full(24, 60.0), np.full(24, 15.0))
    corr = 0.8 * np.ones((24, 24)) + 0.2 * np.eye(24)
    s = sample(q, TAUS, corr, 4000, rng)
    assert s.shape == (4000, 24)
    assert np.median(s, axis=0) == pytest.approx(np.full(24, 60.0), abs=1.0)
    assert np.quantile(s[:, 0], 0.9) == pytest.approx(q[0, 10], abs=1.5)
    assert np.corrcoef(s[:, 0], s[:, 23])[0, 1] == pytest.approx(0.8, abs=0.05)


def test_copula_correlation_from_history():
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 3000)
    r = copula_correlation(stats.norm.cdf(z), shrink=0.0)
    assert r[0, 1] == pytest.approx(0.7, abs=0.04)
    assert copula_correlation(stats.norm.cdf(z), shrink=0.5)[0, 1] == pytest.approx(0.35, abs=0.03)


def test_forecast_mean_sees_skew():
    sym = _normal_q(np.array([50.0]), np.array([10.0]))
    assert forecast_mean(sym, TAUS)[0] == pytest.approx(50.0, abs=0.1)
    skew = sym.copy()
    skew[0, 7:] += np.linspace(5, 80, 6)  # a long upper tail (price-spike risk)
    assert forecast_mean(skew, TAUS)[0] > skew[0, 6] + 5  # mean well above the median


def test_pinball_crps_and_coverage():
    y = np.zeros((2, 24))
    q = np.broadcast_to(stats.norm.ppf(TAUS), (2, 24, len(TAUS)))
    assert M.tau_weights(TAUS).sum() == pytest.approx(1.0)
    perfect = np.zeros((2, 24, len(TAUS)))
    assert M.crps(perfect, y, TAUS) == 0
    assert M.crps(q, y, TAUS) > 0
    # pinball for tau = 0.9 with q = 1 and y = 0 is (1 - 0.9) * 1
    one = np.ones((2, 24, 1))
    assert M.pinball(one, y, np.array([0.9]))[0] == pytest.approx(0.1)
    assert M.coverage(q[..., 1], q[..., -2], y) == 1.0
    assert M.width(q[..., 1], q[..., -2], y) == pytest.approx(2 * 1.6449, abs=1e-3)
    mask = np.ones((2, 24), dtype=bool)
    mask[0] = False
    y2 = y.copy()
    y2[0] = 100.0  # masked rows must not count
    assert M.mae(np.zeros((2, 24)), y2, mask) == 0
    assert M.rmse(np.ones((2, 24)), y, None) == 1


def test_daily_losses_and_tests():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, (300, 24))
    good, bad = y + rng.normal(0, 0.5, y.shape), y + rng.normal(0, 1.5, y.shape)
    la, lb = M.daily_abs_error(good, y), M.daily_abs_error(bad, y)
    assert la.shape == (300,)
    stat, p = M.diebold_mariano(la, lb)
    assert stat < 0 and p < 0.001
    assert M.diebold_mariano(lb, la)[1] > 0.999
    lo, hi = M.paired_bootstrap(lb - la, n_boot=2000)
    assert 0 < lo < (lb - la).mean() < hi
    q = np.repeat(good[..., None], len(TAUS), axis=-1)
    assert M.daily_pinball(q, y, TAUS).mean() == pytest.approx(M.crps(q, y, TAUS) / 2)
