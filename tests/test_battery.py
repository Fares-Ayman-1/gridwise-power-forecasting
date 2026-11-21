import numpy as np
import pytest

from gridwise.battery import BatterySpec, idle, plan, plan_cvar, profit

SPEC = BatterySpec(power_mw=1, capacity_mwh=2, eff_charge=0.9, eff_discharge=0.9, max_cycles_per_day=2,
                   degradation_eur_per_mwh=0, initial_soc=0)  # fmt: skip


def _check_physics(s, spec):
    assert (s.charge >= -1e-9).all() and (s.charge <= spec.power_mw + 1e-9).all()
    assert (s.discharge >= -1e-9).all() and (s.discharge <= spec.power_mw + 1e-9).all()
    assert (s.charge * s.discharge < 1e-9).all(), "charging and discharging in the same hour"
    assert (s.soc >= -1e-6).all() and (s.soc <= spec.capacity_mwh + 1e-6).all()
    assert s.soc[-1] == pytest.approx(spec.initial_soc * spec.capacity_mwh, abs=1e-6)
    delivered_from_store = np.sum(s.discharge) / spec.eff_discharge
    assert delivered_from_store <= spec.max_cycles_per_day * spec.capacity_mwh + 1e-6


def test_buys_low_and_sells_high():
    prices = np.array([10, 10, 10, 10, 100, 100, 100, 100], dtype=float)
    s = plan(prices, SPEC)
    _check_physics(s, SPEC)
    # 2 MWh of storage: buy 2 / 0.9 MWh at 10, sell 2 * 0.9 MWh at 100
    assert profit(s, prices, SPEC) == pytest.approx(2 * 0.9 * 100 - 2 / 0.9 * 10, rel=1e-6)
    assert s.charge[4:].sum() == 0 and s.discharge[:4].sum() == 0


def test_flat_prices_mean_no_trading():
    s = plan(np.full(24, 50.0), SPEC)
    assert profit(s, np.full(24, 50.0), SPEC) == pytest.approx(0, abs=1e-6)


def test_spread_smaller_than_losses_is_not_traded():
    prices = np.array([50.0, 55.0] * 12)  # buying at 50 and selling at 55 loses money at 81% round trip
    assert plan(prices, SPEC).net == pytest.approx(np.zeros(24), abs=1e-6)


def test_negative_prices_do_not_let_the_battery_burn_energy():
    prices = np.array([-50.0] * 6 + [-40.0] * 6)
    s = plan(prices, SPEC)
    _check_physics(s, SPEC)
    # Paid to charge, pays to discharge. Cycling still earns a little through the efficiency loss, but
    # only as much as two full cycles allow: at best all charging at -50 and all discharging at -40.
    cycles = SPEC.max_cycles_per_day * SPEC.capacity_mwh
    assert 0 < profit(s, prices, SPEC) <= cycles / 0.9 * 50 - cycles * 0.9 * 40 + 1e-6


def test_wear_cost_and_cycle_limit():
    prices = np.array([10, 100] * 12, dtype=float)
    spec = SPEC.model_copy(update={"max_cycles_per_day": 1.0})
    s = plan(prices, spec)
    _check_physics(s, spec)
    assert np.sum(s.discharge) / spec.eff_discharge == pytest.approx(2.0, abs=1e-6)
    costly = SPEC.model_copy(update={"degradation_eur_per_mwh": 200.0})
    assert plan(prices, costly).net == pytest.approx(np.zeros(24), abs=1e-6)


def test_perfect_foresight_is_an_upper_bound():
    rng = np.random.default_rng(0)
    for _ in range(10):
        real = rng.normal(80, 40, 24)
        guess = real + rng.normal(0, 30, 24)
        assert profit(plan(real, SPEC), real, SPEC) >= profit(plan(guess, SPEC), real, SPEC) - 1e-6


def test_expected_profit_is_maximised_by_planning_on_the_mean():
    """Profit is linear in price, so with no risk aversion the scenario planner equals the mean planner."""
    rng = np.random.default_rng(1)
    scen = 80 + 30 * np.sin(np.arange(24) / 24 * 2 * np.pi) + rng.normal(0, 25, (60, 24))
    on_mean = plan(scen.mean(axis=0), SPEC)
    neutral = plan_cvar(scen, SPEC, alpha=0.1, weight=0.0)
    expected = lambda s: np.mean([profit(s, p, SPEC) for p in scen])  # noqa: E731
    assert expected(neutral) == pytest.approx(expected(on_mean), rel=1e-6)


def test_risk_aversion_trades_mean_for_a_better_worst_case():
    rng = np.random.default_rng(2)
    base = 80 + 5 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    scen = base + rng.normal(0, 40, (80, 24))  # small expected spread, large uncertainty
    neutral, averse = plan(scen.mean(axis=0), SPEC), plan_cvar(scen, SPEC, alpha=0.1, weight=0.9)
    _check_physics(averse, SPEC)

    def outcomes(s):
        return np.sort([profit(s, p, SPEC) for p in scen])

    assert outcomes(averse)[:8].mean() >= outcomes(neutral)[:8].mean() - 1e-6
    assert outcomes(averse).mean() <= outcomes(neutral).mean() + 1e-6


def test_works_on_23_and_25_hour_days_and_idle():
    for n in (23, 25):
        s = plan(np.linspace(10, 100, n), SPEC)
        assert len(s.charge) == n
        _check_physics(s, SPEC)
    z = idle(24, SPEC)
    assert profit(z, np.full(24, 99.0), SPEC) == 0
