"""Day-ahead schedule for a battery that buys and sells at the hourly auction price.

The schedule for a delivery day is fixed before prices are known and starts and ends at the same
state of charge, so days are independent. A binary variable per hour forbids charging and
discharging at once; without it the solver would "burn" energy through the efficiency loss whenever
prices are negative, which a real battery cannot do.

Profit is linear in the prices: profit = sum_h price_h * (discharge_h - charge_h) - wear cost. So the
schedule that maximises expected profit is the one planned on the forecast mean, and sampling
scenarios cannot improve on it. A distribution only changes the decision when risk matters. The
second planner therefore maximises a mix of expected profit and CVaR (the mean of the worst
alpha-share of scenario outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


class BatterySpec(BaseModel):
    power_mw: float = Field(gt=0)
    capacity_mwh: float = Field(gt=0)
    eff_charge: float = Field(gt=0, le=1)
    eff_discharge: float = Field(gt=0, le=1)
    max_cycles_per_day: float = Field(gt=0)
    degradation_eur_per_mwh: float = Field(ge=0)  # wear cost per MWh delivered to the grid
    initial_soc: float = Field(ge=0, le=1, default=0.0)  # share of capacity at the start and end of a day


@dataclass
class Schedule:
    charge: np.ndarray  # MW drawn from the grid in each hour
    discharge: np.ndarray  # MW delivered to the grid in each hour
    soc: np.ndarray  # MWh stored at the end of each hour
    status: str

    @property
    def net(self) -> np.ndarray:
        return self.discharge - self.charge


def profit(s: Schedule, prices: np.ndarray, spec: BatterySpec) -> float:
    """Money made on `prices` (EUR/MWh), net of the wear cost."""
    return float(np.sum(prices * s.net) - spec.degradation_eur_per_mwh * np.sum(s.discharge))


def _base_problem(n: int, spec: BatterySpec) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Constraints on x = [charge (n), discharge (n), charging flag (n)]; returns A, lower, upper."""
    eye, zero = sparse.identity(n, format="csr"), sparse.csr_matrix((n, n))
    P, E = spec.power_mw, spec.capacity_mwh
    e0 = spec.initial_soc * E
    tril = sparse.csr_matrix(np.tril(np.ones((n, n))))
    stored = sparse.hstack([spec.eff_charge * tril, -tril / spec.eff_discharge, zero])  # cumulative MWh
    rows = [
        sparse.hstack([eye, zero, -P * eye]),  # charge <= P * flag
        sparse.hstack([zero, eye, P * eye]),  # discharge <= P * (1 - flag)
        stored,  # 0 <= e0 + stored <= E
        sparse.csr_matrix(np.concatenate([np.zeros(n), np.full(n, 1 / spec.eff_discharge), np.zeros(n)])),
    ]
    lo = np.concatenate([np.full(n, -np.inf), np.full(n, -np.inf), np.full(n, -e0), [-np.inf]])
    hi = np.concatenate([np.zeros(n), np.full(n, P), np.full(n, E - e0), [spec.max_cycles_per_day * E]])
    lo[3 * n - 1] = hi[3 * n - 1] = 0.0  # end the day at the starting state of charge
    return sparse.vstack(rows, format="csr"), lo, hi


def _unpack(x: np.ndarray, n: int, spec: BatterySpec, status: str) -> Schedule:
    c, d = np.clip(x[:n], 0, None), np.clip(x[n : 2 * n], 0, None)
    c[c < 1e-9] = 0.0
    d[d < 1e-9] = 0.0
    soc = spec.initial_soc * spec.capacity_mwh + np.cumsum(spec.eff_charge * c - d / spec.eff_discharge)
    return Schedule(charge=c, discharge=d, soc=soc, status=status)


def idle(n: int, spec: BatterySpec) -> Schedule:
    z = np.zeros(n)
    return Schedule(z.copy(), z.copy(), np.full(n, spec.initial_soc * spec.capacity_mwh), "idle")


def plan(prices: np.ndarray, spec: BatterySpec) -> Schedule:
    """Schedule that maximises profit on the given price vector (a forecast, or the truth for the bound)."""
    n = len(prices)
    A, lo, hi = _base_problem(n, spec)
    cost = np.concatenate([prices, -prices + spec.degradation_eur_per_mwh, np.zeros(n)])
    res = milp(
        cost,
        constraints=LinearConstraint(A, lo, hi),
        integrality=np.concatenate([np.zeros(2 * n), np.ones(n)]),
        bounds=Bounds(np.zeros(3 * n), np.concatenate([np.full(2 * n, spec.power_mw), np.ones(n)])),
    )
    if res.x is None:
        return idle(n, spec)
    return _unpack(res.x, n, spec, "optimal" if res.success else "feasible")


def plan_cvar(scenarios: np.ndarray, spec: BatterySpec, alpha: float = 0.1, weight: float = 0.5) -> Schedule:
    """Schedule that maximises (1 - weight) * mean profit + weight * CVaR_alpha over price scenarios.

    scenarios: (S, n) EUR/MWh. CVaR_alpha is the mean profit of the worst alpha-share of scenarios,
    written with the Rockafellar-Uryasev linear form: max_z z - 1/(alpha S) sum_s max(0, z - profit_s).
    """
    S, n = scenarios.shape
    A0, lo0, hi0 = _base_problem(n, spec)
    wear = spec.degradation_eur_per_mwh
    mean = scenarios.mean(axis=0)
    # x = [charge, discharge, flag, z, shortfall (S)]
    nx = 3 * n + 1 + S
    cost = np.concatenate(
        [
            (1 - weight) * mean,
            (1 - weight) * (-mean + wear),
            np.zeros(n),
            [-weight],
            np.full(S, weight / (alpha * S)),
        ]
    )
    # z - profit_s - shortfall_s <= 0, with profit_s = sum p_s (d - c) - wear sum d
    shortfall = sparse.hstack(
        [
            sparse.csr_matrix(scenarios),
            sparse.csr_matrix(-scenarios + wear),
            sparse.csr_matrix((S, n)),
            sparse.csr_matrix(np.ones((S, 1))),
            -sparse.identity(S, format="csr"),
        ]
    )
    A = sparse.vstack([sparse.hstack([A0, sparse.csr_matrix((A0.shape[0], 1 + S))]), shortfall], format="csr")
    lo = np.concatenate([lo0, np.full(S, -np.inf)])
    hi = np.concatenate([hi0, np.zeros(S)])
    lb = np.concatenate([np.zeros(3 * n), [-np.inf], np.zeros(S)])
    ub = np.concatenate([np.full(2 * n, spec.power_mw), np.ones(n), [np.inf], np.full(S, np.inf)])
    integrality = np.zeros(nx)
    integrality[2 * n : 3 * n] = 1
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=Bounds(lb, ub))
    if res.x is None:
        return idle(n, spec)
    return _unpack(res.x, n, spec, "optimal" if res.success else "feasible")
