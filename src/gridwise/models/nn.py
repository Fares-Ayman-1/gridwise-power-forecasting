"""Quantile neural network: one multi-output network forecasts all 24 hours and all quantile levels.

This is the "DNN" of the price-forecasting literature (Lago et al., 2021) with a quantile head. The
whole day is one training example, so the network can learn how the shape of yesterday's curve and
tomorrow's weather move the whole price curve together, which the per-hour boosting model cannot.

The head outputs the lowest quantile plus positive increments, so quantiles cannot cross. A few
networks with different seeds are averaged. Needs the optional `nn` extra (PyTorch, CPU is enough).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gridwise.features import Features
from gridwise.frames import SLOTS

DAY_COLUMNS = ["is_weekend", "is_holiday", "holiday_yesterday", "doy_sin", "doy_cos"]
HOUR_COLUMNS = {
    "price": [],
    "weather": ["wx_wind_on_pow", "wx_wind_off_pow", "wx_solar", "wx_temp"],
    "operator_leaky": ["op_load", "op_wind", "op_solar"],
}


class _Net(nn.Module):
    def __init__(self, n_in: int, n_q: int, hidden: tuple[int, int], dropout: float):
        super().__init__()
        self.n_q = n_q
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], SLOTS * n_q),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x).view(-1, SLOTS, self.n_q)
        steps = nn.functional.softplus(out[..., 1:])
        return torch.cat([out[..., :1], out[..., :1] + torch.cumsum(steps, dim=-1)], dim=-1)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(taus * diff, (taus - 1) * diff).mean()


class QuantileNN:
    kind = "quantile"

    def __init__(
        self,
        taus: list[float],
        feature_set: str = "weather",
        hidden: tuple[int, int] = (256, 128),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 150,
        patience: int = 15,
        batch_size: int = 64,
        seeds: int = 3,
        train_days: int | None = 1095,
        refit_every: int = 28,
        threads: int = 4,
        name: str = "qnn",
    ):
        self.taus = list(taus)
        self.hour_columns = HOUR_COLUMNS[feature_set]
        self.hidden, self.dropout, self.lr, self.weight_decay = (
            (hidden[0], hidden[1]),
            dropout,
            lr,
            weight_decay,
        )
        self.max_epochs, self.patience, self.batch_size, self.seeds = max_epochs, patience, batch_size, seeds
        self.train_days, self.refit_every, self.threads, self.name = train_days, refit_every, threads, name
        self._nets: list[_Net] = []
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _design(self, feats: Features, idx: np.ndarray) -> np.ndarray:
        """One row per day: 96 lagged prices, tomorrow's hourly exogenous values, calendar, market level."""
        k = len(idx)
        parts = [np.arcsinh(feats.lags[idx].reshape(k, -1)), np.eye(7)[feats.dow[idx]]]
        cols = [feats.names.index(c) for c in self.hour_columns]
        if cols:
            parts.append(feats.X[idx][:, :, cols].astype(float).reshape(k, -1))
        day_cols = [feats.names.index(c) for c in DAY_COLUMNS]
        parts.append(feats.X[idx][:, 0, day_cols].astype(float))
        parts.append(np.log(feats.scale[idx])[:, None])
        parts.append(np.arcsinh(feats.level[idx] / 50.0)[:, None])
        return np.concatenate(parts, axis=1)

    def fit(self, feats: Features, train_idx: np.ndarray) -> None:
        torch.set_num_threads(self.threads)
        X = self._design(feats, train_idx)
        y = np.arcsinh(feats.y[train_idx])
        ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
        X, y = X[ok], y[ok]
        self._mu, self._sd = X.mean(axis=0), X.std(axis=0)
        self._sd[self._sd < 1e-9] = 1.0
        Xt = torch.tensor((X - self._mu) / self._sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        taus = torch.tensor(self.taus, dtype=torch.float32)
        n_val = max(20, int(0.15 * len(Xt)))  # the most recent days: validation follows training in time
        tr, va = slice(0, len(Xt) - n_val), slice(len(Xt) - n_val, len(Xt))
        self._nets = []
        for seed in range(self.seeds):
            torch.manual_seed(seed)
            net = _Net(Xt.shape[1], len(self.taus), self.hidden, self.dropout)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, best_state, bad = float("inf"), None, 0
            gen = torch.Generator().manual_seed(seed)
            for _ in range(self.max_epochs):
                net.train()
                perm = torch.randperm(tr.stop, generator=gen)
                for b in range(0, tr.stop, self.batch_size):
                    rows = perm[b : b + self.batch_size]
                    opt.zero_grad()
                    pinball_loss(net(Xt[rows]), yt[rows], taus).backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    val = float(pinball_loss(net(Xt[va]), yt[va], taus))
                if val < best - 1e-5:
                    best, bad = val, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            assert best_state is not None
            net.load_state_dict(best_state)
            net.eval()
            self._nets.append(net)

    def predict(self, feats: Features, idx: np.ndarray) -> np.ndarray:
        assert self._mu is not None and self._sd is not None, "fit first"
        X = np.nan_to_num((self._design(feats, idx) - self._mu) / self._sd)
        with torch.no_grad():
            out = torch.stack([net(torch.tensor(X, dtype=torch.float32)) for net in self._nets]).mean(dim=0)
        return np.sinh(out.numpy().astype(float))
