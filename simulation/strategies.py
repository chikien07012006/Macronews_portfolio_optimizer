"""Strategies for the weekly backtest: the main method (Black-Litterman with technical
views) and the baselines, in the ``Strategy`` form of :func:`simulation.backtest.run_backtest`.

A strategy is called at each decision date t with the current (drifted) weights and returns
target weights. Everything that only looks backward is computed once in :class:`Inputs`
and sliced at t inside the strategies:

    covariance   Ledoit-Wolf on the 252 daily returns up to t (cached per date)
    features     MOM / REV / VOL at t (computed from returns up to t)
    views        Ridge trained on weeks s with s + 4 <= t (purged inside generate_views)
    Markowitz    mean of the 156 weekly excess returns up to t

The backtest starts from cash; the first rebalance treats the reference portfolio w_ref as
the current holdings (proposal, Algorithm 1), so the turnover penalty is measured from w_ref.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable

import pandas as pd

from common.config import Config
from data_pipeline.market_data import daily_returns, weekly_excess_returns, weekly_returns
from optimization import baselines
from optimization.black_litterman import posterior
from optimization.equilibrium_prior import equilibrium_prior, reference_weights
from optimization.ledoit_wolf import ledoit_wolf_cov
from optimization.optimizer import solve_qp
from simulation.backtest import Strategy
from simulation.features import technical_features
from simulation.views import forward_target, generate_views


class Inputs:
    """Market data, features and labels for one universe, plus a covariance cache."""

    def __init__(self, conn: sqlite3.Connection, cfg: Config, assets: list[str] | None = None):
        self.cfg = cfg
        self.assets = list(cfg.universe.core) if assets is None else list(assets)
        self.daily = daily_returns(conn)[self.assets]
        self.weekly = weekly_returns(conn)[self.assets]
        self.weekly_excess = weekly_excess_returns(conn)[self.assets]
        self.features = technical_features(self.weekly, self.daily, cfg)   # plain returns
        self.target = forward_target(self.weekly_excess, cfg.views.horizon_weeks)  # excess
        self._cov: dict[pd.Timestamp, pd.DataFrame] = {}

    def cov(self, t: pd.Timestamp) -> pd.DataFrame:
        """Ledoit-Wolf covariance at t in the 4-week unit, shared by all strategies."""
        t = pd.Timestamp(t)
        if t not in self._cov:
            risk = self.cfg.risk
            self._cov[t] = ledoit_wolf_cov(self.daily, t, risk.cov_window_days, risk.horizon_days).cov
        return self._cov[t]


def previous_weights(w_current: pd.Series, assets: list[str], cfg: Config) -> pd.Series:
    """w~ on ``assets``; before the first trade (nothing held) it is w_ref."""
    w = w_current.reindex(assets).fillna(0.0)
    if w.sum() < 1e-9:
        return reference_weights(assets, cfg)
    return w


def bl_technical(inp: Inputs) -> Strategy:
    """Main method: equilibrium prior + technical Ridge views -> Black-Litterman -> QP."""
    cfg, opt = inp.cfg, inp.cfg.optimizer
    alpha = cfg.views.ridge_alpha if cfg.views.ridge_alpha is not None else 1.0

    def strategy(t: pd.Timestamp, w_current: pd.Series) -> pd.Series:
        cov = inp.cov(t)
        assets = list(cov.index)
        pi, _ = equilibrium_prior(cov, cfg)
        P, q, omega = generate_views(inp.features, inp.target, t, assets, cfg, alpha=alpha)
        mu, cov_bl = posterior(cov, pi, P, q, omega, cfg.prior.tau)
        w_prev = previous_weights(w_current, assets, cfg)
        return solve_qp(mu, cov_bl, w_prev, opt.gamma, opt.w_max, opt.turnover_penalty, opt.solver)

    return strategy


def bl_no_views(inp: Inputs) -> Strategy:
    def strategy(t, w_current):
        cov = inp.cov(t)
        return baselines.bl_no_views(cov, inp.cfg, previous_weights(w_current, list(cov.index), inp.cfg))

    return strategy


def markowitz(inp: Inputs) -> Strategy:
    opt = inp.cfg.optimizer

    def strategy(t, w_current):
        cov = inp.cov(t)
        w_prev = previous_weights(w_current, list(cov.index), inp.cfg)
        return baselines.markowitz(inp.weekly_excess, cov, t, w_prev, opt.gamma, opt.w_max,
                                   opt.turnover_penalty, inp.cfg.views.min_train_weeks,
                                   inp.cfg.views.horizon_weeks)

    return strategy


def min_variance(inp: Inputs) -> Strategy:
    opt = inp.cfg.optimizer

    def strategy(t, w_current):
        cov = inp.cov(t)
        w_prev = previous_weights(w_current, list(cov.index), inp.cfg)
        return baselines.min_variance(cov, w_prev, opt.gamma, opt.w_max, opt.turnover_penalty)

    return strategy


def inverse_volatility(inp: Inputs) -> Strategy:
    return lambda t, w_current: baselines.inverse_volatility(inp.cov(t), inp.cfg.optimizer.w_max)


def equal_weight(inp: Inputs) -> Strategy:
    return lambda t, w_current: baselines.equal_weight(inp.cov(t).index)


def buy_and_hold(inp: Inputs, ticker: str | None = None) -> Strategy:
    ticker = ticker or inp.cfg.universe.benchmark
    return lambda t, w_current: pd.Series({ticker: 1.0})


# name -> factory, in the order used in figures and tables
STRATEGIES: dict[str, Callable[[Inputs], Strategy]] = {
    "BL + technical views": bl_technical,
    "BL (no views)": bl_no_views,
    "Markowitz": markowitz,
    "Min variance": min_variance,
    "Inverse volatility": inverse_volatility,
    "Equal weight": equal_weight,
    "SPY buy-and-hold": buy_and_hold,
}
