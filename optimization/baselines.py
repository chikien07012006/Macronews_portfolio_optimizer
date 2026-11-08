"""Baseline allocations, each computed at one rebalancing date.

All baselines use the same constraints as the main strategy (long-only, w_i <= w_max,
fully invested); the QP-based ones also use the same gamma and turnover penalty, so the
comparison isolates one ingredient at a time:

    equal_weight        no information
    inverse_volatility  own volatilities only (ignores correlations)
    min_variance        full covariance, no expected returns
    bl_no_views         equilibrium prior pi, no views   (reference for the ablation)
    markowitz           156-week sample-mean returns instead of BL

SPY buy-and-hold is a benchmark of the backtest, not an allocation rule, so it is not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import Config
from optimization.black_litterman import posterior
from optimization.equilibrium_prior import equilibrium_prior
from optimization.optimizer import solve_qp


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    """w_i = 1/N."""
    assets = pd.Index(assets)
    return pd.Series(1.0 / len(assets), index=assets, name="weight")


def inverse_volatility(cov: pd.DataFrame, w_max: float = 0.30) -> pd.Series:
    """w_i proportional to 1/sigma_i, capped at ``w_max``.

    Assets above the cap are fixed at w_max and the rest of the budget is shared among
    the others, still in proportion to 1/sigma; repeated until no weight exceeds the cap.
    """
    assets = cov.index
    n = len(assets)
    if n * w_max < 1 - 1e-12:
        raise ValueError(f"infeasible: {n} assets x w_max={w_max} < 1")
    inv_vol = 1.0 / np.sqrt(np.diag(cov.to_numpy()))

    capped = np.zeros(n, dtype=bool)
    while True:
        free = ~capped
        w = np.where(capped, w_max, 0.0)
        w[free] = inv_vol[free] / inv_vol[free].sum() * (1 - w_max * capped.sum())
        over = free & (w > w_max + 1e-12)
        if not over.any():
            return pd.Series(w, index=assets, name="weight")
        capped |= over


def min_variance(
    cov: pd.DataFrame,
    w_prev: pd.Series | None = None,
    gamma: float = 2.5,
    w_max: float = 0.30,
    turnover_penalty: float = 0.001,
) -> pd.Series:
    """min w'Sigma w under the same constraints: the QP of :func:`solve_qp` with mu = 0."""
    mu = pd.Series(0.0, index=cov.index)
    return solve_qp(mu, cov, w_prev, gamma, w_max, turnover_penalty)


def sample_mean_returns(
    weekly_excess: pd.DataFrame,
    as_of: str | pd.Timestamp,
    assets: list[str] | pd.Index,
    window_weeks: int = 156,
    horizon_weeks: int = 4,
) -> pd.Series:
    """Historical expected return: mean weekly excess return of the last ``window_weeks``
    weeks up to and including ``as_of``, times ``horizon_weeks`` (same 4-week unit as pi).

    An asset with a shorter history inside the window (e.g. XLRE in its first years) is
    averaged over the weeks it has.
    """
    data = weekly_excess.loc[: pd.Timestamp(as_of), list(assets)]
    if len(data) < window_weeks:
        raise ValueError(f"need {window_weeks} weeks up to {as_of}, got {len(data)}")
    mean = data.iloc[-window_weeks:].mean()
    if mean.isna().any():
        raise ValueError(f"no returns in the window for {list(mean[mean.isna()].index)}")
    return (horizon_weeks * mean).rename("mu_hist")


def markowitz(
    weekly_excess: pd.DataFrame,
    cov: pd.DataFrame,
    as_of: str | pd.Timestamp,
    w_prev: pd.Series | None = None,
    gamma: float = 2.5,
    w_max: float = 0.30,
    turnover_penalty: float = 0.001,
    window_weeks: int = 156,
    horizon_weeks: int = 4,
) -> pd.Series:
    """Classical mean-variance: sample-mean returns and Ledoit-Wolf covariance, same QP."""
    mu = sample_mean_returns(weekly_excess, as_of, cov.index, window_weeks, horizon_weeks)
    return solve_qp(mu, cov, w_prev, gamma, w_max, turnover_penalty)


def bl_no_views(cov: pd.DataFrame, cfg: Config, w_prev: pd.Series | None = None) -> pd.Series:
    """Black-Litterman without views: mu = pi, Sigma_BL = (1 + tau) Sigma; close to w_ref."""
    pi, _ = equilibrium_prior(cov, cfg)
    mu, cov_bl = posterior(cov, pi, tau=cfg.prior.tau)
    opt = cfg.optimizer
    return solve_qp(mu, cov_bl, w_prev, opt.gamma, opt.w_max, opt.turnover_penalty, opt.solver)
