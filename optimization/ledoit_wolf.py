from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LedoitWolfResult:
    cov: pd.DataFrame         # horizon_days * Sigma_LW: the matrix used by the model
    cov_daily: pd.DataFrame   # Sigma_LW = (1 - rho) * S + rho * nu * I
    sample_cov: pd.DataFrame  # S, sample covariance of daily returns
    nu: float                 # average variance tr(S) / N
    shrinkage: float          # rho*, in [0, 1]


def ledoit_wolf_cov(
    daily_returns: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    window: int = 252,
    horizon_days: int = 20,
) -> LedoitWolfResult:
    """Ledoit-Wolf (2004) covariance at a rebalancing date.

    Uses the last ``window`` trading days of daily returns up to and including ``as_of``
    (default: the last row). Assets with a missing return inside the window are left out,
    so an asset enters only once it has a full window of history.

    1. S     = sample covariance of the window (1/T, demeaned)
    2. nu    = tr(S) / N, target F = nu * I
    3. rho*  = b^2 / d^2 with d^2 = ||S - F||^2 (distance to the target) and
               b^2 = min(d^2, 1/T^2 * sum_k ||x_k x_k' - S||^2) (estimation noise in S)
    4. Sigma_LW = (1 - rho*) S + rho* F, scaled by ``horizon_days`` (20 = 4-week horizon)
    """
    data = daily_returns if as_of is None else daily_returns.loc[: pd.Timestamp(as_of)]
    if len(data) < window:
        raise ValueError(f"need {window} daily returns up to {as_of}, got {len(data)}")
    data = data.iloc[-window:]
    data = data.loc[:, data.notna().all()]
    n_assets = data.shape[1]
    if n_assets < 2:
        raise ValueError("fewer than two assets with a full return window")

    x = data.to_numpy() - data.to_numpy().mean(axis=0)
    t = len(x)

    # 1. sample covariance
    s = x.T @ x / t

    # 2. shrinkage target
    nu = np.trace(s) / n_assets
    target = nu * np.eye(n_assets)

    # 3. optimal shrinkage intensity
    d2 = np.sum((s - target) ** 2)
    b2_bar = sum(np.sum((np.outer(xk, xk) - s) ** 2) for xk in x) / t**2
    b2 = min(b2_bar, d2)
    rho = b2 / d2 if d2 > 0 else 0.0

    # 4. shrunk covariance, scaled to the forecast horizon
    lw = (1 - rho) * s + rho * target

    labels = data.columns
    return LedoitWolfResult(
        cov=pd.DataFrame(horizon_days * lw, index=labels, columns=labels),
        cov_daily=pd.DataFrame(lw, index=labels, columns=labels),
        sample_cov=pd.DataFrame(s, index=labels, columns=labels),
        nu=float(nu),
        shrinkage=float(rho),
    )
