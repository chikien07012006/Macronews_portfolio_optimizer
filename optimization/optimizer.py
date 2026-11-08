from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Tighter than the defaults (1e-8): weights come out accurate to ~1e-8, so "no trade"
# (w = w~) is exact enough for the KKT check and for not booking phantom turnover.
CLARABEL_TOLERANCES = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}


def _aligned(mu: pd.Series, cov: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    assets = cov.index
    if set(mu.index) != set(assets):
        raise ValueError("mu and cov must cover the same assets")
    sigma = cov.loc[assets, assets].to_numpy()
    return assets, mu.reindex(assets).to_numpy(), (sigma + sigma.T) / 2


def _current_weights(w_prev: pd.Series | None, assets: pd.Index) -> np.ndarray:
    """w~ on the assets of this week: assets not held (e.g. newly eligible) get 0."""
    if w_prev is None:
        return np.zeros(len(assets))
    return w_prev.reindex(assets).fillna(0.0).to_numpy()


def solve_qp(
    mu: pd.Series,
    cov: pd.DataFrame,
    w_prev: pd.Series | None = None,
    gamma: float = 2.5,
    w_max: float = 0.30,
    turnover_penalty: float = 0.001,
    solver: str = "CLARABEL",
) -> pd.Series:
    """Long-only mean-variance weights with an L1 turnover penalty.

        max  w'mu - gamma/2 w'Sigma w - lambda ||w - w~||_1
        s.t. 1'w = 1,  0 <= w <= w_max

    The L1 term is linearised with auxiliary variables u >= w - w~, u >= w~ - w and the
    penalty becomes lambda 1'u, which gives a convex QP solved by CVXPY with Clarabel.

    ``w_prev`` is the drifted weight vector just before rebalancing, matched by label;
    The backtest starts from w~ = w_ref (proposal, Algorithm 1). None means no holdings
    (w~ = 0: the penalty is then a constant and the problem is plain mean-variance).
    If the solver fails, the current weights w~ are kept (no trade this week).
    """
    assets, m, sigma = _aligned(mu, cov)
    n = len(assets)
    if n * w_max < 1 - 1e-12:
        raise ValueError(f"infeasible: {n} assets x w_max={w_max} < 1")
    w0 = _current_weights(w_prev, assets)

    w = cp.Variable(n)
    u = cp.Variable(n)
    objective = cp.Maximize(
        m @ w - gamma / 2 * cp.quad_form(w, cp.psd_wrap(sigma)) - turnover_penalty * cp.sum(u)
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,
        u >= w - w0,
        u >= w0 - w,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        options = CLARABEL_TOLERANCES if solver.upper() == "CLARABEL" else {}
        problem.solve(solver=solver, **options)
        status = problem.status
    except cp.SolverError as exc:
        status = f"error: {exc}"

    if status != cp.OPTIMAL or w.value is None:
        if w_prev is None:
            raise RuntimeError(f"solver {solver} failed in the first week ({status})")
        log.warning("solver %s failed (%s): keeping current weights", solver, status)
        return pd.Series(w0, index=assets, name="weight")

    weights = np.clip(w.value, 0.0, w_max)   # remove solver round-off such as -1e-10
    return pd.Series(weights / weights.sum(), index=assets, name="weight")


def closed_form_weights(mu: pd.Series, cov: pd.DataFrame, gamma: float = 2.5) -> pd.Series:
    """Mean-variance weights with only the budget constraint 1'w = 1 (Lagrange).

        w* = (1/gamma) Sigma^-1 (mu - nu* 1),  nu* = (1'Sigma^-1 mu - gamma) / (1'Sigma^-1 1)

    Weights can be negative or above w_max; used to test the solver when no bound binds.
    """
    assets, m, sigma = _aligned(mu, cov)
    ones = np.ones(len(assets))
    inv_mu = np.linalg.solve(sigma, m)
    inv_one = np.linalg.solve(sigma, ones)
    nu = (ones @ inv_mu - gamma) / (ones @ inv_one)
    return pd.Series((inv_mu - nu * inv_one) / gamma, index=assets, name="weight")


def check_kkt(
    w: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    w_prev: pd.Series | None = None,
    gamma: float = 2.5,
    w_max: float = 0.30,
    turnover_penalty: float = 0.001,
    tol: float = 1e-6,
) -> bool:
    """True if ``w`` satisfies the KKT conditions of :func:`solve_qp` (hence is optimal).

    With g = mu - gamma Sigma w (risk-adjusted marginal return) there must be one budget
    multiplier nu such that, for every asset,

        traded up   (w > w~): g - nu = +lambda   (or >= at w_max)
        traded down (w < w~): g - nu = -lambda   (or <= at 0)
        not traded  (w = w~): g - nu in [-lambda, +lambda]   (no-trade region)
        at 0:      the upper end of the interval is extended down to -inf
        at w_max:  the lower end is extended up to +inf

    Each asset gives an interval for nu; the conditions hold iff the intervals intersect.
    """
    assets, m, sigma = _aligned(mu, cov)
    x = w.reindex(assets).to_numpy()
    if np.isnan(x).any():
        return False
    if abs(x.sum() - 1) > tol or (x < -tol).any() or (x > w_max + tol).any():
        return False

    w0 = _current_weights(w_prev, assets)
    lam = turnover_penalty
    g = m - gamma * sigma @ x

    up, down = x > w0 + tol, x < w0 - tol
    lo = np.where(up, lam, -lam)                       # bounds on g - nu from the L1 term
    hi = np.where(down, -lam, lam)
    hi = np.where(x >= w_max - tol, np.inf, hi)        # at the cap: may want more
    lo = np.where(x <= tol, -np.inf, lo)               # at zero: may want less

    nu_low = np.max(g - hi)    # nu >= g_i - hi_i
    nu_high = np.min(g - lo)   # nu <= g_i - lo_i
    return bool(nu_low <= nu_high + tol)
