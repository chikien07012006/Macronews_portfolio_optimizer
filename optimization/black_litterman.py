from __future__ import annotations

import numpy as np
import pandas as pd


def posterior(
    cov: pd.DataFrame,
    pi: pd.Series,
    P: pd.DataFrame | None = None,
    q: pd.Series | None = None,
    omega: pd.DataFrame | None = None,
    tau: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """Black-Litterman posterior mean and covariance (Woodbury form).

    Prior mu ~ N(pi, tau * Sigma); views P mu = q + eps, eps ~ N(0, Omega). With
    G = tau Sigma P' (P tau Sigma P' + Omega)^-1:

        mu_BL    = pi + G (q - P pi)              prior corrected by the view surprise
        M        = tau Sigma - G P tau Sigma      remaining uncertainty about mu
        Sigma_BL = Sigma + M

    Only a K x K matrix is inverted (K = number of views). Without views mu_BL = pi and
    Sigma_BL = (1 + tau) Sigma.

    ``P`` columns are matched to the assets of ``cov`` by label (missing columns count as 0);
    views on assets that are not in ``cov`` (e.g. not enough history yet) are dropped.
    ``q`` and ``omega`` are matched to the rows of ``P``. All inputs share one horizon
    (4 weeks in this project).
    """
    assets = cov.index
    if set(pi.index) != set(assets):
        raise ValueError("pi and cov must cover the same assets")
    sigma = cov.to_numpy()
    prior = pi.reindex(assets).to_numpy()

    if P is not None and len(P):
        outside = [c for c in P.columns if c not in assets]
        keep = ~(P[outside] != 0).any(axis=1) if outside else pd.Series(True, index=P.index)
        P = P.loc[keep]
    if P is None or len(P) == 0:
        return (pd.Series(prior, index=assets, name="mu_bl"),
                pd.DataFrame((1 + tau) * sigma, index=assets, columns=assets))

    views = P.index
    p = P.reindex(columns=assets, fill_value=0.0).to_numpy()
    q_vec = q.reindex(views).to_numpy()
    om = omega.reindex(index=views, columns=views).to_numpy()
    if np.isnan(q_vec).any() or np.isnan(om).any():
        raise ValueError("q and omega must be given for every row of P")

    tau_sigma = tau * sigma
    gain_left = tau_sigma @ p.T                   # tau Sigma P'        (N x K)
    a = p @ tau_sigma @ p.T + om                  # P tau Sigma P' + Omega  (K x K)
    surprise = q_vec - p @ prior                  # q - P pi

    mu = prior + gain_left @ np.linalg.solve(a, surprise)
    m = tau_sigma - gain_left @ np.linalg.solve(a, gain_left.T)
    m = (m + m.T) / 2                             # remove rounding asymmetry

    return (pd.Series(mu, index=assets, name="mu_bl"),
            pd.DataFrame(sigma + m, index=assets, columns=assets))
