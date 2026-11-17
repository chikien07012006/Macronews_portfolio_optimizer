"""Ridge view generator, shared by every feature source (technical now, news later).

For each asset i, a Ridge regression maps the asset's own standardized features Z(i,s)
to its forward excess return y(i,s) over the next ``h`` weeks (h = cfg.views.horizon_weeks,
4). It is trained on an expanding window of weeks whose label is already known at the
decision date t (purging: s + h <= t):

    q(i,t)    = a + b' Z(i,t)
    view if   |q(i,t)| > threshold * sigma(i)      sigma^2: out-of-fold mean squared error
    Omega_ii  = c * sigma^2(i)                      a poor model gives an uncertain view
    P row     = e_i'                                absolute view on asset i

q, sigma^2 and Omega are all in the h-week unit used by pi and Sigma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from common.config import Config

OOF_SPLITS = 5


def forward_target(weekly_excess: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Training label y(i,s) = X(i,s+1) + ... + X(i,s+horizon).

    Deliberately uses future data: it is the label, never a feature. The last ``horizon``
    rows are NaN. Look-ahead is prevented when the training weeks are chosen
    (:func:`training_weeks`).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    return weekly_excess.rolling(horizon).sum().shift(-horizon)


def training_weeks(
    calendar: pd.DatetimeIndex, as_of: str | pd.Timestamp, horizon: int = 4
) -> pd.DatetimeIndex:
    """Weeks s of ``calendar`` whose label is known at ``as_of``: position(s) <= position(t) - h.

    The window expands from the start of ``calendar``.
    """
    calendar = pd.DatetimeIndex(calendar)
    as_of = pd.Timestamp(as_of)
    if as_of not in calendar:
        raise ValueError(f"{as_of.date()} is not a decision date of the target")
    pos = calendar.get_loc(as_of)
    return calendar[: max(pos - horizon + 1, 0)]


def _oof_mse(x: np.ndarray, y: np.ndarray, alpha: float, gap: int) -> float:
    """Mean squared error of out-of-fold forecasts (expanding folds, ``gap`` weeks purged)."""
    errors = []
    for train, test in TimeSeriesSplit(n_splits=OOF_SPLITS, gap=gap).split(x):
        model = Ridge(alpha=alpha).fit(x[train], y[train])
        errors.append(y[test] - model.predict(x[test]))
    return float(np.mean(np.concatenate(errors) ** 2))


def _asset_forecast(
    features: pd.DataFrame,
    target: pd.DataFrame,
    asset: str,
    as_of: pd.Timestamp,
    train_weeks: pd.DatetimeIndex,
    min_train: int,
    alpha: float,
    gap: int,
) -> tuple[float, float] | None:
    """(q_hat, sigma2) for one asset, or None if it has no features at t or too little history."""
    if asset not in target.columns or asset not in features.index.get_level_values("ticker"):
        return None
    z = features.xs(asset, level="ticker")
    if as_of not in z.index:
        return None

    y = target[asset].reindex(z.index)
    train = z.index[z.index.isin(train_weeks) & y.notna().to_numpy()]  # time order kept
    if len(train) < min_train:
        return None

    x_train, y_train = z.loc[train].to_numpy(), y.loc[train].to_numpy()
    sigma2 = _oof_mse(x_train, y_train, alpha, gap)
    model = Ridge(alpha=alpha).fit(x_train, y_train)
    q_hat = float(model.predict(z.loc[[as_of]].to_numpy())[0])
    return q_hat, sigma2


def generate_views(
    features: pd.DataFrame,
    target: pd.DataFrame,
    as_of: pd.Timestamp,
    assets: list[str],
    cfg: Config,
    c: float | None = None,
    alpha: float = 1.0,
    name: str = "tech",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Absolute views (P, q, Omega) at decision date ``as_of``.

    ``features``: indexed by (decision_date, ticker); every column is used as a regressor
    (MOM, REV, VOL for technical views). ``target``: :func:`forward_target`, index = decision
    dates. ``assets`` sets the columns of P. ``c`` scales Omega (default ``cfg.views.c_tech``).
    ``name`` prefixes the view labels ("tech:XLK"), so views from different sources on the
    same asset stay distinct.

    Returns P (K x N), q (K) and a diagonal Omega (K x K). With no views K = 0: P is empty
    but still has the asset columns, which Black-Litterman treats as "no views".
    """
    v = cfg.views
    c = v.c_tech if c is None else c
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if "ticker" not in features.index.names:
        raise ValueError("features must be indexed by (decision_date, ticker)")
    as_of = pd.Timestamp(as_of)
    assets = list(assets)
    train_weeks = training_weeks(target.index, as_of, v.horizon_weeks)

    kept = []  # (asset, q_hat, sigma2)
    for asset in assets:
        # TODO: CV for alpha (cfg.views.ridge_alpha); fixed at 1.0 for now
        forecast = _asset_forecast(features, target, asset, as_of, train_weeks,
                                   v.min_train_weeks, alpha, v.horizon_weeks)
        if forecast is None:
            continue
        q_hat, sigma2 = forecast
        if abs(q_hat) > v.signal_threshold * np.sqrt(sigma2):
            kept.append((asset, q_hat, sigma2))

    labels = [f"{name}:{asset}" for asset, _, _ in kept]
    P = pd.DataFrame(0.0, index=labels, columns=assets)
    for label, (asset, _, _) in zip(labels, kept):
        P.loc[label, asset] = 1.0
    q = pd.Series([q_hat for _, q_hat, _ in kept], index=labels, name="q", dtype=float)
    omega = pd.DataFrame(np.diag(np.array([c * s2 for _, _, s2 in kept], dtype=float)),
                         index=labels, columns=labels)
    return P, q, omega
