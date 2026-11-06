from __future__ import annotations

import pandas as pd

from common.config import Config


def reference_weights(assets: list[str], cfg: Config) -> pd.Series:
    """Reference portfolio w_ref over ``assets``.

    The equity share (``cfg.reference["equity"]``, 70%) is split equally across the sector
    ETFs present in ``assets`` (XLRE and XLC count once they are included); the remaining
    assets get their fixed share (TLT 20%, GLD 10%). Weights sum to 1, in the order of ``assets``.
    """
    u = cfg.universe
    sector_like = set(u.sectors) | set(u.optional)
    sectors = [a for a in assets if a in sector_like]
    unknown = [a for a in assets if a not in sector_like and a not in cfg.reference]
    if unknown:
        raise ValueError(f"no reference weight defined for {unknown}")
    missing = [a for a in cfg.reference if a != "equity" and a not in assets]
    if missing:
        raise ValueError(f"reference assets missing from the universe: {missing}")
    if not sectors:
        raise ValueError("no sector ETFs in the universe")

    equity_each = cfg.reference["equity"] / len(sectors)
    w = pd.Series({a: equity_each if a in sector_like else cfg.reference[a] for a in assets},
                  name="w_ref")
    return w


def implied_returns(cov: pd.DataFrame, w_ref: pd.Series, delta: float) -> pd.Series:
    """Equilibrium excess returns by reverse optimization: pi = delta * Sigma * w_ref.

    If w_ref is optimal for a mean-variance investor with risk aversion delta, the first-order
    condition mu - delta * Sigma * w = 0 gives mu = pi. ``cov`` sets the horizon: with the
    4-week covariance, pi is the expected 4-week excess return. Indexed like ``cov``.
    """
    if set(w_ref.index) != set(cov.index):
        raise ValueError(f"w_ref assets {sorted(w_ref.index)} do not match covariance assets "
                         f"{sorted(cov.index)}")
    w = w_ref.reindex(cov.index).to_numpy()
    return pd.Series(delta * cov.to_numpy() @ w, index=cov.index, name="pi")


def equilibrium_prior(cov: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """w_ref for the assets in ``cov`` and the implied returns pi, with delta from the config."""
    w_ref = reference_weights(list(cov.index), cfg)
    pi = implied_returns(cov, w_ref, cfg.prior.delta)
    return pi, w_ref.reindex(cov.index)
