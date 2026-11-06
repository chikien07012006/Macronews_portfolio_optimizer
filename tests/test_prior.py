import numpy as np
import pandas as pd
import pytest

from common.config import load_config
from optimization.equilibrium_prior import equilibrium_prior, implied_returns, reference_weights

CFG = load_config()
CORE = list(CFG.universe.core)


def random_cov(assets, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(assets), len(assets)))
    cov = 0.004 * (a @ a.T / len(assets) + 0.5 * np.eye(len(assets)))
    return pd.DataFrame(cov, index=assets, columns=assets)


class TestReferenceWeights:
    def test_core_universe(self):
        w = reference_weights(CORE, CFG)
        assert list(w.index) == CORE
        assert w.sum() == pytest.approx(1.0)
        assert w[list(CFG.universe.sectors)].sum() == pytest.approx(0.70)
        assert w[list(CFG.universe.sectors)].nunique() == 1  # equal split
        assert w["XLK"] == pytest.approx(0.70 / 9)
        assert w["TLT"] == pytest.approx(0.20) and w["GLD"] == pytest.approx(0.10)

    def test_optional_sectors_share_the_equity_part(self):
        w = reference_weights(CORE + ["XLRE", "XLC"], CFG)
        assert w["XLRE"] == pytest.approx(0.70 / 11) and w["XLK"] == pytest.approx(0.70 / 11)
        assert w["TLT"] == pytest.approx(0.20)
        assert w.sum() == pytest.approx(1.0)

    def test_order_follows_assets(self):
        assets = ["GLD", "XLK", "TLT", "XLE"]
        assert list(reference_weights(assets, CFG).index) == assets

    def test_benchmark_rejected(self):
        with pytest.raises(ValueError):
            reference_weights(CORE + ["SPY"], CFG)

    def test_missing_bond_rejected(self):
        with pytest.raises(ValueError):
            reference_weights([a for a in CORE if a != "TLT"], CFG)


class TestImpliedReturns:
    def test_formula(self):
        cov = random_cov(CORE)
        w = reference_weights(CORE, CFG)
        pi = implied_returns(cov, w, 2.5)
        np.testing.assert_allclose(pi.to_numpy(), 2.5 * cov.to_numpy() @ w.to_numpy())
        assert list(pi.index) == CORE

    def test_reverse_optimization_recovers_w_ref(self):
        """Plugging pi into the budget-constrained mean-variance solution gives back w_ref."""
        cov = random_cov(CORE)
        pi, w_ref = equilibrium_prior(cov, CFG)
        gamma = CFG.prior.delta
        inv = np.linalg.inv(cov.to_numpy())
        ones = np.ones(len(CORE))
        nu = (ones @ inv @ pi.to_numpy() - gamma) / (ones @ inv @ ones)
        w_star = inv @ (pi.to_numpy() - nu * ones) / gamma
        np.testing.assert_allclose(w_star, w_ref.to_numpy(), atol=1e-12)
        assert nu == pytest.approx(0.0, abs=1e-12)  # budget constraint is not binding

    def test_aligns_by_label_not_position(self):
        cov = random_cov(CORE)
        w = reference_weights(CORE, CFG)
        shuffled = w.iloc[::-1]
        pd.testing.assert_series_equal(implied_returns(cov, shuffled, 2.5), implied_returns(cov, w, 2.5))

    def test_scales_with_delta_and_horizon(self):
        cov = random_cov(CORE)
        w = reference_weights(CORE, CFG)
        base = implied_returns(cov, w, 2.5)
        pd.testing.assert_series_equal(implied_returns(cov, w, 5.0), 2 * base)
        pd.testing.assert_series_equal(implied_returns(20 * cov, w, 2.5), 20 * base)

    def test_mismatched_assets_rejected(self):
        cov = random_cov(CORE)
        with pytest.raises(ValueError):
            implied_returns(cov, reference_weights(CORE + ["XLRE"], CFG), 2.5)
