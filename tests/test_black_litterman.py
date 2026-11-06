import numpy as np
import pandas as pd
import pytest

from optimization.black_litterman import posterior

ASSETS = ["XLB", "XLE", "XLK", "XLY", "TLT"]
TAU = 0.05


def random_cov(assets=ASSETS, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(len(assets), len(assets)))
    cov = 0.004 * (a @ a.T / len(assets) + 0.5 * np.eye(len(assets)))
    return pd.DataFrame(cov, index=assets, columns=assets)


def absolute_views(spec, assets=ASSETS):
    """spec: list of (asset, q, omega) -> P, q, Omega as the view generator returns them."""
    names = [f"v{k}:{a}" for k, (a, _, _) in enumerate(spec)]
    P = pd.DataFrame(0.0, index=names, columns=assets)
    for name, (a, _, _) in zip(names, spec):
        P.loc[name, a] = 1.0
    q = pd.Series([v for _, v, _ in spec], index=names)
    omega = pd.DataFrame(np.diag([w for _, _, w in spec]), index=names, columns=names)
    return P, q, omega


@pytest.fixture
def prior():
    cov = random_cov()
    pi = pd.Series(2.5 * cov.to_numpy() @ np.full(len(ASSETS), 1 / len(ASSETS)), index=ASSETS)
    return cov, pi


def test_no_views_returns_prior(prior):
    cov, pi = prior
    mu, cov_bl = posterior(cov, pi, tau=TAU)
    pd.testing.assert_series_equal(mu, pi.rename("mu_bl"))
    np.testing.assert_allclose(cov_bl, (1 + TAU) * cov)
    empty = pd.DataFrame(columns=ASSETS, dtype=float)
    mu2, _ = posterior(cov, pi, empty, pd.Series(dtype=float), pd.DataFrame(dtype=float), TAU)
    pd.testing.assert_series_equal(mu2, mu)


def test_woodbury_matches_precision_form(prior):
    cov, pi = prior
    P, q, omega = absolute_views([("XLK", 0.02, 0.0009), ("TLT", -0.01, 0.0004)])
    mu, cov_bl = posterior(cov, pi, P, q, omega, TAU)

    s, p, om = cov.to_numpy(), P.to_numpy(), omega.to_numpy()
    m = np.linalg.inv(np.linalg.inv(TAU * s) + p.T @ np.linalg.inv(om) @ p)
    mu_direct = m @ (np.linalg.inv(TAU * s) @ pi.to_numpy() + p.T @ np.linalg.inv(om) @ q.to_numpy())
    np.testing.assert_allclose(mu.to_numpy(), mu_direct, rtol=1e-9)
    np.testing.assert_allclose(cov_bl.to_numpy(), s + m, rtol=1e-9)


def test_single_asset_is_precision_weighted_average():
    cov = pd.DataFrame([[0.0045]], index=["XLK"], columns=["XLK"])
    pi = pd.Series([0.002], index=["XLK"])
    P, q, omega = absolute_views([("XLK", 0.02, 0.0009)], assets=["XLK"])
    mu, _ = posterior(cov, pi, P, q, omega, TAU)
    # prior precision 1/0.000225 = 4444, view precision 1/0.0009 = 1111 -> weights 0.8 / 0.2
    assert mu["XLK"] == pytest.approx(0.8 * 0.002 + 0.2 * 0.02)  # 0.56%


def test_uninformative_view_leaves_prior(prior):
    cov, pi = prior
    P, q, omega = absolute_views([("XLK", 0.5, 1e12)])
    mu, cov_bl = posterior(cov, pi, P, q, omega, TAU)
    np.testing.assert_allclose(mu, pi, atol=1e-12)
    np.testing.assert_allclose(cov_bl, (1 + TAU) * cov, atol=1e-12)


def test_certain_view_is_matched_exactly(prior):
    cov, pi = prior
    P, q, omega = absolute_views([("XLK", 0.03, 1e-14), ("XLE", -0.01, 1e-14)])
    mu, _ = posterior(cov, pi, P, q, omega, TAU)
    assert mu["XLK"] == pytest.approx(0.03, abs=1e-8)
    assert mu["XLE"] == pytest.approx(-0.01, abs=1e-8)


def test_view_moves_correlated_assets():
    cov = pd.DataFrame([[0.004, 0.003, 0.0], [0.003, 0.004, 0.0], [0.0, 0.0, 0.004]],
                       index=["XLK", "XLY", "GLD"], columns=["XLK", "XLY", "GLD"])
    pi = pd.Series(0.002, index=cov.index)
    P, q, omega = absolute_views([("XLK", 0.03, 0.001)], assets=list(cov.index))
    mu, _ = posterior(cov, pi, P, q, omega, TAU)
    assert mu["XLK"] > pi["XLK"]
    assert mu["XLY"] > pi["XLY"]                       # positively correlated: pulled up too
    assert mu["GLD"] == pytest.approx(pi["GLD"])        # uncorrelated: unchanged


def test_two_views_on_same_asset_combine_by_precision(prior):
    cov, pi = prior
    stacked = absolute_views([("XLK", 0.02, 0.0009), ("XLK", 0.04, 0.0018)])
    w1, w2 = 1 / 0.0009, 1 / 0.0018
    combined = absolute_views([("XLK", (0.02 * w1 + 0.04 * w2) / (w1 + w2), 1 / (w1 + w2))])
    mu_a, cov_a = posterior(cov, pi, *stacked, TAU)
    mu_b, cov_b = posterior(cov, pi, *combined, TAU)
    np.testing.assert_allclose(mu_a, mu_b, rtol=1e-10)
    np.testing.assert_allclose(cov_a, cov_b, rtol=1e-10)


def test_posterior_covariance_properties(prior):
    cov, pi = prior
    P, q, omega = absolute_views([("XLK", 0.02, 0.0009), ("XLB", 0.01, 0.0005)])
    _, cov_bl = posterior(cov, pi, P, q, omega, TAU)
    np.testing.assert_allclose(cov_bl, cov_bl.T)
    np.linalg.cholesky(cov_bl.to_numpy())
    m = cov_bl.to_numpy() - cov.to_numpy()
    eig = np.linalg.eigvalsh(m)
    assert eig.min() > -1e-15                           # M is positive semi-definite
    assert eig.max() <= TAU * np.linalg.eigvalsh(cov.to_numpy()).max() + 1e-15  # views reduce M


def test_columns_matched_by_label_and_outside_views_dropped(prior):
    cov, pi = prior
    P, q, omega = absolute_views([("XLK", 0.02, 0.0009)])
    shuffled = P[ASSETS[::-1]]
    pd.testing.assert_series_equal(posterior(cov, pi, shuffled, q, omega, TAU)[0],
                                   posterior(cov, pi, P, q, omega, TAU)[0])

    extra, q2, omega2 = absolute_views([("XLK", 0.02, 0.0009), ("XLRE", 0.05, 0.0009)],
                                       assets=ASSETS + ["XLRE"])
    mu_extra, _ = posterior(cov, pi, extra, q2, omega2, TAU)
    pd.testing.assert_series_equal(mu_extra, posterior(cov, pi, P, q, omega, TAU)[0])


def test_missing_q_rejected(prior):
    cov, pi = prior
    P, q, omega = absolute_views([("XLK", 0.02, 0.0009), ("XLE", 0.01, 0.0009)])
    with pytest.raises(ValueError):
        posterior(cov, pi, P, q.iloc[:1], omega, TAU)
