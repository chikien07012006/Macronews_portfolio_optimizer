import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from common.config import load_config
from optimization.black_litterman import posterior
from simulation.features import technical_features
from simulation.views import forward_target, generate_views, training_weeks

ASSETS = ["XLE", "XLK", "TLT"]
N_WEEKS = 260
T = 200                       # position of the decision date in most tests


def make_features(n=N_WEEKS, assets=ASSETS, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n, freq="W-FRI", name="decision_date")
    index = pd.MultiIndex.from_product([dates, assets], names=["decision_date", "ticker"])
    return pd.DataFrame(rng.normal(size=(len(index), 3)), index=index,
                        columns=["MOM", "REV", "VOL"])


def linear_target(features, beta, noise, seed=1, horizon=4):
    """y(i,s) = beta_i * MOM(i,s) + noise_i, last ``horizon`` rows NaN like forward_target."""
    rng = np.random.default_rng(seed)
    mom = features["MOM"].unstack("ticker")
    eps = pd.DataFrame(rng.normal(size=mom.shape), index=mom.index, columns=mom.columns)
    y = mom * pd.Series(beta) + eps * pd.Series(noise)
    y.iloc[-horizon:] = np.nan
    return y


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def setup():
    """XLK and XLE carry a strong MOM signal, TLT is pure noise. At t, MOM(XLK) = +2, MOM(XLE) = -2."""
    feats = make_features()
    t = feats.index.get_level_values("decision_date").unique()[T]
    feats.loc[(t, "XLK"), "MOM"] = 2.0
    feats.loc[(t, "XLE"), "MOM"] = -2.0
    target = linear_target(feats, beta={"XLE": 0.03, "XLK": 0.03, "TLT": 0.0},
                           noise={"XLE": 0.005, "XLK": 0.005, "TLT": 0.03})
    return feats, target, t


# --- forward target -----------------------------------------------------------------------

def test_forward_target_sums_next_weeks():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(20, 2)), columns=["A", "B"],
                     index=pd.date_range("2020-01-03", periods=20, freq="W-FRI"))
    y = forward_target(x, horizon=4)
    for p in range(16):
        np.testing.assert_allclose(y.iloc[p].to_numpy(), x.iloc[p + 1:p + 5].sum().to_numpy())
    assert y.iloc[-4:].isna().all().all()


# --- purging ------------------------------------------------------------------------------

def test_training_weeks_are_purged():
    calendar = pd.date_range("2015-01-02", periods=50, freq="W-FRI")
    for pos in (10, 30, 49):
        weeks = training_weeks(calendar, calendar[pos], horizon=4)
        assert len(weeks) == pos - 3                                  # positions 0 .. pos-4
        assert all(calendar.get_loc(s) + 4 <= pos for s in weeks)
    assert len(training_weeks(calendar, calendar[2], horizon=4)) == 0


def test_unknown_decision_date_rejected(cfg, setup):
    feats, target, _ = setup
    with pytest.raises(ValueError, match="not a decision date"):
        generate_views(feats, target, pd.Timestamp("2016-01-01 12:00"), ASSETS, cfg)


def test_purged_labels_do_not_matter(cfg, setup):
    feats, target, t = setup
    P, q, omega = generate_views(feats, target, t, ASSETS, cfg)

    leaked = target.copy()
    leaked.iloc[T - 3:] = 1e3              # labels not yet known at t (s + 4 > t)
    P2, q2, omega2 = generate_views(feats, leaked, t, ASSETS, cfg)
    pd.testing.assert_series_equal(q, q2)
    pd.testing.assert_frame_equal(omega, omega2)

    known = target.copy()
    known.iloc[T - 4] = 1e3                # the newest label that is known: must matter
    _, q3, _ = generate_views(feats, known, t, ASSETS, cfg)
    assert not q3.equals(q)


# --- forecasting and filtering ------------------------------------------------------------

def test_strong_signal_gives_views_with_the_right_sign(cfg, setup):
    feats, target, t = setup
    P, q, omega = generate_views(feats, target, t, ASSETS, cfg)
    assert list(q.index) == ["tech:XLE", "tech:XLK"]            # TLT is noise: filtered
    assert q["tech:XLK"] == pytest.approx(0.06, rel=0.1)          # 0.03 * MOM(t) = 0.03 * 2
    assert q["tech:XLE"] == pytest.approx(-0.06, rel=0.1)


def test_omega_is_out_of_fold_mse(cfg, setup):
    feats, target, t = setup
    _, q, omega = generate_views(feats, target, t, ASSETS, cfg)

    z = feats.xs("XLK", level="ticker")
    x, y = z.iloc[:T - 3].to_numpy(), target["XLK"].iloc[:T - 3].to_numpy()
    errors = []
    for tr, te in TimeSeriesSplit(n_splits=5, gap=4).split(x):
        errors.append(y[te] - Ridge(alpha=1.0).fit(x[tr], y[tr]).predict(x[te]))
    sigma2 = np.mean(np.concatenate(errors) ** 2)
    q_hat = Ridge(alpha=1.0).fit(x, y).predict(z.loc[[t]].to_numpy())[0]

    assert omega.loc["tech:XLK", "tech:XLK"] == pytest.approx(cfg.views.c_tech * sigma2)
    assert q["tech:XLK"] == pytest.approx(q_hat)


def test_threshold_filters_views(setup):
    feats, target, t = setup
    everything = load_config(overrides={"views.signal_threshold": 0.0})
    nothing = load_config(overrides={"views.signal_threshold": 1e6})
    assert len(generate_views(feats, target, t, ASSETS, everything)[1]) == 3
    assert len(generate_views(feats, target, t, ASSETS, nothing)[1]) == 0


def test_minimum_training_history(setup):
    feats, target, t = setup
    n_train = T - 3                                       # positions 0 .. T-4
    enough = load_config(overrides={"views.min_train_weeks": n_train,
                                    "views.signal_threshold": 0.0})
    short = load_config(overrides={"views.min_train_weeks": n_train + 1,
                                   "views.signal_threshold": 0.0})
    assert len(generate_views(feats, target, t, ASSETS, enough)[1]) == 3
    assert len(generate_views(feats, target, t, ASSETS, short)[1]) == 0


def test_default_needs_156_weeks(cfg, setup):
    feats, target, _ = setup
    dates = feats.index.get_level_values("decision_date").unique()
    loose = load_config(overrides={"views.signal_threshold": 0.0})
    assert loose.views.min_train_weeks == 156
    assert len(generate_views(feats, target, dates[158], ASSETS, loose)[1]) == 0   # 155 weeks
    assert len(generate_views(feats, target, dates[159], ASSETS, loose)[1]) == 3   # 156 weeks


def test_asset_without_features_gets_no_view(setup):
    feats, target, t = setup
    loose = load_config(overrides={"views.signal_threshold": 0.0})
    P, q, _ = generate_views(feats.drop(index=(t, "XLK")), target, t, ASSETS + ["GLD"], loose)
    assert list(q.index) == ["tech:XLE", "tech:TLT"]      # XLK: no features at t; GLD: none at all
    assert list(P.columns) == ["XLE", "XLK", "TLT", "GLD"]


# --- output format ------------------------------------------------------------------------

def test_output_format(cfg, setup):
    feats, target, t = setup
    P, q, omega = generate_views(feats, target, t, ASSETS, cfg)
    assert list(P.columns) == ASSETS
    assert list(P.index) == list(q.index) == list(omega.index) == list(omega.columns)
    assert ((P == 0) | (P == 1)).all().all()
    assert (P.sum(axis=1) == 1).all()
    for label in P.index:
        assert P.loc[label, label.split(":")[1]] == 1.0
    off_diagonal = omega.to_numpy()[~np.eye(len(omega), dtype=bool)]
    assert (off_diagonal == 0).all()
    assert (np.diag(omega) > 0).all()


def test_no_views_returns_empty_but_keeps_columns(setup):
    feats, target, t = setup
    nothing = load_config(overrides={"views.signal_threshold": 1e6})
    P, q, omega = generate_views(feats, target, t, ASSETS, nothing)
    assert P.shape == (0, 3) and list(P.columns) == ASSETS
    assert len(q) == 0 and omega.shape == (0, 0)


def test_omega_scales_with_confidence(cfg, setup):
    feats, target, t = setup
    _, q1, omega1 = generate_views(feats, target, t, ASSETS, cfg, c=1.0)
    _, q2, omega2 = generate_views(feats, target, t, ASSETS, cfg, c=2.0)
    pd.testing.assert_series_equal(q1, q2)
    np.testing.assert_allclose(omega2.to_numpy(), 2 * omega1.to_numpy())
    with pytest.raises(ValueError):
        generate_views(feats, target, t, ASSETS, cfg, c=0.0)


def test_name_prefix(cfg, setup):
    feats, target, t = setup
    P, q, omega = generate_views(feats, target, t, ASSETS, cfg, name="news")
    assert list(q.index) == ["news:XLE", "news:XLK"]


def test_feature_columns_are_not_hard_coded(cfg, setup):
    """The generator uses whatever columns it is given (news features later)."""
    feats, target, t = setup
    renamed = feats.rename(columns={"MOM": "S_macro", "REV": "H", "VOL": "U"})
    _, q, _ = generate_views(feats, target, t, ASSETS, cfg)
    _, q2, _ = generate_views(renamed, target, t, ASSETS, cfg)
    pd.testing.assert_series_equal(q, q2)


def test_views_feed_black_litterman(cfg, setup):
    feats, target, t = setup
    P, q, omega = generate_views(feats, target, t, ASSETS, cfg)
    cov = pd.DataFrame(np.diag([0.01, 0.008, 0.004]), index=ASSETS, columns=ASSETS)
    pi = pd.Series(0.005, index=ASSETS)
    mu, _ = posterior(cov, pi, P, q, omega, tau=cfg.prior.tau)
    assert mu["XLK"] > pi["XLK"]          # positive view pulls the mean up
    assert mu["XLE"] < pi["XLE"]          # negative view pulls it down
    assert mu["TLT"] == pytest.approx(pi["TLT"])   # no view, uncorrelated asset


# --- end to end: no look-ahead ------------------------------------------------------------

def test_no_look_ahead_end_to_end():
    """Changing prices after t leaves features, targets used and views at t unchanged."""
    cfg = load_config(overrides={"views.min_train_weeks": 52, "views.signal_threshold": 0.0})
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2015-01-01", periods=900)
    tickers = ["XLE", "XLK", "TLT", "GLD"]
    daily = pd.DataFrame(rng.normal(0, 0.01, (len(days), 4)), index=days, columns=tickers)
    fridays = pd.DatetimeIndex(days[days.weekday == 4], name="decision_date")
    weekly = pd.DataFrame(rng.normal(0.001, 0.02, (len(fridays), 4)), index=fridays,
                          columns=tickers)
    t = fridays[120]

    def views(w, d):
        return generate_views(technical_features(w, d, cfg), forward_target(w), t, tickers, cfg)

    P, q, omega = views(weekly, daily)
    assert len(q) == 4

    weekly2, daily2 = weekly.copy(), daily.copy()
    weekly2.loc[fridays > t] = rng.uniform(-0.5, 0.5, weekly2.loc[fridays > t].shape)
    daily2.loc[days > t] = rng.uniform(-0.5, 0.5, daily2.loc[days > t].shape)
    P2, q2, omega2 = views(weekly2, daily2)

    pd.testing.assert_frame_equal(P, P2)
    pd.testing.assert_series_equal(q, q2)
    pd.testing.assert_frame_equal(omega, omega2)
