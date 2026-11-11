import numpy as np
import pytest
from scipy import integrate, optimize

from fleetmaint.survival import WeibullAFT, concordance_index, fit_weibull_aft, kaplan_meier, neg_log_lik


def sample(n, k, mu, beta, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    lam = np.exp(mu + X @ np.array(beta))
    return lam * rng.weibull(k, n), X


def test_fit_recovers_parameters_on_clean_sample():
    t, X = sample(6000, 2.5, 6.0, [-0.3, 0.2])
    m = fit_weibull_aft(t, np.ones_like(t), X)
    assert m.shape == pytest.approx(2.5, rel=0.04)
    assert m.mu == pytest.approx(6.0, abs=0.02)
    assert m.beta == pytest.approx([-0.3, 0.2], abs=0.02)


def test_fit_with_right_censoring():
    t, X = sample(6000, 1.8, 5.5, [0.4], seed=1)
    c = np.random.default_rng(2).uniform(0, np.quantile(t, 0.9), len(t))
    m = fit_weibull_aft(np.minimum(t, c), t <= c, X)
    assert m.shape == pytest.approx(1.8, rel=0.06)
    assert m.beta[0] == pytest.approx(0.4, abs=0.04)


def test_fit_with_left_truncation():
    """Units that entered observation at age `entry` and survived to it: the truncated fit stays unbiased, the naive one
    does not."""
    t, X = sample(20000, 2.2, 6.0, [-0.25], seed=3)
    entry = np.random.default_rng(4).uniform(0, 400, len(t))
    keep = t > entry
    t, X, entry = t[keep], X[keep], entry[keep]
    m = fit_weibull_aft(t, np.ones_like(t), X, entry)
    naive = fit_weibull_aft(t, np.ones_like(t), X)
    assert m.mu == pytest.approx(6.0, abs=0.03) and m.shape == pytest.approx(2.2, rel=0.05)
    assert abs(naive.mu - 6.0) > 3 * abs(m.mu - 6.0)


def test_gradient_matches_finite_differences():
    t, X = sample(300, 2.0, 4.0, [0.1, -0.2], seed=5)
    d = (np.arange(300) % 3 != 0).astype(float)
    entry = np.where(np.arange(300) % 4 == 0, t * 0.3, 0)
    y0, has0 = np.log(np.where(entry > 0, entry, 1)), entry > 0
    theta = np.array([3.9, 0.05, -0.1, np.log(0.6)])
    f = lambda th: neg_log_lik(th, np.log(t), d, X, y0, has0)[0]  # noqa: E731
    g = lambda th: neg_log_lik(th, np.log(t), d, X, y0, has0)[1]  # noqa: E731
    assert optimize.check_grad(f, g, theta) < 1e-4 * np.abs(g(theta)).max() + 1e-5


def test_concordance_hand_worked():
    # Comparable pairs (shorter time had the event): (0,1) (0,2) (0,3) (1,2) (1,3). Unit 2 is censored at 6.
    # Concordant when the shorter-lived unit has the higher risk: (0,1) (0,2) (0,3) (1,3). (1,2): 0.5 < 0.7.
    assert concordance_index([2, 4, 6, 8], [1, 1, 0, 1], [0.9, 0.5, 0.7, 0.1]) == pytest.approx(4 / 5)


def test_concordance_ties_and_extremes():
    assert concordance_index([1, 2], [1, 1], [0.5, 0.5]) == 0.5
    t = np.arange(1, 50)
    assert concordance_index(t, np.ones(49), -t) == 1.0
    assert concordance_index(t, np.ones(49), t) == 0.0


def model():
    return WeibullAFT(mu=6.0, beta=np.array([-0.2]), sigma=0.5)


def test_conditional_failure_probability():
    m, x = model(), np.array([[0.5]])
    s = lambda a: m.survival(a, x)[0]  # noqa: E731
    assert m.cond_fail_prob(np.array([300.0]), 14, x)[0] == pytest.approx(1 - s(314) / s(300))
    assert m.cond_survival_curve(300, np.array([0, 14]), x[0])[1] == pytest.approx(s(314) / s(300))


def test_mean_residual_life_matches_integration():
    m, x = model(), np.array([[0.0]])
    for a in (0.0, 200.0, 600.0):
        num, _ = integrate.quad(lambda u: m.survival(u, x)[0], a, np.inf)
        assert m.mean_residual_life(np.array([a]), x)[0] == pytest.approx(num / m.survival(a, x)[0], rel=1e-4)
    assert m.mean_residual_life(np.array([0.0]), x)[0] == pytest.approx(m.mean_life(x)[0], rel=1e-6)
    far = m.mean_residual_life(np.array([5000.0]), x)[0]
    assert np.isfinite(far) and 0 < far < 50  # increasing hazard: little left at very old ages


def test_quantile_inverts_survival():
    m, x = model(), np.array([[1.0]])
    assert m.survival(m.quantile(0.1, x), x)[0] == pytest.approx(0.9)


def test_kaplan_meier_hand_worked():
    ts, s = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert ts.tolist() == [1, 2, 3]
    assert s.tolist() == pytest.approx([0.8, 0.8 * 3 / 4, 0.8 * 3 / 4 * 1 / 2])
