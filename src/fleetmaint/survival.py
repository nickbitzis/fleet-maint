"""Weibull accelerated failure time model with right censoring, fitted by maximum likelihood with scipy.

log T = mu + x . beta + sigma * W, where W is a standard minimum extreme value variable. That is a Weibull with
shape k = 1 / sigma and scale lambda(x) = exp(mu + x . beta). A negative beta shortens life: an AFT coefficient
multiplies every quantile of the lifetime by exp(beta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special


@dataclass
class WeibullAFT:
    mu: float
    beta: np.ndarray
    sigma: float
    n: int = 0
    events: int = 0

    @property
    def shape(self) -> float:
        return 1.0 / self.sigma

    def scale(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.exp(self.mu + X @ self.beta)

    def survival(self, t, X) -> np.ndarray:
        return np.exp(-((np.maximum(t, 0) / self.scale(X)) ** self.shape))

    def cum_hazard(self, t, lam) -> np.ndarray:
        return (np.maximum(t, 0) / lam) ** self.shape

    def cond_fail_prob(self, age, horizon, X) -> np.ndarray:
        """P(fail within `horizon` days | survived to `age`)."""
        lam = self.scale(X)
        return 1.0 - np.exp(-(self.cum_hazard(age + horizon, lam) - self.cum_hazard(age, lam)))

    def cond_survival_curve(self, age: float, ts: np.ndarray, x: np.ndarray) -> np.ndarray:
        lam = float(self.scale(x)[0])
        return np.exp(-(self.cum_hazard(age + ts, lam) - self.cum_hazard(age, lam)))

    def mean_life(self, X) -> np.ndarray:
        return self.scale(X) * special.gamma(1.0 + self.sigma)

    def quantile(self, p: float, X) -> np.ndarray:
        return self.scale(X) * (-np.log1p(-p)) ** self.sigma

    def mean_residual_life(self, age, X) -> np.ndarray:
        """E[T - a | T > a] = integral of S(t) from a to infinity, divided by S(a)."""
        lam = self.scale(X)
        k = self.shape
        z = np.asarray(self.cum_hazard(age, lam), dtype=float)
        zc = np.minimum(z, 600.0)
        tail = lam / k * special.gamma(1.0 / k) * special.gammaincc(1.0 / k, zc) * np.exp(zc)
        asym = lam / k * np.maximum(z, 1e-12) ** (1.0 / k - 1.0)  # large-z limit, avoids 0 * inf
        return np.where(z > 50, asym, tail)

    def to_dict(self, names) -> dict:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "shape": self.shape,
            "beta": dict(zip(names, self.beta.tolist(), strict=True)),
            "n": self.n,
            "events": self.events,
        }


def neg_log_lik(theta: np.ndarray, y: np.ndarray, d: np.ndarray, X: np.ndarray, y0: np.ndarray, has0: np.ndarray):
    """Negative log likelihood and gradient. Rows with has0 entered observation at age exp(y0) (left truncation):
    their likelihood is divided by S(entry), which adds exp(w0) back."""
    mu, beta, log_s = theta[0], theta[1:-1], theta[-1]
    s = np.exp(log_s)
    lin = mu + X @ beta
    w = (y - lin) / s
    ew = np.exp(np.minimum(w, 50))
    w0 = np.where(has0, (y0 - lin) / s, -np.inf)
    ew0 = np.where(has0, np.exp(np.minimum(w0, 50)), 0.0)
    w0f = np.where(has0, w0, 0.0)
    ll = np.sum(d * (w - log_s) - ew + ew0)
    r = d - ew  # d/dw of the per-row log likelihood
    g_mu = -(np.sum(r) + np.sum(ew0)) / s
    g_beta = -(X.T @ (r + ew0)) / s
    g_ls = np.sum(-r * w - d) - np.sum(ew0 * w0f)
    return -ll, -np.concatenate([[g_mu], g_beta, [g_ls]])


def fit_weibull_aft(t, event, X, entry=None) -> WeibullAFT:
    """Maximum likelihood fit. t: age at failure or censoring; event: 1 if failed; X: covariates; entry: age when
    the unit came under observation (0 or None for parts installed inside the window)."""
    t = np.maximum(np.asarray(t, dtype=float), 0.5)
    d = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(t), -1)
    y = np.log(t)
    e = np.zeros_like(t) if entry is None else np.asarray(entry, dtype=float)
    has0 = e > 0
    y0 = np.log(np.where(has0, e, 1.0))
    theta0 = np.concatenate([[np.log(t.mean()) + 0.3], np.zeros(X.shape[1]), [np.log(0.5)]])
    res = optimize.minimize(neg_log_lik, theta0, args=(y, d, X, y0, has0), jac=True, method="L-BFGS-B")
    th = res.x
    return WeibullAFT(float(th[0]), th[1:-1].copy(), float(np.exp(th[-1])), n=len(t), events=int(d.sum()))


def concordance_index(time, event, risk) -> float:
    """Harrell's C. A pair (i, j) is comparable when t_i < t_j and i had the event. It is concordant when the
    shorter-lived unit has the higher risk score; tied scores count one half."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    r = np.asarray(risk, dtype=float)
    num = 0.0
    den = 0.0
    idx = np.flatnonzero(e)
    for start in range(0, len(idx), 512):
        i = idx[start : start + 512]
        comp = t[i][:, None] < t[None, :]
        num += np.sum(comp & (r[i][:, None] > r[None, :])) + 0.5 * np.sum(comp & (r[i][:, None] == r[None, :]))
        den += np.sum(comp)
    return float(num / den) if den else float("nan")


def kaplan_meier(time, event) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    ts = np.unique(t[e])
    s = 1.0
    out = []
    for u in ts:
        at_risk = np.sum(t >= u)
        s *= 1.0 - np.sum((t == u) & e) / at_risk
        out.append(s)
    return ts, np.array(out)
