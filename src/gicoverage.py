"""Chapter 10: coverage modifications (deductibles, limits, coinsurance, inflation).

Book parameterizations (STYLE_GUIDE.md 1.1): exponential rate beta, Pareto (Lomax) shape
alpha and scale theta, lognormal mu and sigma. `u` is the maximum covered loss (the loss at
which the limit is reached), `d` the ordinary deductible, `c` the coinsurance share.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def lev(dist: str, par: dict, x: float) -> float:
    """Limited expected value E[X ^ x] in closed form. x = inf gives the mean."""
    if x <= 0:
        return 0.0
    if dist == "exponential":
        b = par["beta"]
        return float((1 - np.exp(-b * x)) / b) if np.isfinite(x) else 1 / b
    if dist == "pareto":
        a, t = par["alpha"], par["theta"]
        if not np.isfinite(x):
            return t / (a - 1)
        if a == 1:
            return float(-t * np.log(t / (t + x)))
        return float(t / (a - 1) * (1 - (t / (t + x)) ** (a - 1)))
    if dist == "lognormal":
        mu, s = par["mu"], par["sigma"]
        mean = np.exp(mu + s * s / 2)
        if not np.isfinite(x):
            return float(mean)
        z = (np.log(x) - mu) / s
        return float(mean * stats.norm.cdf(z - s) + x * stats.norm.sf(z))
    raise ValueError(dist)


def survival(dist: str, par: dict, x: float) -> float:
    if dist == "exponential":
        return float(np.exp(-par["beta"] * x))
    if dist == "pareto":
        return float((par["theta"] / (par["theta"] + x)) ** par["alpha"])
    if dist == "lognormal":
        return float(stats.norm.sf((np.log(x) - par["mu"]) / par["sigma"])) if x > 0 else 1.0
    raise ValueError(dist)


def lev_empirical(x, u: float) -> float:
    """Empirical limited expected value: mean of min(x, u)."""
    return float(np.minimum(np.asarray(x, float), u).mean())


def payment(x, d: float = 0.0, u: float = np.inf, c: float = 1.0):
    """Per-loss payment c [ (X ^ u) - (X ^ d) ] for each loss x."""
    x = np.asarray(x, float)
    return c * (np.minimum(x, u) - np.minimum(x, d))


def per_loss(dist: str, par: dict, d: float = 0.0, u: float = np.inf, c: float = 1.0) -> float:
    """E[Y^L] = c (E[X ^ u] - E[X ^ d])."""
    return c * (lev(dist, par, u) - lev(dist, par, d))


def per_payment(dist: str, par: dict, d: float = 0.0, u: float = np.inf, c: float = 1.0) -> float:
    """E[Y^P] = E[Y^L] / Pr(X > d)."""
    return per_loss(dist, par, d, u, c) / survival(dist, par, d)


def ler(dist: str, par: dict, d: float) -> float:
    """Loss elimination ratio E[X ^ d] / E[X]."""
    return lev(dist, par, d) / lev(dist, par, np.inf)


def franchise_per_loss(dist: str, par: dict, d: float) -> float:
    """Franchise deductible: the full loss is paid once it exceeds d.

    E[Y] = E[X] - E[X ^ d] + d Pr(X > d).
    """
    return lev(dist, par, np.inf) - lev(dist, par, d) + d * survival(dist, par, d)


def inflate(dist: str, par: dict, r: float) -> dict:
    """Parameters of (1 + r) X: a pure change of scale."""
    if dist == "exponential":
        return {"beta": par["beta"] / (1 + r)}
    if dist == "pareto":
        return {"alpha": par["alpha"], "theta": par["theta"] * (1 + r)}
    if dist == "lognormal":
        return {"mu": par["mu"] + np.log(1 + r), "sigma": par["sigma"]}
    raise ValueError(dist)


def layer_cost(dist: str, par: dict, attach: float, width: float) -> float:
    """Expected cost per loss of the layer `width` xs `attach`: E[X ^ (a + w)] - E[X ^ a]."""
    return lev(dist, par, attach + width) - lev(dist, par, attach)
