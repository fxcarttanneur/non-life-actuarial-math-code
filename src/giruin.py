"""Chapter 12: premium principles, the Cramer-Lundberg model and ruin probabilities.

theta is the relative safety loading: premium rate c = (1 + theta) lam E[X].
"""
from __future__ import annotations

import numpy as np
from scipy import optimize


# ------------------------------------------------------- premium principles

def expected_value_premium(mean: float, theta: float) -> float:
    return (1 + theta) * mean


def variance_premium(mean: float, var: float, a: float) -> float:
    return mean + a * var


def std_premium(mean: float, var: float, kappa: float) -> float:
    return mean + kappa * np.sqrt(var)


def exceedance(g: np.ndarray, h: float, x: float) -> float:
    """Pr(S > x) for a distribution on the grid 0, h, 2h, ..."""
    s = np.arange(len(g)) * h
    return float(g[s > x + 1e-9].sum())


# --------------------------------------------------- Cramer-Lundberg model

def adjustment_coefficient(lam: float, c: float, mgf, upper: float) -> float:
    """Positive root R of lam (M_X(r) - 1) = c r on (0, upper), upper = the MGF's limit."""
    h = lambda r: lam * (mgf(r) - 1) - c * r
    return float(optimize.brentq(h, 1e-12 * upper, upper * (1 - 1e-9)))


def exponential_R(beta: float, theta: float) -> float:
    """R = beta theta / (1 + theta) for exponential claims with rate beta."""
    return beta * theta / (1 + theta)


def exponential_ruin(u: float, beta: float, theta: float) -> float:
    """Infinite-horizon ruin probability, exponential claims: exp(-R u) / (1 + theta)."""
    return float(np.exp(-exponential_R(beta, theta) * u) / (1 + theta))


def loading_for_ruin(u: float, beta: float, target: float) -> float:
    """Loading theta with exponential_ruin(u, beta, theta) = target."""
    return float(optimize.brentq(lambda t: exponential_ruin(u, beta, t) - target, 1e-9, 100))


def capital_for_ruin(beta: float, theta: float, target: float) -> float:
    """Initial surplus u with exponential_ruin(u) = target (0 if target >= 1/(1+theta))."""
    return max(0.0, float(-np.log(target * (1 + theta)) / exponential_R(beta, theta)))


def simulate_ruin(u: float, lam: float, c: float, severity_sampler, horizon: float,
                  n_paths: int, seed: int = 12) -> float:
    """Probability of ruin before `horizon` in the Cramer-Lundberg model, by simulation.

    Ruin can only occur at a claim time t_k: u + c t_k - (X_1 + ... + X_k) < 0.
    """
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam * horizon, n_paths)
    total = int(n.sum())
    path = np.repeat(np.arange(n_paths), n)
    t = rng.uniform(0, horizon, total)
    order = np.lexsort((t, path))
    t = t[order]
    x = severity_sampler(rng, total)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    cum = np.cumsum(x)
    base = np.repeat(np.concatenate([[0.0], cum])[start], n)
    surplus = u + c * t - (cum - base)
    ruined = np.zeros(n_paths, bool)
    has = n > 0
    mins = np.minimum.reduceat(surplus, start[has]) if total else np.array([])
    ruined[has] = mins < 0
    return float(ruined.mean())
