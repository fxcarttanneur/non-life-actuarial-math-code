"""Chapter 11: aggregate loss models.

S = X_1 + ... + X_N with N and the X_i independent, S = 0 when N = 0. Severities are
discretized on a grid 0, h, 2h, ...; the Panjer recursion, the FFT and Monte Carlo
simulation then give the distribution of S.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ moments

def compound_moments(mean_n: float, var_n: float, mean_x: float, var_x: float) -> tuple[float, float]:
    """E[S] = E[N] E[X]; Var(S) = E[N] Var(X) + Var(N) E[X]^2."""
    return mean_n * mean_x, mean_n * var_x + var_n * mean_x ** 2


def compound_poisson_skewness(lam: float, m2: float, m3: float) -> float:
    """Skewness of a compound Poisson: lam E[X^3] / (lam E[X^2])^(3/2)."""
    return lam * m3 / (lam * m2) ** 1.5


def lognormal_raw_moment(mu: float, sigma: float, k: int) -> float:
    return float(np.exp(k * mu + k * k * sigma * sigma / 2))


# ------------------------------------------------------------ discretization

def discretize(cdf, h: float, m: int) -> np.ndarray:
    """Rounding (mass dispersal) on 0, h, ..., (m-1)h: f_0 = F(h/2),
    f_k = F(kh + h/2) - F(kh - h/2); the mass above the grid goes to the last point."""
    edges = (np.arange(m) + 0.5) * h
    F = cdf(edges)
    f = np.diff(np.concatenate([[0.0], F]))
    f[-1] += 1 - F[-1]
    return f


# --------------------------------------------------------------------- Panjer

def ab0(dist: str, **par) -> tuple[float, float]:
    """(a, b) of the (a,b,0) class."""
    if dist == "poisson":
        return 0.0, par["lam"]
    if dist == "negbin":
        q = 1 - par["p"]
        return q, (par["r"] - 1) * q
    raise ValueError(dist)


def pgf(dist: str, z, **par):
    """Probability generating function E[z^N]."""
    if dist == "poisson":
        return np.exp(par["lam"] * (z - 1))
    if dist == "negbin":
        return (par["p"] / (1 - (1 - par["p"]) * z)) ** par["r"]
    raise ValueError(dist)


def panjer(dist: str, f: np.ndarray, n: int | None = None, **par) -> np.ndarray:
    """Aggregate probabilities g_0, ..., g_{n-1} by the Panjer recursion

    g_0 = P_N(f_0),  g_s = 1/(1 - a f_0) sum_{j=1}^{s} (a + b j / s) f_j g_{s-j}.
    """
    n = len(f) if n is None else n
    a, b = ab0(dist, **par)
    fx = np.zeros(n)
    fx[:min(n, len(f))] = f[:n]
    g = np.zeros(n)
    g[0] = pgf(dist, fx[0], **par)
    j = np.arange(1, n)
    for s in range(1, n):
        jj = j[:s]
        g[s] = np.dot((a + b * jj / s) * fx[1:s + 1], g[s - 1::-1][:s]) / (1 - a * fx[0])
    return g


# ------------------------------------------------------------------------ FFT

def fft_aggregate(dist: str, f: np.ndarray, n: int, **par) -> np.ndarray:
    """Aggregate probabilities on a grid of length n (a power of 2 is fastest):
    g = IFFT( P_N( FFT(f) ) ). The grid must be long enough to avoid wrap-around."""
    fx = np.zeros(n)
    fx[:min(n, len(f))] = f[:n]
    return np.real(np.fft.ifft(pgf(dist, np.fft.fft(fx), **par)))


# ----------------------------------------------------------------- simulation

def simulate(n_sims: int, count_sampler, severity_sampler, seed: int = 11) -> np.ndarray:
    """Monte Carlo aggregate losses: draw N, then N severities, and sum."""
    rng = np.random.default_rng(seed)
    n = count_sampler(rng, n_sims)
    x = severity_sampler(rng, int(n.sum()))
    owner = np.repeat(np.arange(n_sims), n)
    return np.bincount(owner, weights=x, minlength=n_sims)


# ------------------------------------------------------------- approximations

def normal_approx(mean: float, var: float):
    return stats.norm(mean, np.sqrt(var))


def translated_gamma(mean: float, var: float, skew: float):
    """Gamma shifted by x0, matching mean, variance and skewness:
    alpha = 4/skew^2, beta = 2/(skew sd), x0 = mean - alpha/beta. (scipy scale = 1/beta.)"""
    sd = np.sqrt(var)
    alpha = 4 / skew ** 2
    beta = 2 / (skew * sd)
    x0 = mean - alpha / beta
    return stats.gamma(alpha, loc=x0, scale=1 / beta), (alpha, beta, x0)


def lognormal_approx(mean: float, var: float):
    s2 = np.log(1 + var / mean ** 2)
    return stats.lognorm(np.sqrt(s2), scale=np.exp(np.log(mean) - s2 / 2)), (np.log(mean) - s2 / 2, np.sqrt(s2))


# ------------------------------------------------------------ risk measures

def grid_quantile(g: np.ndarray, h: float, q: float) -> float:
    """Smallest grid point s h with Pr(S <= s h) >= q."""
    return float(np.searchsorted(np.cumsum(g), q) * h)


def grid_tvar(g: np.ndarray, h: float, q: float) -> float:
    """Mean of S over its upper (1 - q) tail, with the boundary mass split as needed."""
    G = np.cumsum(g)
    k = int(np.searchsorted(G, q))
    s = np.arange(len(g)) * h
    tail = (s[k + 1:] * g[k + 1:]).sum() + s[k] * (G[k] - q)
    return float(tail / (1 - q))


def grid_moments(g: np.ndarray, h: float) -> tuple[float, float]:
    s = np.arange(len(g)) * h
    m = float((s * g).sum())
    return m, float(((s - m) ** 2 * g).sum())


def sample_tvar(x: np.ndarray, q: float) -> float:
    v = np.quantile(x, q)
    return float(x[x >= v].mean())
