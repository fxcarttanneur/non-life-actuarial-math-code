"""Chapter 9: claim frequency and severity distributions.

Parameterizations follow STYLE_GUIDE.md section 1.1. scipy.stats uses scale = 1/rate for
the exponential and the gamma; every conversion is made explicitly below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


# ---------------------------------------------------------------- frequency

def claim_counts(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.Series:
    """Number of claims of each policy (zero for policies without a claim)."""
    n = claims.groupby("policy_id").size()
    return n.reindex(policies["policy_id"], fill_value=0).rename("n_claims")


def count_table(counts: pd.Series) -> pd.Series:
    """Number of policies with k = 0, 1, 2, ... claims."""
    t = counts.value_counts().sort_index()
    return t.reindex(range(int(t.index.max()) + 1), fill_value=0).rename("policies")


def poisson_fit(counts: pd.Series) -> float:
    """Poisson MLE (and moment estimate): lambda = sample mean."""
    return float(np.mean(counts))


def negbin_moments(counts: pd.Series) -> tuple[float, float]:
    """Negative binomial (r, p) by moments: p = mean/variance, r = mean p/(1-p).

    Requires variance > mean (overdispersion); raises ValueError otherwise.
    """
    m, v = float(np.mean(counts)), float(np.var(counts, ddof=1))
    if v <= m:
        raise ValueError("variance <= mean: no negative binomial by moments")
    p = m / v
    return m * p / (1 - p), p


def negbin_pmf(k, r: float, p: float):
    """Pr(N = k) = C(k+r-1, k) p^r (1-p)^k (scipy's nbinom uses the same r, p)."""
    return stats.nbinom.pmf(k, r, p)


def ab0_ratios(table: pd.Series) -> pd.Series:
    """k n_k / n_{k-1}, which is linear in k (a k + b) for the (a,b,0) class."""
    k = table.index.to_numpy()[1:]
    r = k * table.to_numpy()[1:] / table.to_numpy()[:-1]
    return pd.Series(r, index=k, name="k_nk_over_nk1")


def expected_counts(table: pd.Series, pmf) -> pd.DataFrame:
    """Observed and expected policies by count; the last row is k or more."""
    n = table.sum()
    k = table.index.to_numpy()
    p = np.asarray(pmf(k), dtype=float)
    p[-1] = 1 - p[:-1].sum()
    return pd.DataFrame({"observed": table.to_numpy(), "expected": n * p}, index=k)


def chi_square(obs_exp: pd.DataFrame, n_params: int, min_expected: float = 5.0):
    """Pearson chi-square, pooling tail cells until each expected count is >= min_expected.

    Returns (statistic, degrees of freedom, p-value).
    """
    o, e = list(obs_exp["observed"]), list(obs_exp["expected"])
    while len(e) > 1 and e[-1] < min_expected:
        lo, le = o.pop(), e.pop()
        o[-1] += lo
        e[-1] += le
    o, e = np.array(o, float), np.array(e, float)
    stat = float(((o - e) ** 2 / e).sum())
    df = len(o) - 1 - n_params
    return stat, df, float(stats.chi2.sf(stat, df))


# ----------------------------------------------------------------- severity

def fit_mle(x, dist: str) -> dict:
    """Maximum likelihood fit. Returns the parameters in the book's parameterization."""
    x = np.asarray(x, float)
    if dist == "exponential":
        return {"beta": 1 / x.mean()}
    if dist == "lognormal":
        lx = np.log(x)
        return {"mu": lx.mean(), "sigma": lx.std(ddof=0)}
    if dist == "gamma":
        a, _, scale = stats.gamma.fit(x, floc=0)
        return {"alpha": a, "beta": 1 / scale}          # rate = 1 / scipy scale
    if dist == "pareto":
        a, _, scale = stats.lomax.fit(x, floc=0)
        return {"alpha": a, "theta": scale}
    if dist == "weibull":
        tau, _, scale = stats.weibull_min.fit(x, floc=0)
        return {"tau": tau, "theta": scale}
    raise ValueError(dist)


def fit_moments(x, dist: str) -> dict:
    """Method of moments (sample mean m and variance v with ddof = 0)."""
    x = np.asarray(x, float)
    m, v = x.mean(), x.var(ddof=0)
    if dist == "exponential":
        return {"beta": 1 / m}
    if dist == "gamma":
        return {"alpha": m * m / v, "beta": m / v}
    if dist == "lognormal":
        s2 = np.log(1 + v / m ** 2)
        return {"mu": np.log(m) - s2 / 2, "sigma": np.sqrt(s2)}
    if dist == "pareto":
        if v <= m * m:
            raise ValueError("Pareto moments need variance > mean^2")
        a = 2 * v / (v - m * m)
        return {"alpha": a, "theta": m * (a - 1)}
    raise ValueError(dist)


def frozen(dist: str, par: dict):
    """scipy frozen distribution from book parameters (rate converted to scale)."""
    if dist == "exponential":
        return stats.expon(scale=1 / par["beta"])
    if dist == "gamma":
        return stats.gamma(par["alpha"], scale=1 / par["beta"])
    if dist == "lognormal":
        return stats.lognorm(par["sigma"], scale=np.exp(par["mu"]))
    if dist == "pareto":
        return stats.lomax(par["alpha"], scale=par["theta"])
    if dist == "weibull":
        return stats.weibull_min(par["tau"], scale=par["theta"])
    raise ValueError(dist)


def loglik(x, dist: str, par: dict) -> float:
    return float(frozen(dist, par).logpdf(np.asarray(x, float)).sum())


def fit_summary(x, dists=("exponential", "gamma", "lognormal", "pareto", "weibull")) -> pd.DataFrame:
    """MLE for each distribution with log-likelihood, AIC, BIC, K-S statistic and fitted mean."""
    x = np.asarray(x, float)
    rows = []
    for d in dists:
        par = fit_mle(x, d)
        ll = loglik(x, d, par)
        k = len(par)
        rows.append({"distribution": d, "params": par, "loglik": ll,
                     "aic": 2 * k - 2 * ll, "bic": k * np.log(x.size) - 2 * ll,
                     "ks": float(stats.kstest(x, frozen(d, par).cdf).statistic),
                     "mean": float(frozen(d, par).mean())})
    return pd.DataFrame(rows).set_index("distribution")


def mean_excess(x, thresholds) -> pd.Series:
    """Empirical mean excess e(d) = average of (X - d) over the claims above d."""
    x = np.asarray(x, float)
    return pd.Series([float((x[x > d] - d).mean()) for d in thresholds],
                     index=list(thresholds), name="mean_excess")


def top_share(x, frac: float) -> float:
    """Share of total claim amount produced by the largest `frac` of claims."""
    x = np.sort(np.asarray(x, float))[::-1]
    k = int(np.ceil(frac * x.size))
    return float(x[:k].sum() / x.sum())


def lognormal_mean_excess(mu: float, sigma: float, d: float) -> float:
    """Theoretical mean excess of the lognormal at d."""
    z = (np.log(d) - mu - sigma ** 2) / sigma
    m = np.exp(mu + sigma ** 2 / 2)
    return float(m * stats.norm.sf(z) / stats.norm.sf((np.log(d) - mu) / sigma) - d)
