"""Chapter 18: credibility (limited fluctuation, Buhlmann, Buhlmann-Straub)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def full_credibility_claims(p: float = 0.90, k: float = 0.05, cv_severity: float = 0.0) -> float:
    """Claims needed for full credibility: (z / k)^2 (1 + CV^2), z the (1+p)/2 normal quantile.

    cv_severity = 0 gives the standard for frequency only.
    """
    z = stats.norm.ppf((1 + p) / 2)
    return float((z / k) ** 2 * (1 + cv_severity ** 2))


def square_root_z(n: float, n_full: float) -> float:
    """Partial credibility by the square-root rule: min(1, sqrt(n / n_full))."""
    return float(min(1.0, np.sqrt(n / n_full)))


def buhlmann(x: pd.DataFrame) -> dict:
    """Buhlmann estimators for a groups x periods table with equal weights.

    EPV = mean of the within-group sample variances; VHM = variance of the group means
    minus EPV / n; k = EPV / VHM; Z = n / (n + k).
    """
    n = x.shape[1]
    means = x.mean(axis=1)
    epv = float(x.var(axis=1, ddof=1).mean())
    vhm = float(means.var(ddof=1) - epv / n)
    mu = float(means.mean())
    k = epv / vhm if vhm > 0 else np.inf
    z = n / (n + k)
    return {"mu": mu, "epv": epv, "vhm": vhm, "k": k, "z": z,
            "premium": mu + z * (means - mu), "means": means}


def buhlmann_straub(x: pd.DataFrame, m: pd.DataFrame) -> dict:
    """Buhlmann-Straub with exposure weights m (same shape as x; NaN for missing cells).

    sigma^2 = sum m_ij (x_ij - xbar_i)^2 / sum (n_i - 1);
    a = [sum m_i (xbar_i - xbar)^2 - sigma^2 (r - 1)] / (m - sum m_i^2 / m);
    Z_i = m_i / (m_i + sigma^2 / a); the complement is the credibility-weighted mean.
    """
    mask = x.notna() & m.notna()
    mi = m.where(mask).sum(axis=1)
    xbar_i = (x * m).where(mask).sum(axis=1) / mi
    mtot = mi.sum()
    xbar = float((mi * xbar_i).sum() / mtot)
    ni = mask.sum(axis=1)
    sigma2 = float((m * (x.sub(xbar_i, axis=0)) ** 2).where(mask).sum().sum() / (ni - 1).sum())
    r = len(mi)
    a = float(((mi * (xbar_i - xbar) ** 2).sum() - sigma2 * (r - 1)) / (mtot - (mi ** 2).sum() / mtot))
    k = sigma2 / a if a > 0 else np.inf
    z = mi / (mi + k)
    mu = float((z * xbar_i).sum() / z.sum())
    return {"sigma2": sigma2, "a": a, "k": k, "z": z, "xbar_i": xbar_i, "xbar": xbar,
            "mu": mu, "premium": z * xbar_i + (1 - z) * mu, "m_i": mi}
