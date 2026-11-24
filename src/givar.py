"""Chapter 22: reserve variability (Mack's model and the over-dispersed Poisson bootstrap)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(tri: pd.DataFrame) -> np.ndarray:
    return tri.to_numpy(float)


def mack(tri: pd.DataFrame) -> dict:
    """Mack (1993) chain-ladder standard errors for a cumulative triangle (no tail).

    Returns factors f_j, variance parameters sigma2_j, the projected square, reserves,
    standard errors by accident year and in total.
    """
    C = _arr(tri)
    n, m = C.shape
    f = np.empty(m - 1)
    s2 = np.empty(m - 1)
    for j in range(m - 1):
        rows = ~np.isnan(C[:, j + 1])
        f[j] = C[rows, j + 1].sum() / C[rows, j].sum()
        k = rows.sum()
        if k > 1:
            s2[j] = (C[rows, j] * (C[rows, j + 1] / C[rows, j] - f[j]) ** 2).sum() / (k - 1)
        else:
            s2[j] = np.nan
    if np.isnan(s2[-1]) and m >= 3:                        # Mack's extrapolation
        s2[-1] = min(s2[-2] ** 2 / s2[-3], s2[-3], s2[-2])
    full = C.copy()
    for i in range(n):
        for j in range(m - 1):
            if np.isnan(full[i, j + 1]):
                full[i, j + 1] = full[i, j] * f[j]
    latest_age = np.array([np.max(np.where(~np.isnan(C[i]))[0]) for i in range(n)])
    latest = C[np.arange(n), latest_age]
    ult = full[:, -1]
    colsum = np.array([np.nansum(C[:n - 1 - j if n - 1 - j > 0 else 0, j]) for j in range(m)])
    # sum of C_kj over accident years with C_k,j+1 observed (denominator of f_j)
    den = np.array([C[~np.isnan(C[:, j + 1]), j].sum() for j in range(m - 1)])
    mse = np.zeros(n)
    for i in range(n):
        for j in range(latest_age[i], m - 1):
            mse[i] += s2[j] / f[j] ** 2 * (1 / full[i, j] + 1 / den[j])
        mse[i] *= ult[i] ** 2
    # total: add covariance terms
    tot = mse.sum()
    for i in range(n):
        for k in range(i + 1, n):
            extra = sum(2 * s2[j] / f[j] ** 2 / den[j] for j in range(latest_age[i], m - 1))
            tot += ult[i] * ult[k] * extra
    idx = tri.index
    return {"f": pd.Series(f, index=tri.columns[:-1]), "sigma2": pd.Series(s2, index=tri.columns[:-1]),
            "full": pd.DataFrame(full, index=idx, columns=tri.columns),
            "latest": pd.Series(latest, index=idx), "ultimate": pd.Series(ult, index=idx),
            "reserve": pd.Series(ult - latest, index=idx), "se": pd.Series(np.sqrt(mse), index=idx),
            "total_reserve": float((ult - latest).sum()), "total_se": float(np.sqrt(tot))}


def odp_fitted(tri: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Chain-ladder fitted incremental values for the observed cells (backward recursion
    from the latest diagonal) and the future incremental means. Returns (fitted, future)."""
    C = _arr(tri)
    n, m = C.shape
    f = np.array([C[~np.isnan(C[:, j + 1]), j + 1].sum() / C[~np.isnan(C[:, j + 1]), j].sum()
                  for j in range(m - 1)])
    fitted_cum = np.full_like(C, np.nan)
    future_cum = np.full_like(C, np.nan)
    for i in range(n):
        last = np.max(np.where(~np.isnan(C[i]))[0])
        fitted_cum[i, last] = C[i, last]
        for j in range(last - 1, -1, -1):
            fitted_cum[i, j] = fitted_cum[i, j + 1] / f[j]
        future_cum[i, last] = C[i, last]
        for j in range(last + 1, m):
            future_cum[i, j] = future_cum[i, j - 1] * f[j - 1]
    fit_inc = np.diff(np.concatenate([np.zeros((n, 1)), fitted_cum], axis=1), axis=1)
    fut_inc = np.diff(future_cum, axis=1)
    fut = np.concatenate([np.full((n, 1), np.nan), fut_inc], axis=1)
    return fit_inc, fut


def odp_bootstrap(tri: pd.DataFrame, n_sims: int = 10_000, seed: int = 22) -> dict:
    """Over-dispersed Poisson bootstrap of the chain-ladder reserve (England and Verrall).

    Pearson residuals of the incremental triangle, scaled by sqrt(n / (n - p)), are
    resampled; each pseudo triangle is refitted by the chain ladder; process variance is
    added by drawing each future increment from a gamma with mean m and variance phi m.
    """
    rng = np.random.default_rng(seed)
    C = _arr(tri)
    n, m = C.shape
    X = np.diff(np.concatenate([np.zeros((n, 1)), C], axis=1), axis=1)
    fit, fut = odp_fitted(tri)
    obs = ~np.isnan(X)
    r = (X[obs] - fit[obs]) / np.sqrt(fit[obs])
    n_obs, p = obs.sum(), n + m - 1
    phi = float((r ** 2).sum() / (n_obs - p))
    r_adj = r * np.sqrt(n_obs / (n_obs - p))
    reserves = np.empty(n_sims)
    by_year = np.empty((n_sims, n))
    for s in range(n_sims):
        rs = rng.choice(r_adj, size=n_obs, replace=True)
        Xs = np.full_like(X, np.nan)
        Xs[obs] = fit[obs] + rs * np.sqrt(fit[obs])
        Cs = np.cumsum(np.where(obs, Xs, 0.0), axis=1)
        Cs[~obs] = np.nan
        _, futs = odp_fitted(pd.DataFrame(Cs))
        mean = np.where(np.isnan(futs), 0.0, np.maximum(futs, 1e-9))
        draw = np.where(mean > 1e-9, rng.gamma(mean / phi, phi), 0.0)
        by_year[s] = draw.sum(axis=1)
        reserves[s] = by_year[s].sum()
    return {"phi": phi, "residuals": r, "reserves": reserves,
            "by_year": pd.DataFrame(by_year, columns=tri.index), "mean_reserve": float(np.nansum(fut))}


def summarize(sims: np.ndarray, qs=(0.5, 0.75, 0.9, 0.95, 0.995)) -> pd.Series:
    out = {"mean": sims.mean(), "sd": sims.std(ddof=1), "cv": sims.std(ddof=1) / sims.mean()}
    out.update({f"q{int(q * 1000) / 10:g}": np.quantile(sims, q) for q in qs})
    return pd.Series(out)
