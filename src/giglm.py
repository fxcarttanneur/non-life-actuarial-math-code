"""Chapter 16: generalized linear models for classification ratemaking.

A small, transparent IRLS implementation (log link) for the Poisson, gamma and Tweedie
families, with design matrices for categorical rating factors and validation tools.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def design(df: pd.DataFrame, factors: list, base: dict) -> pd.DataFrame:
    """Intercept plus one dummy column per non-base level of each categorical factor."""
    cols = {"intercept": np.ones(len(df))}
    for f in factors:
        for lev in sorted(df[f].unique()):
            if lev != base[f]:
                cols[f"{f}:{lev}"] = (df[f] == lev).to_numpy(float)
    return pd.DataFrame(cols, index=df.index)


def variance_power(family: str, p: float = 1.5) -> float:
    return {"poisson": 1.0, "gamma": 2.0, "tweedie": p}[family]


def deviance(y, mu, w, family: str, p: float = 1.5) -> float:
    y, mu, w = (np.asarray(a, float) for a in (y, mu, w))
    if family == "poisson":
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu)
        return float(2 * (w * t).sum())
    if family == "gamma":
        return float(2 * (w * (-np.log(y / mu) + (y - mu) / mu)).sum())
    if family == "tweedie":
        t = (np.where(y > 0, y ** (2 - p), 0) / ((1 - p) * (2 - p))
             - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p))
        return float(2 * (w * t).sum())
    raise ValueError(family)


def glm_fit(X: pd.DataFrame, y, family: str, offset=None, weights=None, p: float = 1.5,
            tol: float = 1e-10, max_iter: int = 100) -> dict:
    """Fit a log-link GLM by iteratively reweighted least squares.

    Working weight w * mu^(2 - power) and working response eta - offset + (y - mu) / mu.
    Returns coefficients, standard errors, fitted means, deviance and dispersion.
    """
    Xm = X.to_numpy(float)
    y = np.asarray(y, float)
    n = len(y)
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    w0 = np.ones(n) if weights is None else np.asarray(weights, float)
    vp = variance_power(family, p)
    mu = np.where(y > 0, y, y.mean() if y.mean() > 0 else 1.0) * 0.5 + 0.5 * y.mean()
    eta = np.log(mu)
    beta = np.zeros(Xm.shape[1])
    for it in range(max_iter):
        z = eta - off + (y - mu) / mu
        W = w0 * mu ** (2 - vp)
        XtW = Xm.T * W
        new = np.linalg.solve(XtW @ Xm, XtW @ z)
        done = np.max(np.abs(new - beta)) < tol
        beta = new
        eta = Xm @ beta + off
        mu = np.exp(eta)
        if done:
            break
    W = w0 * mu ** (2 - vp)
    cov = np.linalg.inv((Xm.T * W) @ Xm)
    k = Xm.shape[1]
    phi = 1.0 if family == "poisson" else float((w0 * (y - mu) ** 2 / mu ** vp).sum() / (n - k))
    return {"coef": pd.Series(beta, index=X.columns),
            "se": pd.Series(np.sqrt(np.diag(cov) * phi), index=X.columns),
            "mu": mu, "deviance": deviance(y, mu, w0, family, p), "phi": phi,
            "df_resid": n - k, "iterations": it + 1}


def relativities(fit: dict) -> pd.Series:
    """exp(coefficients): base rate (intercept) and multiplicative relativities."""
    return np.exp(fit["coef"])


def lr_test(small: dict, big: dict, phi: float = 1.0) -> tuple[float, int, float]:
    """Deviance (likelihood ratio) test of a nested model: (statistic, df, p-value)."""
    stat = (small["deviance"] - big["deviance"]) / phi
    df = small["df_resid"] - big["df_resid"]
    return float(stat), int(df), float(stats.chi2.sf(stat, df))


# ------------------------------------------------------------ validation

def lift_table(pred, actual, exposure, n_bins: int = 5) -> pd.DataFrame:
    """Sort by predicted pure premium, form bins of equal exposure, compare actual and predicted."""
    d = pd.DataFrame({"pred": np.asarray(pred, float), "act": np.asarray(actual, float),
                      "e": np.asarray(exposure, float)})
    d = d.sort_values("pred", kind="mergesort")
    cum = d["e"].cumsum() / d["e"].sum()
    d["bin"] = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1) + 1
    g = d.groupby("bin").apply(lambda b: pd.Series({
        "exposure": b.e.sum(), "predicted": (b.pred * b.e).sum() / b.e.sum(),
        "actual": b.act.sum() / b.e.sum()}), include_groups=False)
    return g


def gini(pred, actual, exposure) -> float:
    """Gini index of the ordered Lorenz curve: twice the area between the curve and the diagonal.

    Risks are sorted from lowest to highest predicted pure premium; the curve plots the
    cumulative share of losses against the cumulative share of exposure.
    """
    d = pd.DataFrame({"pred": np.asarray(pred, float), "act": np.asarray(actual, float),
                      "e": np.asarray(exposure, float)}).sort_values("pred", kind="mergesort")
    x = np.concatenate([[0], d.e.cumsum() / d.e.sum()])
    yv = np.concatenate([[0], d.act.cumsum() / d.act.sum()])
    area = np.trapezoid(yv, x)
    return float(1 - 2 * area)


def double_lift(pred_a, pred_b, actual, exposure, n_bins: int = 5) -> pd.DataFrame:
    """Sort by pred_a / pred_b; in each bin compare actual with both predictions (as loss ratios)."""
    a, b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    d = pd.DataFrame({"r": a / b, "a": a, "b": b, "act": np.asarray(actual, float),
                      "e": np.asarray(exposure, float)}).sort_values("r", kind="mergesort")
    cum = d["e"].cumsum() / d["e"].sum()
    d["bin"] = np.minimum((cum * n_bins - 1e-12).astype(int), n_bins - 1) + 1
    return d.groupby("bin").apply(lambda g: pd.Series({
        "ratio_a_b": (g.a * g.e).sum() / (g.b * g.e).sum(),
        "actual_over_a": g.act.sum() / (g.a * g.e).sum(),
        "actual_over_b": g.act.sum() / (g.b * g.e).sum()}), include_groups=False)


def policy_data(policies: pd.DataFrame, claims: pd.DataFrame, years, trend: float = 0.04,
                target: float = 2028.0, seed: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per policy (counts, trended losses) and one row per claim (trended amount).

    Losses are trended from the policy year's average accident date (year + 1) to `target`.
    A random factor `color` with four levels, unrelated to the claims, is added to test
    factor selection.
    """
    pol = policies[policies["start"].dt.year.isin(list(years))].copy()
    pol["policy_year"] = pol["start"].dt.year
    rng = np.random.default_rng(seed)
    pol["color"] = rng.choice(["black", "blue", "red", "white"], len(pol))
    pol["exposure"] = 1.0
    cl = claims.merge(pol[["policy_id", "policy_year", "region", "vehicle", "color"]], on="policy_id")
    cl["amount"] = cl["ultimate_amount"] * (1 + trend) ** (target - (cl["policy_year"] + 1))
    agg = cl.groupby("policy_id").agg(n_claims=("claim_id", "size"), losses=("amount", "sum"))
    pol = pol.merge(agg, on="policy_id", how="left").fillna({"n_claims": 0, "losses": 0.0})
    return pol.reset_index(drop=True), cl
