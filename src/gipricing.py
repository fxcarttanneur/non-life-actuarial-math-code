"""Chapters 13-15: pure premium, trend, on-level premium and indicated rates."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------ Chapter 13

def policy_year_experience(policies: pd.DataFrame, claims: pd.DataFrame, by=None) -> pd.DataFrame:
    """Exposure (car-years), claim count and ultimate claims by policy year (and `by` columns).

    Every policy has a 12-month term, so one policy is one car-year. The policy year is
    taken from the policy identifier prefix (synthetic_history) or the inception year.
    """
    pol = policies.copy()
    pol["policy_year"] = pol["start"].dt.year
    keys = ["policy_year"] + ([] if by is None else list(by))
    cl = claims.merge(pol[["policy_id"] + keys], on="policy_id")
    out = pol.groupby(keys).agg(exposure=("policy_id", "size"), premium=("premium", "sum"))
    out = out.join(cl.groupby(keys).agg(claims=("claim_id", "size"), losses=("ultimate_amount", "sum")))
    out[["claims", "losses"]] = out[["claims", "losses"]].fillna(0)
    return add_ratios(out)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["frequency"] = df["claims"] / df["exposure"]
    df["severity"] = df["losses"] / df["claims"]
    df["pure_premium"] = df["losses"] / df["exposure"]
    if "premium" in df:
        df["loss_ratio"] = df["losses"] / df["premium"]
    return df


def one_way(df: pd.DataFrame, factor: str, base=None) -> pd.DataFrame:
    """One-way pure premium relativities by `factor` (base level = 1.000)."""
    g = add_ratios(df.groupby(factor)[["exposure", "claims", "losses"]].sum())
    base = g["exposure"].idxmax() if base is None else base
    g["relativity"] = g["pure_premium"] / g.loc[base, "pure_premium"]
    return g


# ------------------------------------------------------------ Chapter 14

def fit_trend(values: pd.Series, weights=None) -> tuple[float, float]:
    """Exponential trend: least squares of ln(value) on the year. Returns (annual trend, intercept)."""
    x = np.asarray(values.index, float)
    y = np.log(np.asarray(values, float))
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    return float(np.exp(beta[1]) - 1), float(beta[0])


def trend_factor(rate: float, years: float) -> float:
    return float((1 + rate) ** years)


def average_accident_date(effective: float, policy_term: float = 1.0, writing_period: float = 1.0) -> float:
    """Average accident date (in years) of policies written uniformly over `writing_period`
    from `effective`, each with term `policy_term`: effective + writing/2 + term/2."""
    return effective + writing_period / 2 + policy_term / 2


def earned_share_before(change: float, year: int, term: float = 1.0, n: int = 200_000) -> float:
    """Share of calendar-year `year` earned premium from policies written before `change`
    (annual policies written uniformly; the parallelogram method)."""
    t = year + (np.arange(n) + 0.5) / n                  # earning instants in the year
    return float(np.clip((change - (t - term)) / term, 0, 1).mean())


def parallelogram(rate_changes: dict, years, term: float = 1.0) -> pd.DataFrame:
    """Average rate level of each calendar year's earned premium and on-level factor.

    rate_changes: {effective date in decimal years: change, e.g. 2023.5: 0.05}.
    """
    dates = sorted(rate_changes)
    levels = np.cumprod([1.0] + [1 + rate_changes[d] for d in dates])
    rows = []
    for y in years:
        # share earned at each level: differences of the "written before" shares
        before = [earned_share_before(d, y, term) for d in dates]
        shares = np.diff(np.concatenate([[0.0], before, [1.0]]))
        avg = float((shares * levels).sum())
        rows.append({"year": y, "average_level": avg, "on_level_factor": levels[-1] / avg})
    return pd.DataFrame(rows).set_index("year")


# ------------------------------------------------------------ Chapter 15

def pure_premium_rate(pure_premium: float, fixed_per_exposure: float, variable: float, profit: float) -> float:
    """Indicated rate per exposure: (pure premium incl. LAE + fixed expense) / (1 - V - Q)."""
    return (pure_premium + fixed_per_exposure) / (1 - variable - profit)


def loss_ratio_change(loss_ratio: float, fixed_ratio: float, variable: float, profit: float) -> float:
    """Indicated rate change: (projected loss & LAE ratio + fixed expense ratio) / (1 - V - Q) - 1."""
    return (loss_ratio + fixed_ratio) / (1 - variable - profit) - 1


def claims_made_step_factors(report_shares) -> pd.Series:
    """Step factors for claims-made years 1, 2, ...: cumulative share of an accident year's
    claims reported within k years, relative to a mature claims-made policy (all lags)."""
    s = np.asarray(report_shares, float)
    s = s / s.sum()
    return pd.Series(np.cumsum(s), index=range(1, len(s) + 1), name="step_factor")
