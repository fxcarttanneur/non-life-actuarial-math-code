"""Chapter 3: claims development triangles built from claim and transaction files.

A triangle is a DataFrame indexed by accident year, with development ages in months
(12, 24, ...) as columns. Cells after the latest valuation date are NaN.
Development age at an annual valuation date: age = 12 x (valuation year - accident year + 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gidata import claim_position, exposure_table, synthetic_portfolio, synthetic_transactions

MEASURES = ("paid", "case_reserve", "incurred", "reported_count", "closed_count", "open_count")


def development_age(accident_year: int, valuation: str) -> int:
    """Months from the start of the accident year to the valuation date (a month end)."""
    v = pd.Timestamp(valuation)
    return (v.year - accident_year) * 12 + v.month


def claim_snapshot(claims: pd.DataFrame, tx: pd.DataFrame, valuation: str) -> pd.DataFrame:
    """One row per claim reported by `valuation`, with its position at that date.

    Columns: accident_year, paid, case_reserve, incurred, reported, closed, open.
    A claim is closed when it is reported, has at least one payment and a zero case
    reserve; reported claims with no payment and a zero reserve are closed without payment
    and also count as closed.
    """
    v = pd.Timestamp(valuation)
    c = claims[pd.to_datetime(claims["report_date"]) <= v].set_index("claim_id")
    pos = claim_position(tx, valuation).reindex(c.index).fillna(0.0)
    has_tx = tx[pd.to_datetime(tx["date"]) <= v]["claim_id"].unique()
    out = pos.copy()
    out["accident_year"] = pd.to_datetime(c["accident_date"]).dt.year
    out["reported"] = 1
    out["closed"] = ((out["case_reserve"] == 0) & out.index.isin(has_tx)).astype(int)
    out["open"] = out["reported"] - out["closed"]
    return out


def triangle(claims: pd.DataFrame, tx: pd.DataFrame, measure: str,
             first_ay: int, last_year: int) -> pd.DataFrame:
    """Cumulative triangle of `measure` at year ends `first_ay` ... `last_year`.

    measure: 'paid', 'case_reserve', 'incurred', 'reported_count', 'closed_count'
    or 'open_count'.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    col = {"reported_count": "reported", "closed_count": "closed",
           "open_count": "open"}.get(measure, measure)
    ays = list(range(first_ay, last_year + 1))
    ages = [12 * k for k in range(1, len(ays) + 1)]
    tri = pd.DataFrame(np.nan, index=pd.Index(ays, name="accident_year"),
                       columns=pd.Index(ages, name="age"))
    for y in ays:
        snap = claim_snapshot(claims, tx, f"{y}-12-31")
        sums = snap.groupby("accident_year")[col].sum()
        for ay in ays:
            if ay <= y:
                tri.loc[ay, 12 * (y - ay + 1)] = float(sums.get(ay, 0.0))
    return tri


def to_incremental(cum: pd.DataFrame) -> pd.DataFrame:
    """Incremental triangle: X[i, j] = C[i, j] - C[i, j-1], first column unchanged."""
    inc = cum.diff(axis=1)
    inc.iloc[:, 0] = cum.iloc[:, 0]
    return inc


def to_cumulative(inc: pd.DataFrame) -> pd.DataFrame:
    """Cumulative triangle from an incremental one (NaN cells stay NaN)."""
    return inc.cumsum(axis=1).where(inc.notna())


def calendar_diagonal(tri: pd.DataFrame, year: int) -> pd.Series:
    """Cells of calendar year `year`: accident year i at age 12 x (year - i + 1)."""
    out = {}
    for ay in tri.index:
        age = 12 * (year - ay + 1)
        if age in tri.columns and pd.notna(tri.loc[ay, age]):
            out[ay] = tri.loc[ay, age]
    return pd.Series(out, name=year, dtype=float)


def to_long(tri: pd.DataFrame, name: str) -> pd.DataFrame:
    """Long format: one row per (accident_year, valuation date) with the value `name`.

    This is the layout expected by most triangle libraries, e.g. chainladder-python.
    """
    rows = []
    for ay in tri.index:
        for age in tri.columns:
            v = tri.loc[ay, age]
            if pd.notna(v):
                rows.append((pd.Timestamp(f"{ay}-01-01"),
                             pd.Timestamp(f"{ay + age // 12 - 1}-12-31"), v))
    return pd.DataFrame(rows, columns=["accident_year", "valuation", name])


def exposure_by_accident_year(policies: pd.DataFrame, years) -> pd.DataFrame:
    """Earned exposure and earned premium by calendar (= accident) year."""
    rows = {}
    for y in years:
        e = exposure_table(policies, y)
        rows[y] = {"earned_exposure": e["earned_exposure"].sum(),
                   "earned_premium": e["earned_premium"].sum()}
    return pd.DataFrame(rows).T.rename_axis("accident_year")


def synthetic_history(first_year: int = 2020, last_year: int = 2025, n_policies: int = 5_000,
                      growth: float = 0.05, severity_trend: float = 0.04, seed: int = 100):
    """Several underwriting years of the Chapter 2 portfolio, with growth and inflation.

    Underwriting year y has n_policies x (1 + growth)^(y - first_year) policies and claim
    sizes inflated by (1 + severity_trend)^(y - first_year). Policy and claim identifiers are
    prefixed with the underwriting year. Returns (policies, claims, transactions).
    """
    pols, cls, txs = [], [], []
    for k, y in enumerate(range(first_year, last_year + 1)):
        n = int(round(n_policies * (1 + growth) ** k))
        mu = 7.6 + k * np.log(1 + severity_trend)
        p, c = synthetic_portfolio(n_policies=n, year=y, sev_mu=mu, seed=seed + k)
        p["policy_id"] = f"{y}-" + p["policy_id"]
        c["policy_id"] = f"{y}-" + c["policy_id"]
        c["claim_id"] = f"{y}-" + c["claim_id"]
        t = synthetic_transactions(c, seed=seed + 1000 + k)
        pols.append(p)
        cls.append(c)
        txs.append(t)
    return (pd.concat(pols, ignore_index=True), pd.concat(cls, ignore_index=True),
            pd.concat(txs, ignore_index=True))
