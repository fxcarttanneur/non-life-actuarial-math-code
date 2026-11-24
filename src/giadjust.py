"""Chapter 20: adjusting claims data for changing conditions (Berquist-Sherman)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gidata import synthetic_portfolio, synthetic_transactions


def changing_history(first_year: int = 2020, last_year: int = 2025, change_year: int = 2023,
                     bias_before: float = 0.85, bias_after: float = 1.05,
                     settle_before: float = 240.0, settle_after: float = 150.0,
                     n_policies: int = 5_000, growth: float = 0.05,
                     severity_trend: float = 0.04, seed: int = 100):
    """The synthetic history of Chapter 3 with two changes for claims whose accident year is `change_year` or later:
    case reserves are set at `bias_after` instead of `bias_before` times the true remaining
    amount (strengthening), and the mean settlement delay falls from `settle_before` to
    `settle_after` days (faster settlement). Claims and their ultimate amounts are identical
    to `synthetic_history`."""
    pols, cls, txs = [], [], []
    for k, y in enumerate(range(first_year, last_year + 1)):
        n = int(round(n_policies * (1 + growth) ** k))
        mu = 7.6 + k * np.log(1 + severity_trend)
        p, c = synthetic_portfolio(n_policies=n, year=y, sev_mu=mu, seed=seed + k)
        for d in (p, c):
            d["policy_id"] = f"{y}-" + d["policy_id"]
        c["claim_id"] = f"{y}-" + c["claim_id"]
        late = pd.to_datetime(c["accident_date"]).dt.year >= change_year
        parts = []
        for flag, sub in ((False, c[~late]), (True, c[late])):
            if len(sub):
                parts.append(synthetic_transactions(
                    sub, mean_settlement=settle_after if flag else settle_before,
                    reserve_bias=bias_after if flag else bias_before,
                    seed=seed + 1000 + k + (500 if flag else 0)))
        t = pd.concat(parts, ignore_index=True)
        pols.append(p); cls.append(c); txs.append(t)
    return (pd.concat(pols, ignore_index=True), pd.concat(cls, ignore_index=True),
            pd.concat(txs, ignore_index=True))


def average_case(incurred: pd.DataFrame, paid: pd.DataFrame, open_counts: pd.DataFrame) -> pd.DataFrame:
    """Average case reserve per open claim: (incurred - paid) / open count."""
    return (incurred - paid) / open_counts.replace(0, np.nan)


def bs_case_adjust(incurred: pd.DataFrame, paid: pd.DataFrame, open_counts: pd.DataFrame,
                   trend: float) -> pd.DataFrame:
    """Berquist-Sherman case reserve adjustment.

    The average case reserve of the latest diagonal at each age is taken as the current
    level of adequacy and de-trended to earlier accident years: avg[i, j] = avg[i_j, j] /
    (1 + trend)^(i_j - i), where i_j is the latest accident year observed at age j.
    Adjusted incurred = paid + adjusted average x open count.
    """
    avg = average_case(incurred, paid, open_counts)
    adj = incurred.astype(float).copy()
    years = list(incurred.index)
    for age in incurred.columns:
        col = avg[age].dropna()
        if col.empty:
            continue
        i_last = col.index.max()
        for i in col.index:
            level = col[i_last] / (1 + trend) ** (i_last - i)
            adj.loc[i, age] = paid.loc[i, age] + level * open_counts.loc[i, age]
    return adj


def disposal_ratios(closed: pd.DataFrame, ultimate_counts: pd.Series) -> pd.DataFrame:
    """Closed counts as a share of ultimate counts."""
    return closed.div(ultimate_counts, axis=0)


def bs_paid_adjust(paid: pd.DataFrame, closed: pd.DataFrame, ultimate_counts: pd.Series) -> pd.DataFrame:
    """Berquist-Sherman paid adjustment for settlement speed.

    The disposal ratio of the latest diagonal at each age is applied to every accident
    year; adjusted paid is read from each year's own (closed count, paid) points by linear
    interpolation, starting from (0, 0). Points beyond the last observed closed count of a
    year use its last observed paid per closed claim.
    """
    ratio = disposal_ratios(closed, ultimate_counts)
    adj = paid.astype(float).copy()
    for age in paid.columns:
        col = ratio[age].dropna()
        if col.empty:
            continue
        target = col[col.index.max()]
        for i in col.index:
            x = np.concatenate([[0.0], closed.loc[i].dropna().to_numpy(float)])
            y = np.concatenate([[0.0], paid.loc[i].dropna().to_numpy(float)])
            want = target * ultimate_counts[i]
            if want <= x[-1]:
                order = np.argsort(x, kind="mergesort")
                adj.loc[i, age] = float(np.interp(want, x[order], y[order]))
            else:
                adj.loc[i, age] = float(y[-1] / x[-1] * want)
    return adj
