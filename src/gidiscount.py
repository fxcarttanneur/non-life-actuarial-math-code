"""Chapter 23: discounting unpaid claims and risk margins."""
from __future__ import annotations

import numpy as np
import pandas as pd

from givar import odp_fitted


def future_cash_flows(tri: pd.DataFrame, valuation_year: int) -> pd.Series:
    """Expected chain-ladder payments by future calendar year (sum of each future diagonal)."""
    _, fut = odp_fitted(tri)
    out = {}
    for r, i in enumerate(tri.index):
        for k in range(fut.shape[1]):
            v = fut[r, k]
            if not np.isnan(v):
                cy = i + k
                out[cy] = out.get(cy, 0.0) + float(v)
    s = pd.Series(out).sort_index()
    return s[s.index > valuation_year]


def discount_factors(years, valuation_year: int, rates, timing: float = 0.5) -> pd.Series:
    """v_t = (1 + r_t)^-(t - 1 + timing) for payments in year valuation + t (mid-year by default).

    `rates` is a flat rate or a list of spot rates by term (year 1, 2, ...).
    """
    out = {}
    for y in years:
        t = y - valuation_year
        r = rates if np.isscalar(rates) else rates[min(t, len(rates)) - 1]
        out[y] = (1 + r) ** -(t - 1 + timing)
    return pd.Series(out)


def present_value(cf: pd.Series, valuation_year: int, rates, timing: float = 0.5) -> float:
    return float((cf * discount_factors(cf.index, valuation_year, rates, timing)).sum())


def cost_of_capital_margin(cf: pd.Series, scr0: float, valuation_year: int, coc: float = 0.06,
                           rate: float = 0.03) -> dict:
    """Cost-of-capital risk margin: capital runs off with the undiscounted outstanding
    reserve; RM = coc x sum_t SCR_t / (1 + rate)^(t + 1), t = 0, 1, ..."""
    outstanding = cf[::-1].cumsum()[::-1].to_numpy()          # reserve at start of each year
    scr = scr0 * outstanding / outstanding[0]
    disc = (1 + rate) ** -(np.arange(len(scr)) + 1)
    return {"scr": pd.Series(scr, index=cf.index), "margin": float(coc * (scr * disc).sum())}


def confidence_level(sims: np.ndarray, amount: float) -> float:
    """Share of simulated outcomes at or below `amount` (IFRS 17 confidence level)."""
    return float((sims <= amount).mean())
