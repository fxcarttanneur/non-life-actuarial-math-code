"""Chapter 28: the complete worked case, one pipeline from data to report.

Every step calls the functions of earlier chapters; this module only chains them and
collects the figures that the actuarial report needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gitriangle import synthetic_history, triangle, exposure_by_accident_year
from gidata import check_claims
from gidevelop import average_factors, cumulative_factors, develop, latest_diagonal
from giselect import capped_triangle, selection_summary
from givar import mack, odp_bootstrap
from giulae import calendar_activity, synthetic_ulae, unit_costs, count_based_unpaid
from gidiscount import future_cash_flows, present_value, cost_of_capital_margin, confidence_level
from gipricing import pure_premium_rate, loss_ratio_change
from gireins import ceded_triangle, true_ceded

YEARS = range(2021, 2026)


def _cdf(t: pd.DataFrame) -> pd.Series:
    return cumulative_factors(average_factors(t, "volume"), 1.0)


def data_summary(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Policies, claims, earned exposure and premium by accident year, and data-check failures."""
    e = exposure_by_accident_year(policies, YEARS)
    ay = pd.to_datetime(claims["accident_date"]).dt.year
    out = e.copy()
    out["claims"] = ay.value_counts().reindex(out.index)
    out.attrs["failed_checks"] = len(check_claims(claims, policies))
    return out


def reserve_review(policies, claims, tx) -> dict:
    """Chapter 8 selection: incurred development, capped development plus excess load for 2025."""
    I = triangle(claims, tx, "incurred", 2021, 2025)
    P = triangle(claims, tx, "paid", 2021, 2025)
    R = triangle(claims, tx, "reported_count", 2021, 2025)
    Ic = capped_triangle(claims, tx, 25_000, 2021, 2025)
    dI, dc = develop(I, _cdf(I)), develop(Ic, _cdf(Ic))
    load = (dI["ultimate"] - dc["ultimate"]).sum() / dc["ultimate"].sum()
    sel = dI["ultimate"].copy()
    sel[2025] = dc.loc[2025, "ultimate"] * (1 + load)
    E = exposure_by_accident_year(policies, YEARS)
    s = selection_summary(sel, latest_diagonal(P)["latest"], dI["latest"], E["earned_premium"],
                          E["earned_exposure"], develop(R, _cdf(R))["ultimate"])
    return {"summary": s, "paid_triangle": P, "load": load}


def variability(paid: pd.DataFrame, selected_unpaid: float, n_sims: int = 10_000) -> dict:
    """Mack and bootstrap on the paid triangle, rescaled to the selected unpaid."""
    m = mack(paid)
    boot = odp_bootstrap(paid, n_sims, seed=22)["reserves"]
    scale = selected_unpaid / boot.mean()
    sims = boot * scale
    return {"mack_cv": m["total_se"] / m["reserve"].sum(), "boot_cv": boot.std() / boot.mean(),
            "sims": sims, "q75": float(np.quantile(sims, 0.75)), "q995": float(np.quantile(sims, 0.995))}


def ulae_reserve(claims, tx) -> float:
    """Count-based unpaid ULAE (Chapter 21)."""
    tri = {m: triangle(claims, tx, m, 2020, 2025) for m in ("reported_count", "closed_count")}
    rep, clo = calendar_activity(tri["reported_count"], 2021), calendar_activity(tri["closed_count"], 2021)
    u_r, u_c = unit_costs(synthetic_ulae(rep, clo)[2025], rep[2025], clo[2025])
    R = triangle(claims, tx, "reported_count", 2021, 2025)
    C = triangle(claims, tx, "closed_count", 2021, 2025)
    dR = develop(R, _cdf(R))
    future_reports = (dR["ultimate"] - dR["latest"]).sum()
    future_closings = (dR["ultimate"] - latest_diagonal(C)["latest"]).sum()
    return count_based_unpaid(u_r, u_c, future_reports, future_closings, inflation=0.03)


def liability(paid: pd.DataFrame, claims_unpaid: float, ulae: float, sims: np.ndarray,
              rate: float = 0.03) -> dict:
    """Discounted best estimate (paid pattern), cost-of-capital risk margin, LIC and its
    confidence level (Chapter 23), for claims plus ULAE."""
    cf = future_cash_flows(paid, 2025)
    cf = cf / cf.sum() * (claims_unpaid + ulae)
    pv = present_value(cf, 2025, rate)
    scr0 = float(np.quantile(sims, 0.995) - sims.mean()) * (claims_unpaid + ulae) / claims_unpaid
    rm = cost_of_capital_margin(cf, scr0, 2025, coc=0.06, rate=rate)["margin"]
    disc_sims = (sims + ulae) * pv / cf.sum()
    return {"undiscounted": float(cf.sum()), "pv": pv, "risk_margin": rm, "lic": pv + rm,
            "confidence": confidence_level(disc_sims, pv + rm)}


def reinsurance(claims, tx, attach: float = 25_000, limit: float = 25_000) -> dict:
    """Ceded reserve of the XL layer by the ceded paid chain ladder (Chapter 27)."""
    cp = ceded_triangle(claims, tx, "paid", attach, limit, 2021, 2025)
    ci = ceded_triangle(claims, tx, "incurred", attach, limit, 2021, 2025)
    ult = develop(cp, _cdf(cp))["ultimate"]
    return {"ceded_ultimate": float(ult.sum()), "ceded_paid": float(latest_diagonal(cp)["latest"].sum()),
            "ceded_incurred": float(latest_diagonal(ci)["latest"].sum()),
            "true_ceded": float(true_ceded(claims, attach, limit, YEARS).sum())}


def indicated_rate(pure_premium: float = 564.15, lae: float = 0.08, fixed: float = 30.0,
                   variable: float = 0.15, profit: float = 0.04, reins_cost: float = 0.0) -> float:
    """Indicated rate per car-year (Chapter 15), with an optional net reinsurance cost per
    car-year treated as part of the cost of claims."""
    return pure_premium_rate(pure_premium * (1 + lae) + reins_cost, fixed, variable, profit)


def run_case() -> dict:
    policies, claims, tx = synthetic_history()
    out = {"data": data_summary(policies, claims)}
    rv = reserve_review(policies, claims, tx)
    s = rv["summary"]
    out["selection"] = s
    unpaid = float(s["unpaid"].sum())
    var = variability(rv["paid_triangle"], unpaid)
    out["variability"] = {k: v for k, v in var.items() if k != "sims"}
    out["ulae"] = ulae_reserve(claims, tx)
    out["liability"] = liability(rv["paid_triangle"], unpaid, out["ulae"], var["sims"])
    out["reinsurance"] = reinsurance(claims, tx)
    return out
