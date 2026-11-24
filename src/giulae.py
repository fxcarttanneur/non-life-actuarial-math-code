"""Chapter 21: unallocated loss adjustment expenses (ULAE)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_activity(tri: pd.DataFrame, first_cy: int) -> pd.Series:
    """Calendar-year increments of a cumulative triangle (sum of each diagonal's increments)."""
    inc = tri.diff(axis=1)
    inc.iloc[:, 0] = tri.iloc[:, 0]
    out = {}
    for i in tri.index:
        for k, age in enumerate(tri.columns):
            cy = i + k
            v = inc.loc[i, age]
            if not np.isnan(v):
                out[cy] = out.get(cy, 0.0) + float(v)
    s = pd.Series(out).sort_index()
    return s[s.index >= first_cy]


def synthetic_ulae(reports: pd.Series, closings: pd.Series, per_report: float = 120.0,
                   per_closing: float = 80.0, inflation: float = 0.03, base_year: int = 2025) -> pd.Series:
    """Calendar-year ULAE paid: per_report x reports + per_closing x closings, at the cost
    level of each year (costs given at `base_year`, deflated by `inflation`)."""
    lvl = pd.Series({y: (1 + inflation) ** (y - base_year) for y in reports.index})
    return (per_report * reports + per_closing * closings) * lvl


def paid_to_paid(ulae_paid: pd.Series, loss_paid: pd.Series) -> pd.Series:
    """Ratio of ULAE paid to losses paid, by calendar year."""
    return ulae_paid / loss_paid


def classical_unpaid(ratio: float, case: float, ibnr: float, share_at_opening: float = 0.5) -> float:
    """Classical paid-to-paid method: ratio x (share_at_opening x case + IBNR).

    The case reserves have had the opening part of their ULAE spent already; IBNR claims
    will need all of it.
    """
    return ratio * (share_at_opening * case + ibnr)


def kittel_ratio(ulae_paid: pd.Series, loss_paid: pd.Series, loss_incurred: pd.Series) -> pd.Series:
    """Kittel refinement: ULAE paid / average of paid and incurred losses."""
    return ulae_paid / ((loss_paid + loss_incurred) / 2)


def count_based_unpaid(cost_per_report: float, cost_per_closing: float,
                       future_reports: float, future_closings: float, inflation: float = 0.0,
                       years_ahead: float = 1.0) -> float:
    """Count-based method: future claim activity x unit costs, inflated to the payment period."""
    return (cost_per_report * future_reports + cost_per_closing * future_closings) * (1 + inflation) ** years_ahead


def unit_costs(ulae_paid: float, reports: float, closings: float, report_weight: float = 0.5) -> tuple[float, float]:
    """Split one year's ULAE paid into a cost per report and per closing, with a share
    `report_weight` of the cost of a claim incurred at the report."""
    per_claim = ulae_paid / (report_weight * reports + (1 - report_weight) * closings)
    return report_weight * per_claim, (1 - report_weight) * per_claim
