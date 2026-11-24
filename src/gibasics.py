"""Chapter 1 helpers: exposures, premiums, ultimates and a first indicated rate.

Time is counted in whole months throughout Chapter 1, so that every figure in the
text can be checked by hand. Chapter 2 replaces this with exact day counts.
"""
from __future__ import annotations

import pandas as pd


def months_between(start: str, valuation: str) -> int:
    """Whole months from the first day of `start` to the day after `valuation`.

    Dates are ISO strings; `start` is a policy inception date (first of a month)
    and `valuation` is a month-end. Example: months_between("2025-04-01",
    "2025-12-31") == 9.
    """
    s = pd.Timestamp(start)
    v = pd.Timestamp(valuation) + pd.Timedelta(days=1)
    return (v.year - s.year) * 12 + (v.month - s.month)


def earned_fraction(start: str, valuation: str, term_months: int = 12) -> float:
    """Fraction of the policy term elapsed at the valuation date, in [0, 1]."""
    m = months_between(start, valuation)
    return min(max(m, 0), term_months) / term_months


def premium_summary(policies: pd.DataFrame, valuation: str) -> pd.DataFrame:
    """Add earned fraction, earned exposure, earned and unearned premium.

    Required columns: start, term_months, units, written_premium.
    `units` is the number of exposure units (e.g. cars) in the row and
    `written_premium` the total premium of the row for its full term.
    Exposure is measured in unit-years.
    """
    df = policies.copy()
    df["earned_fraction"] = [
        earned_fraction(s, valuation, t) for s, t in zip(df["start"], df["term_months"])
    ]
    df["earned_exposure"] = df["units"] * df["earned_fraction"] * df["term_months"] / 12
    df["earned_premium"] = df["written_premium"] * df["earned_fraction"]
    df["unearned_premium"] = df["written_premium"] - df["earned_premium"]
    return df


def ultimate_from_factor(reported: float, factor: float) -> float:
    """Ultimate = reported (incurred or count) x development factor."""
    return reported * factor


def indicated_premium(pure_premium: float, variable_expense: float, profit: float) -> float:
    """Premium covering the pure premium, variable expenses and profit (as % of premium)."""
    return pure_premium / (1.0 - variable_expense - profit)
