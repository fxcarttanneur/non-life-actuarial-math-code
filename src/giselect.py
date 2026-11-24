"""Chapter 8: tools for selecting an ultimate claim estimate.

Large claims (capped and excess triangles), roll-forward of a prior review
(expected versus actual emergence), and the selection summary with reasonableness ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gitriangle import claim_snapshot


def capped_triangle(claims: pd.DataFrame, tx: pd.DataFrame, cap: float, first_ay: int,
                    last_year: int, part: str = "capped") -> pd.DataFrame:
    """Incurred triangle with each claim's incurred limited to `cap` (part='capped')
    or only the amount above `cap` (part='excess')."""
    ays = list(range(first_ay, last_year + 1))
    ages = [12 * k for k in range(1, len(ays) + 1)]
    tri = pd.DataFrame(np.nan, index=pd.Index(ays, name="accident_year"),
                       columns=pd.Index(ages, name="age"))
    for y in ays:
        snap = claim_snapshot(claims, tx, f"{y}-12-31")
        inc = snap["incurred"].clip(upper=cap) if part == "capped" else (snap["incurred"] - cap).clip(lower=0)
        sums = inc.groupby(snap["accident_year"]).sum()
        for ay in ays:
            if ay <= y:
                tri.loc[ay, 12 * (y - ay + 1)] = float(sums.get(ay, 0.0))
    return tri


def large_claims(claims: pd.DataFrame, tx: pd.DataFrame, valuation: str, threshold: float) -> pd.DataFrame:
    """Claims whose incurred at `valuation` exceeds `threshold`, largest first."""
    snap = claim_snapshot(claims, tx, valuation)
    big = snap[snap["incurred"] > threshold].sort_values("incurred", ascending=False)
    return big[["accident_year", "paid", "case_reserve", "incurred"]]


def expected_emergence(prior_ultimate: pd.Series, cdf_old: pd.Series, cdf_new: pd.Series) -> pd.Series:
    """Claims expected to emerge between two reviews: U x (1/CDF_new - 1/CDF_old)."""
    return (prior_ultimate * (1.0 / cdf_new - 1.0 / cdf_old)).rename("expected_emergence")


def selection_summary(selected: pd.Series, paid: pd.Series, incurred: pd.Series,
                      premium: pd.Series, exposure: pd.Series, counts: pd.Series) -> pd.DataFrame:
    """Reserves and reasonableness ratios for a selected ultimate."""
    out = pd.DataFrame({"selected": selected, "paid": paid, "incurred": incurred})
    out["case_reserve"] = out["incurred"] - out["paid"]
    out["ibnr"] = out["selected"] - out["incurred"]
    out["unpaid"] = out["selected"] - out["paid"]
    out["loss_ratio"] = out["selected"] / premium
    out["pure_premium"] = out["selected"] / exposure
    out["severity"] = out["selected"] / counts
    out["ibnr_to_case"] = out["ibnr"] / out["case_reserve"]
    return out
