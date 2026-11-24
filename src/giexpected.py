"""Chapter 5: the expected claims method and actual-versus-expected emergence."""
from __future__ import annotations

import pandas as pd


def expected_ultimate(base: pd.Series, ratio) -> pd.Series:
    """Expected ultimate = base x ratio.

    `base` is earned premium (ratio = expected loss ratio) or earned exposure
    (ratio = expected pure premium). `ratio` may be a scalar or a Series by accident year.
    """
    return (base * ratio).rename("expected_ultimate")


def trend_factor(rate: float, from_year, to_year) -> float:
    """Compound trend (1 + rate)^(to_year - from_year)."""
    return (1.0 + rate) ** (to_year - from_year)


def trended_ratios(ultimate: pd.Series, base: pd.Series, rate: float, to_year: int) -> pd.Series:
    """Historical ratios (ultimate / base) brought to the level of `to_year` at `rate` a year."""
    r = ultimate / base
    return pd.Series({ay: r[ay] * trend_factor(rate, ay, to_year) for ay in r.index}, name=to_year)


def expected_reserves(expected_ult: pd.Series, paid: pd.Series, incurred: pd.Series) -> pd.DataFrame:
    """IBNR and unpaid implied by an expected ultimate."""
    out = pd.DataFrame({"expected_ultimate": expected_ult, "paid": paid, "incurred": incurred})
    out["ibnr"] = out["expected_ultimate"] - out["incurred"]
    out["unpaid"] = out["expected_ultimate"] - out["paid"]
    return out


def actual_vs_expected(expected_ult: pd.Series, latest: pd.Series, cdf_at_age: pd.Series) -> pd.DataFrame:
    """Compare emerged claims with what the expected ultimate and the pattern imply.

    expected emerged = expected_ult / CDF at the current age; ratio = actual / expected emerged.
    """
    out = pd.DataFrame({"expected_ultimate": expected_ult, "cdf": cdf_at_age, "actual": latest})
    out["expected_emerged"] = out["expected_ultimate"] / out["cdf"]
    out["difference"] = out["actual"] - out["expected_emerged"]
    out["ratio"] = out["actual"] / out["expected_emerged"]
    return out
