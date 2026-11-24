"""Chapter 7: Cape Cod and frequency-severity methods."""
from __future__ import annotations

import pandas as pd

from gibf import bornhuetter_ferguson


def used_up_base(base: pd.Series, cdf: pd.Series) -> pd.Series:
    """Used-up premium or exposure: base x percent developed (1/CDF)."""
    return (base / cdf).rename("used_up")


def cape_cod_ratio(latest: pd.Series, base: pd.Series, cdf: pd.Series) -> float:
    """Cape Cod (Stanard-Buhlmann) ratio = sum of latest claims / sum of used-up base."""
    return float(latest.sum() / used_up_base(base, cdf).sum())


def cape_cod(latest: pd.Series, base: pd.Series, cdf: pd.Series, level=None) -> pd.DataFrame:
    """Cape Cod ultimates: BF with an expected ratio derived from all accident years.

    `level` optionally restates each year's base to a common level before the ratio is
    computed (e.g. trend factors to the latest year); the expected ultimate of each year is
    then ratio x base x level.
    """
    lvl = pd.Series(1.0, index=base.index) if level is None else level
    adj = base * lvl
    ratio = cape_cod_ratio(latest, adj, cdf)
    expected = ratio * adj
    out = bornhuetter_ferguson(latest, cdf, expected)
    out["ratio"] = ratio
    return out


def average_severity(amounts: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Severity triangle: amount / count cell by cell."""
    return amounts / counts
