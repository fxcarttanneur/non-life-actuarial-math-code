"""Chapter 6: Bornhuetter-Ferguson and Benktander methods."""
from __future__ import annotations

import pandas as pd


def bornhuetter_ferguson(latest: pd.Series, cdf: pd.Series, expected: pd.Series) -> pd.DataFrame:
    """BF ultimate = latest + (1 - 1/CDF) x expected ultimate.

    Equivalently a credibility weight Z = 1/CDF on the development ultimate and
    1 - Z on the expected ultimate.
    """
    out = pd.DataFrame({"latest": latest, "cdf": cdf, "expected_ultimate": expected})
    out["pct_unreported"] = 1.0 - 1.0 / out["cdf"]
    out["bf_ibnr"] = out["pct_unreported"] * out["expected_ultimate"]
    out["bf_ultimate"] = out["latest"] + out["bf_ibnr"]
    return out


def benktander(latest: pd.Series, cdf: pd.Series, expected: pd.Series, k: int = 1) -> pd.Series:
    """Benktander ultimate after k iterations of BF, starting from the expected ultimate.

    k = 0 returns the expected ultimate, k = 1 the BF ultimate, k = 2 the classical
    Benktander (Hovinen) ultimate; k -> infinity tends to the development ultimate.
    """
    q = 1.0 - 1.0 / cdf
    u = expected.astype(float).copy()
    for _ in range(k):
        u = latest + q * u
    return u.rename(f"benktander_{k}")


def credibility_weight(cdf: pd.Series, k: int = 1) -> pd.Series:
    """Weight on the development ultimate after k iterations: Z = 1 - (1 - 1/CDF)^k."""
    return 1.0 - (1.0 - 1.0 / cdf) ** k
