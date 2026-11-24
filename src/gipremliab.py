"""Chapter 24: premium liabilities and actual-to-expected monitoring."""
from __future__ import annotations

import pandas as pd

from gidevelop import average_factors, cumulative_factors, develop
from giselect import expected_emergence


def premium_liability(unearned_exposure: float, pure_premium: float, lae_ratio: float,
                      maintenance: float, discount: float = 1.0) -> dict:
    """Expected cost of the unexpired cover: claims + LAE on the unearned exposure, plus
    policy maintenance expenses, discounted."""
    claims = unearned_exposure * pure_premium
    lae = claims * lae_ratio
    total = (claims + lae + maintenance) * discount
    return {"claims": claims, "lae": lae, "maintenance": maintenance, "total": total}


def premium_deficiency(upr: float, dac: float, liability: float) -> float:
    """Additional reserve needed when UPR net of deferred acquisition costs is below the
    premium liability (zero otherwise)."""
    return max(0.0, liability - (upr - dac))


def ae_review(tri_old: pd.DataFrame, tri_new: pd.DataFrame) -> pd.DataFrame:
    """Actual versus expected emergence between two valuations one year apart.

    Expected = prior chain-ladder ultimate x (1/CDF at the new age - 1/CDF at the old age),
    with the factors of the old triangle; actual = change in the latest diagonal.
    """
    cdf = cumulative_factors(average_factors(tri_old, "volume"), 1.0)
    old = develop(tri_old, cdf)
    new_lat = develop(tri_new.loc[old.index], cumulative_factors(average_factors(tri_new, "volume"), 1.0))
    age_new = old["age"] + 12
    cdf_new = age_new.map(lambda a: cdf.get(a, 1.0))
    exp = expected_emergence(old["ultimate"], old["cdf"], cdf_new)
    act = new_lat["latest"] - old["latest"]
    out = pd.DataFrame({"prior_ultimate": old["ultimate"], "expected": exp, "actual": act})
    out["a_minus_e"] = out["actual"] - out["expected"]
    out["a_over_e"] = out["actual"] / out["expected"]
    out["bf_update"] = out["prior_ultimate"] + out["a_minus_e"]
    out["dev_update"] = new_lat["ultimate"]
    return out
