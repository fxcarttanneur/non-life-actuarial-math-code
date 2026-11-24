"""Chapter 17: increased limits factors, deductible relativities and layer pricing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gicoverage import lev


def ilf_table(dist: str, par: dict, limits, basic: float) -> pd.DataFrame:
    """ILF(u) = E[X ^ u] / E[X ^ basic], with the marginal cost per unit of limit."""
    limits = sorted(limits)
    lv = np.array([lev(dist, par, u) for u in limits])
    out = pd.DataFrame({"lev": lv, "ilf": lv / lev(dist, par, basic)}, index=pd.Index(limits, name="limit"))
    prev_u = np.concatenate([[0.0], limits[:-1]])
    prev_i = np.concatenate([[0.0], out["ilf"].to_numpy()[:-1]])
    out["marginal_per_1000"] = (out["ilf"].to_numpy() - prev_i) / (np.array(limits) - prev_u) * 1000
    return out


def ilf_consistent(ilf: pd.Series) -> pd.DataFrame:
    """Consistency tests: ILFs increase with the limit, and the increase per unit of limit falls."""
    u = np.asarray(ilf.index, float)
    v = ilf.to_numpy(float)
    du = np.diff(np.concatenate([[0.0], u]))
    dv = np.diff(np.concatenate([[0.0], v]))
    per_unit = dv / du
    return pd.DataFrame({"increase": dv > 0,
                         "per_unit": per_unit,
                         "per_unit_falls": np.concatenate([[True], per_unit[1:] <= per_unit[:-1] * (1 + 1e-9)])},
                        index=ilf.index)


def deductible_relativity(dist: str, par: dict, d: float, base_d: float = 0.0,
                          pure_premium: float | None = None, fixed: float = 0.0) -> float:
    """Premium relativity of deductible d to base_d.

    Without expenses: (1 - LER(d)) / (1 - LER(base_d)). With a fixed expense per policy,
    the pure premium is reduced but the fixed expense is not:
    [PP (1 - LER(d)) + F] / [PP (1 - LER(base_d)) + F], PP at no deductible.
    """
    mean = lev(dist, par, np.inf)
    keep = 1 - lev(dist, par, d) / mean
    keep0 = 1 - lev(dist, par, base_d) / mean
    if pure_premium is None:
        return keep / keep0
    return (pure_premium * keep + fixed) / (pure_premium * keep0 + fixed)


def layer_premium(basic_premium: float, ilf: dict, lower: float, upper: float) -> float:
    """Premium of the layer (upper - lower) xs lower: basic premium x (ILF(upper) - ILF(lower))."""
    return basic_premium * (ilf[upper] - ilf[lower])
