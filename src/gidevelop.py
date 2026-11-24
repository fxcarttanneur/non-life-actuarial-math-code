"""Chapter 4: the development (chain-ladder) method.

Triangles are DataFrames from gitriangle: accident years as index, ages (months) as
columns, NaN for unknown cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def age_to_age(tri: pd.DataFrame) -> pd.DataFrame:
    """Age-to-age factors f[i, j] = C[i, j+1] / C[i, j]; columns labelled '12-24', ..."""
    ages = list(tri.columns)
    out = pd.DataFrame(index=tri.index)
    for a, b in zip(ages[:-1], ages[1:]):
        out[f"{a}-{b}"] = tri[b] / tri[a]
    return out


def average_factors(tri: pd.DataFrame, method: str = "volume", n: int | None = None) -> pd.Series:
    """Average age-to-age factors per development period.

    method: 'volume' (sum C[j+1] / sum C[j]), 'simple' (mean of factors),
    'medial' (mean excluding highest and lowest, when at least 3 factors),
    'geometric'. `n` keeps only the latest n accident years with a factor.
    """
    ages = list(tri.columns)
    res = {}
    for a, b in zip(ages[:-1], ages[1:]):
        pair = tri[[a, b]].dropna()
        if n is not None:
            pair = pair.tail(n)
        f = pair[b] / pair[a]
        if pair.empty:
            res[f"{a}-{b}"] = np.nan
        elif method == "volume":
            res[f"{a}-{b}"] = pair[b].sum() / pair[a].sum()
        elif method == "simple":
            res[f"{a}-{b}"] = f.mean()
        elif method == "medial":
            res[f"{a}-{b}"] = f.sort_values().iloc[1:-1].mean() if len(f) >= 3 else f.mean()
        elif method == "geometric":
            res[f"{a}-{b}"] = float(np.exp(np.log(f).mean()))
        else:
            raise ValueError("method must be volume, simple, medial or geometric")
    return pd.Series(res)


def cumulative_factors(selected: pd.Series, tail: float = 1.0) -> pd.Series:
    """Age-to-ultimate factors CDF[j] = tail x product of selected factors from age j onward.

    Index: the starting age of each period plus the last age ('12', '24', ...).
    """
    labels = list(selected.index)
    ages = [lab.split("-")[0] for lab in labels] + [labels[-1].split("-")[1]]
    cdf = [tail]
    for f in reversed(selected.to_numpy()):
        cdf.append(cdf[-1] * f)
    return pd.Series(list(reversed(cdf)), index=[int(a) for a in ages], name="cdf")


def fit_exponential_tail(selected: pd.Series, horizon: int = 200) -> dict:
    """Fit f_j - 1 = a exp(-b j) to selected factors (j = 1, 2, ...) and extrapolate a tail.

    The fit is least squares on ln(f_j - 1), so every factor must exceed 1. The tail is the
    product of the fitted factors for periods len(selected) + 1 to `horizon`.
    """
    f = selected.to_numpy(dtype=float)
    if (f <= 1).any():
        raise ValueError("exponential decay needs every selected factor above 1")
    j = np.arange(1, len(f) + 1)
    slope, intercept = np.polyfit(j, np.log(f - 1), 1)
    a, b = float(np.exp(intercept)), float(-slope)
    tail = float(np.prod([1 + a * np.exp(-b * k) for k in range(len(f) + 1, horizon + 1)]))
    fitted = [1 + a * np.exp(-b * k) for k in j]
    return {"a": a, "b": b, "tail": tail, "fitted": fitted}


def latest_diagonal(tri: pd.DataFrame) -> pd.DataFrame:
    """Latest known value and its age for each accident year."""
    rows = {}
    for ay, r in tri.iterrows():
        r = r.dropna()
        rows[ay] = {"age": int(r.index[-1]), "latest": float(r.iloc[-1])}
    return pd.DataFrame(rows).T.astype({"age": int})


def develop(tri: pd.DataFrame, cdf: pd.Series) -> pd.DataFrame:
    """Ultimate = latest x CDF at its age; also percent developed = 1 / CDF."""
    d = latest_diagonal(tri)
    d["cdf"] = d["age"].map(cdf)
    d["pct_developed"] = 1.0 / d["cdf"]
    d["ultimate"] = d["latest"] * d["cdf"]
    return d


def reserve_summary(paid: pd.DataFrame, incurred: pd.DataFrame, ultimate: pd.Series) -> pd.DataFrame:
    """Case reserves, IBNR and unpaid by accident year for a selected ultimate."""
    p = latest_diagonal(paid)["latest"]
    i = latest_diagonal(incurred)["latest"]
    out = pd.DataFrame({"paid": p, "incurred": i, "ultimate": ultimate})
    out["case_reserve"] = out["incurred"] - out["paid"]
    out["ibnr"] = out["ultimate"] - out["incurred"]
    out["unpaid"] = out["ultimate"] - out["paid"]
    return out
