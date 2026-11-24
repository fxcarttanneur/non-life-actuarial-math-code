"""Chapters 25-27: reinsurance structures, pricing and reserving."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gitriangle import claim_snapshot


# ------------------------------------------------------------- Chapter 25: structures

def layer(x, attach: float, limit: float = np.inf):
    """Amount of each loss x in the layer `limit` xs `attach`: min(max(x - attach, 0), limit)."""
    return np.minimum(np.maximum(np.asarray(x, float) - attach, 0.0), limit)


def quota_share(x, share: float) -> tuple[np.ndarray, np.ndarray]:
    """(ceded, retained) under a quota share ceding `share` of every loss."""
    x = np.asarray(x, float)
    return share * x, (1 - share) * x


def surplus_share(sum_insured, retention: float, lines: int) -> np.ndarray:
    """Ceded proportion of each risk under a surplus treaty with `lines` lines of `retention`."""
    si = np.asarray(sum_insured, float)
    ceded = np.minimum(np.maximum(si - retention, 0.0), lines * retention)
    return ceded / si


def aggregate_layer(s, attach: float, limit: float = np.inf):
    """Stop loss or aggregate layer on annual totals s: min(max(s - attach, 0), limit)."""
    return layer(s, attach, limit)


def xl_with_reinstatements(losses_by_year: list, attach: float, limit: float, n_reinst: int,
                           aad: float = 0.0) -> np.ndarray:
    """Annual recoveries of a per-risk XL with an annual aggregate deductible `aad` and
    `n_reinst` reinstatements (aggregate limit (n_reinst + 1) x limit)."""
    out = []
    for xs in losses_by_year:
        tot = layer(xs, attach, limit).sum()
        out.append(min(max(tot - aad, 0.0), (n_reinst + 1) * limit))
    return np.array(out)


def reinstatement_premium(recovered, limit: float, n_reinst: int, upfront: float,
                          rate: float = 1.0) -> np.ndarray:
    """Pro rata as to amount: rate x upfront x min(recovered, n_reinst x limit) / limit."""
    r = np.asarray(recovered, float)
    return rate * upfront * np.minimum(r, n_reinst * limit) / limit


def risk_transfer(ceded_losses, premium: float) -> dict:
    """10-10 test (probability that the reinsurer loses at least 10% of the premium) and the
    expected reinsurer deficit ERD = E[max(L - P, 0)] / P (all amounts present values)."""
    L = np.asarray(ceded_losses, float)
    loss = L - premium
    return {"p_loss_10": float((loss >= 0.10 * premium).mean()),
            "erd": float(np.maximum(loss, 0.0).mean() / premium),
            "mean_loss_ratio": float(L.mean() / premium)}


# ------------------------------------------------------------- Chapter 26: pricing

def burning_cost(claims: pd.DataFrame, attach: float, limit: float, trend: float,
                 to_year: float, years, exposure: pd.Series) -> pd.DataFrame:
    """Trended burning cost by accident year: trend each ultimate loss to the cost level of
    `to_year` (mid-year accident dates), apply the layer, and divide by the exposure."""
    c = claims.copy()
    ay = pd.to_datetime(c["accident_date"]).dt.year
    c = c[ay.isin(list(years))]
    ay = ay[c.index]
    t = to_year - (ay + 0.5)
    trended = c["ultimate_amount"] * (1 + trend) ** t
    c = c.assign(ay=ay, trended=trended, layer=layer(trended, attach, limit),
                 untrended=layer(c["ultimate_amount"], attach, limit))
    out = c.groupby("ay").agg(n_claims=("layer", lambda s: int((s > 0).sum())),
                              untrended=("untrended", "sum"), layer=("layer", "sum"))
    out["exposure"] = exposure.reindex(out.index)
    out["burning_cost"] = out["layer"] / out["exposure"]
    return out


def exposure_rate(ilf, policy_limit: float, attach: float, limit: float) -> float:
    """Share of a policy's losses falling in the layer, from an ILF curve ilf(u):
    [ILF(min(PL, a + l)) - ILF(min(PL, a))] / ILF(PL)."""
    hi, lo = min(policy_limit, attach + limit), min(policy_limit, attach)
    if hi <= lo:
        return 0.0
    return (ilf(hi) - ilf(lo)) / ilf(policy_limit)


def simulate_layers(lam: float, sampler, layers: list, n_sims: int, seed: int = 26) -> pd.DataFrame:
    """Simulate a year of claims (Poisson count, severity `sampler(rng, n)`) n_sims times and
    return the annual amount in each layer (attach, limit), with the gross total."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam, n_sims)
    x = sampler(rng, int(n.sum()))
    owner = np.repeat(np.arange(n_sims), n)
    out = {"gross": np.bincount(owner, weights=x, minlength=n_sims)}
    for a, l in layers:
        out[f"{l:,.0f} xs {a:,.0f}"] = np.bincount(owner, weights=layer(x, a, l), minlength=n_sims)
    return pd.DataFrame(out)


def simulate_layer_years(lam: float, sampler, n_sims: int, seed: int = 26) -> list:
    """List of per-year arrays of individual losses (for reinstatement and AAD calculations)."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam, n_sims)
    x = sampler(rng, int(n.sum()))
    return np.split(x, np.cumsum(n)[:-1])


def cat_event_set(n_events: int, seed: int = 26) -> pd.DataFrame:
    """A small synthetic event loss table: annual rate and portfolio loss of each event."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(0.002, 0.05, n_events)
    loss = np.round(rng.pareto(1.5, n_events) * 200_000 + 50_000, -3)
    return pd.DataFrame({"rate": rate, "loss": loss}).sort_values("loss", ascending=False,
                                                                  ignore_index=True)


def cat_layer_cost(events: pd.DataFrame, attach: float, limit: float) -> float:
    """Expected annual layer loss from an event loss table: sum of rate x layer loss."""
    return float((events["rate"] * layer(events["loss"], attach, limit)).sum())


# ------------------------------------------------------------- Chapter 27: reserving

def ceded_triangle(claims: pd.DataFrame, tx: pd.DataFrame, measure: str, attach: float,
                   limit: float, first_ay: int, last_year: int, trend: float = 0.0,
                   base_year: int | None = None) -> pd.DataFrame:
    """Cumulative triangle of the layer applied claim by claim to `measure` ('paid' or
    'incurred') at each year end. With `trend`, the attachment and limit are indexed:
    for accident year i they are multiplied by (1 + trend)^(i - base_year)."""
    ays = list(range(first_ay, last_year + 1))
    ages = [12 * k for k in range(1, len(ays) + 1)]
    tri = pd.DataFrame(np.nan, index=pd.Index(ays, name="accident_year"), columns=pd.Index(ages, name="age"))
    base = first_ay if base_year is None else base_year
    for y in ays:
        s = claim_snapshot(claims, tx, f"{y}-12-31")
        s = s[s["accident_year"].isin(ays)]
        k = (1 + trend) ** (s["accident_year"] - base)
        s = s.assign(ceded=layer(s[measure] / k, attach, limit) * k)
        sums = s.groupby("accident_year")["ceded"].sum()
        for ay in ays:
            if ay <= y:
                tri.loc[ay, 12 * (y - ay + 1)] = float(sums.get(ay, 0.0))
    return tri


def true_ceded(claims: pd.DataFrame, attach: float, limit: float, years) -> pd.Series:
    """Ceded ultimate by accident year from the known ultimate claim amounts."""
    ay = pd.to_datetime(claims["accident_date"]).dt.year
    c = claims[ay.isin(list(years))]
    return pd.Series(layer(c["ultimate_amount"], attach, limit), index=c.index).groupby(ay[c.index]).sum()


def excess_factor(f_unl: float, f_lim: float, c_unl: float, c_lim: float) -> float:
    """Excess age-to-age factor implied by unlimited and limited factors:
    f_xs = (f_unl C_unl - f_lim C_lim) / (C_unl - C_lim)."""
    return (f_unl * c_unl - f_lim * c_lim) / (c_unl - c_lim)


def pareto_tail(x, threshold: float) -> float:
    """MLE of alpha for a single-parameter Pareto above `threshold`: n / sum ln(x / threshold)."""
    big = np.asarray(x, float)
    big = big[big > threshold]
    return float(len(big) / np.log(big / threshold).sum())


def pareto_tail_layer(alpha: float, threshold: float, attach: float, limit: float) -> float:
    """Expected layer amount per claim above `threshold` (attach >= threshold), with
    survival (threshold / x)^alpha: integral of the survival over the layer."""
    f = lambda u: threshold ** alpha * u ** (1 - alpha) / (1 - alpha)
    return float(f(attach + limit) - f(attach))
