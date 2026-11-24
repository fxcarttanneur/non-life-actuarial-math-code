"""Chapter 2: policy and claim data, exact-day exposure, data checks, synthetic portfolio.

Conventions
-----------
* Cover runs from `start` to `end` inclusive (both dates are days on risk).
* A cancelled policy has `cancel` set: the last day on risk is `cancel - 1 day`.
* Exposure is measured in years with an actual/365 day count (a leap day counts as
  1/365 too; Chapter 2 discusses the alternative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def _ts(x):
    return pd.Timestamp(x)


def last_day_on_risk(policies: pd.DataFrame) -> pd.Series:
    """End of cover, taking cancellations into account."""
    end = pd.to_datetime(policies["end"])
    if "cancel" in policies:
        cancel = pd.to_datetime(policies["cancel"])
        end = end.where(cancel.isna(), cancel - DAY)
    return end


def days_on_risk(start, last, period_start=None, period_end=None) -> pd.Series:
    """Days of cover inside [period_start, period_end] (all inclusive)."""
    s = pd.to_datetime(start)
    e = pd.to_datetime(last)
    if period_start is not None:
        s = s.where(s > _ts(period_start), _ts(period_start))
    if period_end is not None:
        e = e.where(e < _ts(period_end), _ts(period_end))
    return ((e - s).dt.days + 1).clip(lower=0)


def exposure_table(policies: pd.DataFrame, year: int) -> pd.DataFrame:
    """Written, earned and unearned premium and earned exposure for calendar `year`.

    Required columns: policy_id, start, end, premium (full-term premium);
    optional: cancel. Written premium is net of the pro rata refund on cancellation.
    """
    df = policies.copy()
    start = pd.to_datetime(df["start"])
    term_days = (pd.to_datetime(df["end"]) - start).dt.days + 1
    last = last_day_on_risk(df)
    covered = days_on_risk(start, last)
    df["term_days"] = term_days
    df["days_covered"] = covered
    df["written_premium"] = df["premium"] * covered / term_days
    in_year = days_on_risk(start, last, f"{year}-01-01", f"{year}-12-31")
    df["earned_days"] = in_year
    df["earned_exposure"] = in_year / 365
    df["earned_premium"] = df["premium"] * in_year / term_days
    before = days_on_risk(start, last, None, f"{year - 1}-12-31")
    df["unearned_premium"] = df["premium"] * (covered - before - in_year) / term_days
    return df


def claim_position(transactions: pd.DataFrame, valuation: str) -> pd.DataFrame:
    """Paid, case reserve and incurred per claim at `valuation` (inclusive).

    `transactions` columns: claim_id, date, type ('payment' or 'reserve'), amount.
    A 'reserve' row is the new case reserve *level* set on that date (not a change).
    Payments reduce nothing automatically: handlers reset the reserve explicitly.
    """
    t = transactions[pd.to_datetime(transactions["date"]) <= _ts(valuation)]
    t = t.sort_values(["claim_id", "date"], kind="stable")
    paid = t[t["type"] == "payment"].groupby("claim_id")["amount"].sum()
    reserve = t[t["type"] == "reserve"].groupby("claim_id")["amount"].last()
    out = pd.DataFrame({"paid": paid, "case_reserve": reserve}).fillna(0.0)
    out["incurred"] = out["paid"] + out["case_reserve"]
    return out


def check_claims(claims: pd.DataFrame, policies: pd.DataFrame) -> pd.DataFrame:
    """Return one row per failed check: claim_id and the reason.

    Checks: unknown policy, accident outside cover, report before accident,
    duplicate claim_id.
    """
    pol = policies.set_index("policy_id")
    last = last_day_on_risk(policies)
    last.index = policies["policy_id"]
    issues = []
    for c in claims.itertuples(index=False):
        acc, rep = _ts(c.accident_date), _ts(c.report_date)
        if c.policy_id not in pol.index:
            issues.append((c.claim_id, "unknown policy"))
            continue
        if not (_ts(pol.at[c.policy_id, "start"]) <= acc <= last[c.policy_id]):
            issues.append((c.claim_id, "accident outside cover"))
        if rep < acc:
            issues.append((c.claim_id, "reported before accident"))
    dup = claims["claim_id"][claims["claim_id"].duplicated()]
    issues += [(d, "duplicate claim_id") for d in dup]
    return pd.DataFrame(issues, columns=["claim_id", "issue"])


def synthetic_portfolio(n_policies: int = 5_000, year: int = 2025, frequency: float = 0.10,
                        sev_mu: float = 7.6, sev_sigma: float = 1.2,
                        mean_report_delay: float = 30.0, seed: int = 2025):
    """Simulate one year of motor policies and their claims.

    Returns (policies, claims). Inception dates are uniform over `year`; terms are 12
    months; each policy has a rating region and vehicle group. Claim counts are Poisson
    with mean `frequency` x relativity x exposure; severities are lognormal; report
    delays are exponential (days). Everything is reproducible from `seed`.
    """
    rng = np.random.default_rng(seed)
    first = _ts(f"{year}-01-01")
    start = first + pd.to_timedelta(rng.integers(0, 365, n_policies), unit="D")
    end = start + pd.DateOffset(years=1) - DAY
    region = rng.choice(["North", "South", "City"], n_policies, p=[0.4, 0.35, 0.25])
    vehicle = rng.choice(["A", "B", "C"], n_policies, p=[0.5, 0.3, 0.2])
    rel = (pd.Series(region).map({"North": 0.9, "South": 1.0, "City": 1.3}).to_numpy()
           * pd.Series(vehicle).map({"A": 0.9, "B": 1.0, "C": 1.25}).to_numpy())
    premium = np.round(600 * rel, 2)
    policies = pd.DataFrame({"policy_id": [f"P{i:05d}" for i in range(1, n_policies + 1)],
                             "start": start, "end": end, "region": region,
                             "vehicle": vehicle, "premium": premium})
    term = (policies["end"] - policies["start"]).dt.days + 1
    n_claims = rng.poisson(frequency * rel * term / 365)
    idx = np.repeat(np.arange(n_policies), n_claims)
    offset = np.floor(rng.uniform(0, 1, idx.size) * term.to_numpy()[idx]).astype(int)
    acc = policies["start"].to_numpy()[idx] + pd.to_timedelta(offset, unit="D").to_numpy()
    delay = np.floor(rng.exponential(mean_report_delay, idx.size)).astype(int)
    claims = pd.DataFrame({"claim_id": [f"C{j:06d}" for j in range(1, idx.size + 1)],
                           "policy_id": policies["policy_id"].to_numpy()[idx],
                           "accident_date": pd.to_datetime(acc),
                           "report_date": pd.to_datetime(acc) + pd.to_timedelta(delay, unit="D"),
                           "ultimate_amount": np.round(rng.lognormal(sev_mu, sev_sigma, idx.size), 2)})
    return policies, claims


def synthetic_transactions(claims: pd.DataFrame, mean_settlement: float = 240.0,
                           max_payments: int = 4, reserve_bias: float = 0.85,
                           reserve_sigma: float = 0.35, seed: int = 2026) -> pd.DataFrame:
    """Simulate the payment and case-reserve history of each claim.

    For every claim of `synthetic_portfolio`:

    * the settlement delay after report is exponential with mean `mean_settlement` days,
      and scales with claim size (large claims take longer: delay x (amount/median)^0.25);
    * the ultimate amount is paid in 1 to `max_payments` instalments at sorted uniform
      dates between report and settlement, with Dirichlet-distributed shares, so the
      payments add up exactly to `ultimate_amount`;
    * an opening case reserve is set on the report date at
      ultimate x `reserve_bias` x lognormal(0, `reserve_sigma`); `reserve_bias` < 1
      means handlers under-reserve on average, which produces upward incurred
      development (IBNER);
    * after each payment the handler resets the reserve to the same proportion of the
      true remaining amount, and to 0 when the claim closes with the final payment.

    Returns a transaction table in the format of `claim_position`
    (claim_id, date, type, amount). A separate seed keeps the claims of
    `synthetic_portfolio` unchanged.
    """
    rng = np.random.default_rng(seed)
    median = float(claims["ultimate_amount"].median())
    rows = []
    for c in claims.itertuples(index=False):
        report = pd.Timestamp(c.report_date)
        ult = float(c.ultimate_amount)
        scale = (ult / median) ** 0.25
        settle = max(1, int(np.floor(rng.exponential(mean_settlement) * scale)))
        k = int(rng.integers(1, max_payments + 1))
        days = np.sort(rng.integers(1, settle + 1, k))
        days[-1] = settle
        shares = rng.dirichlet(np.ones(k))
        pays = np.round(ult * shares, 2)
        pays[-1] = round(ult - pays[:-1].sum(), 2)
        ratio = reserve_bias * float(rng.lognormal(0.0, reserve_sigma))
        rows.append((c.claim_id, report, "reserve", round(ult * ratio, 2)))
        remaining = ult
        for j, (d, p) in enumerate(zip(days, pays)):
            date = report + pd.Timedelta(days=int(d))
            remaining -= p
            rows.append((c.claim_id, date, "payment", float(p)))
            level = 0.0 if j == k - 1 else round(max(remaining, 0.0) * ratio, 2)
            rows.append((c.claim_id, date, "reserve", level))
    return pd.DataFrame(rows, columns=["claim_id", "date", "type", "amount"])
