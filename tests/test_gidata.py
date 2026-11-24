"""Unit tests for src/gidata.py: every figure quoted in Chapter 2 is checked here."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gidata import (  # noqa: E402
    check_claims,
    claim_position,
    days_on_risk,
    exposure_table,
    last_day_on_risk,
    synthetic_portfolio,
)


@pytest.fixture
def policies():
    return pd.DataFrame({
        "policy_id": ["P001", "P002", "P003", "P004", "P005", "P006"],
        "start": ["2025-01-01", "2025-03-15", "2025-07-01", "2025-10-01", "2025-05-20", "2025-12-01"],
        "end": ["2025-12-31", "2026-03-14", "2026-06-30", "2026-09-30", "2026-05-19", "2026-11-30"],
        "cancel": [None, None, None, None, "2025-09-01", None],
        "region": ["North", "City", "South", "City", "North", "South"],
        "premium": [730.0, 584.0, 900.0, 1095.0, 650.0, 365.0],
    })


@pytest.fixture
def claims():
    return pd.DataFrame({
        "claim_id": ["C01", "C02", "C03", "C04", "C05"],
        "policy_id": ["P001", "P002", "P003", "P004", "P005"],
        "accident_date": ["2025-02-10", "2025-06-03", "2025-08-20", "2025-11-15", "2025-09-10"],
        "report_date": ["2025-02-12", "2025-06-20", "2025-08-22", "2025-11-05", "2025-09-12"],
    })


@pytest.fixture
def tx():
    return pd.DataFrame({
        "claim_id": ["C01", "C01", "C01", "C02", "C02", "C02", "C02", "C03", "C03", "C04"],
        "date": ["2025-02-12", "2025-03-01", "2025-04-15", "2025-06-20", "2025-09-30",
                 "2025-09-30", "2026-01-20", "2025-08-22", "2025-10-05", "2025-11-20"],
        "type": ["reserve", "payment", "reserve", "reserve", "payment",
                 "reserve", "payment", "reserve", "reserve", "reserve"],
        "amount": [1500, 1200, 0, 8000, 3000, 5500, 2500, 2000, 3500, 4000],
    })


@pytest.fixture(scope="module")
def synth():
    return synthetic_portfolio(n_policies=5_000, seed=2025)


# --- Section 2.2 -----------------------------------------------------------

def test_days_on_risk_inclusive():
    s = pd.Series(["2025-06-01"])
    assert days_on_risk(s, s).iloc[0] == 1
    assert days_on_risk(pd.Series(["2026-01-01"]), pd.Series(["2026-12-31"]),
                        "2025-01-01", "2025-12-31").iloc[0] == 0


def test_cancellation_last_day(policies):
    assert last_day_on_risk(policies).iloc[4] == pd.Timestamp("2025-08-31")


def test_days_in_2025(policies):
    t = exposure_table(policies, 2025)
    assert t["earned_days"].tolist() == [365, 292, 184, 92, 104, 31]
    assert (t["term_days"] == 365).all()


def test_cancelled_policy_premium(policies):
    p5 = exposure_table(policies, 2025).iloc[4]
    assert p5["written_premium"] == pytest.approx(185.21, abs=0.005)
    assert p5["earned_premium"] == pytest.approx(185.21, abs=0.005)
    assert p5["unearned_premium"] == pytest.approx(0.0)
    assert 650 - p5["written_premium"] == pytest.approx(464.79, abs=0.005)


def test_p003_split(policies):
    p3 = exposure_table(policies, 2025).iloc[2]
    assert p3["earned_premium"] == pytest.approx(453.70, abs=0.005)
    assert p3["unearned_premium"] == pytest.approx(446.30, abs=0.005)


def test_2025_totals(policies):
    t = exposure_table(policies, 2025)
    assert t["written_premium"].sum() == pytest.approx(3859.21, abs=0.005)
    assert t["earned_exposure"].sum() == pytest.approx(2.9260, abs=0.00005)
    assert t["earned_premium"].sum() == pytest.approx(2143.10, abs=0.005)
    assert t["unearned_premium"].sum() == pytest.approx(1716.10, abs=0.005)
    assert t["unearned_premium"].sum() / t["written_premium"].sum() == pytest.approx(0.44, abs=0.005)


def test_premium_control_total(policies):
    t25, t26 = exposure_table(policies, 2025), exposure_table(policies, 2026)
    assert (t25["earned_premium"] + t25["unearned_premium"]).sum() == pytest.approx(
        t25["written_premium"].sum())
    assert t26["earned_premium"].sum() == pytest.approx(t25["unearned_premium"].sum())
    assert t26["earned_exposure"].sum() == pytest.approx(2.3589, abs=0.00005)
    total = t25["earned_exposure"].sum() + t26["earned_exposure"].sum()
    assert total == pytest.approx(5 + 104 / 365)


def test_region_grouping(policies):
    g = exposure_table(policies, 2025).groupby("region")[["earned_exposure", "earned_premium"]].sum()
    assert g.loc["City"].tolist() == pytest.approx([1.0521, 743.20], abs=0.005)
    assert g.loc["North"].tolist() == pytest.approx([1.2849, 915.21], abs=0.005)
    assert g.loc["South"].tolist() == pytest.approx([0.5890, 484.70], abs=0.005)


# --- Section 2.3 -----------------------------------------------------------

def test_claim_position_year_end(tx):
    cp = claim_position(tx, "2025-12-31")
    assert list(cp.index) == ["C01", "C02", "C03", "C04"]
    assert cp.loc["C02"].tolist() == [3000, 5500, 8500]
    assert cp.sum().tolist() == [4200, 13000, 17200]


def test_claim_position_after_unreduced_payment(tx):
    cp = claim_position(tx, "2026-01-31")
    assert cp.loc["C02", "incurred"] == 11000
    assert cp["incurred"].sum() == 19700


# --- Section 2.4 -----------------------------------------------------------

def test_check_claims(claims, policies):
    issues = check_claims(claims, policies)
    assert issues.values.tolist() == [["C04", "reported before accident"],
                                      ["C05", "accident outside cover"]]


def test_check_claims_unknown_policy_and_duplicate(claims, policies):
    extra = pd.DataFrame({"claim_id": ["C99", "C01"], "policy_id": ["PXXX", "P001"],
                          "accident_date": ["2025-03-01", "2025-03-01"],
                          "report_date": ["2025-03-02", "2025-03-02"]})
    issues = check_claims(pd.concat([claims, extra], ignore_index=True), policies)
    assert ["C99", "unknown policy"] in issues.values.tolist()
    assert ["C01", "duplicate claim_id"] in issues.values.tolist()


# --- Section 2.5 -----------------------------------------------------------

def test_synthetic_is_reproducible():
    a = synthetic_portfolio(n_policies=200, seed=7)
    b = synthetic_portfolio(n_policies=200, seed=7)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_synthetic_figures(synth, policies):
    pol, cl = synth
    assert (len(pol), len(cl)) == (5000, 485)
    assert exposure_table(pol, 2025)["earned_exposure"].sum() == pytest.approx(2508.25, abs=0.005)
    assert cl["ultimate_amount"].mean() == pytest.approx(3873.94, abs=0.005)
    assert cl["ultimate_amount"].median() == pytest.approx(1967.71, abs=0.005)
    assert check_claims(cl, pol).empty
    assert (cl["report_date"] >= cl["accident_date"]).all()


def test_synthetic_theoretical_moments():
    assert math.exp(7.6 + 1.2 ** 2 / 2) == pytest.approx(4105, abs=1)
    assert math.exp(7.6) == pytest.approx(1998, abs=1)


def test_synthetic_region_frequencies(synth):
    pol, cl = synth
    expo = exposure_table(pol, 2025).groupby("region")["earned_exposure"].sum()
    acc = cl[cl["accident_date"] <= "2025-12-31"].merge(pol[["policy_id", "region"]])
    n = acc.groupby("region")["claim_id"].count()
    assert n.to_dict() == {"City": 72, "North": 86, "South": 94}
    assert (n / expo).round(4).to_dict() == {"City": 0.1106, "North": 0.0876, "South": 0.1074}


# --- Section 2.7 exercises -------------------------------------------------

def test_exercise1_leap_year():
    leap = pd.DataFrame({"policy_id": ["L1"], "start": ["2027-09-01"],
                         "end": ["2028-08-31"], "premium": [732.0]})
    a, b = exposure_table(leap, 2027).iloc[0], exposure_table(leap, 2028).iloc[0]
    assert a["term_days"] == 366
    assert (a["earned_days"], b["earned_days"]) == (122, 244)
    assert (a["earned_premium"], b["earned_premium"], a["unearned_premium"]) == pytest.approx((244, 488, 488))
    assert a["earned_exposure"] + b["earned_exposure"] == pytest.approx(1.0027, abs=0.00005)


def test_exercise2_earlier_valuation(tx):
    cp = claim_position(tx, "2025-09-30")
    assert list(cp.index) == ["C01", "C02", "C03"]
    assert cp.sum().tolist() == [4200, 7500, 11700]


def test_exercise3_vehicle_relativities(synth):
    pol, cl = synth
    expo = exposure_table(pol, 2025).groupby("vehicle")["earned_exposure"].sum()
    acc = cl[cl["accident_date"] <= "2025-12-31"].merge(pol[["policy_id", "vehicle"]])
    n = acc.groupby("vehicle")["claim_id"].count()
    assert n.to_dict() == {"A": 116, "B": 72, "C": 64}
    freq = n / expo
    assert (freq / freq["A"]).round(3).tolist() == [1.000, 1.028, 1.513]


# --- Section 2.5.4: claim transactions ---------------------------------------

@pytest.fixture(scope="module")
def synth_tx(synth):
    from gidata import synthetic_transactions
    return synthetic_transactions(synth[1], seed=2026)


def test_transactions_do_not_change_claims(synth, synth_tx):
    pol, cl = synthetic_portfolio(n_policies=5_000, seed=2025)
    pd.testing.assert_frame_equal(cl, synth[1])
    assert len(synth_tx) == 2827


def test_payments_add_up_to_ultimate(synth, synth_tx):
    pay = synth_tx[synth_tx["type"] == "payment"].groupby("claim_id")["amount"].sum()
    ult = synth[1].set_index("claim_id")["ultimate_amount"]
    assert (pay - ult.loc[pay.index]).abs().max() < 1e-6
    assert set(pay.index) == set(ult.index)


def test_transactions_after_report_and_closed(synth, synth_tx):
    rep = synth[1].set_index("claim_id")["report_date"]
    assert (pd.to_datetime(synth_tx["date"]).to_numpy() >= rep.loc[synth_tx["claim_id"]].to_numpy()).all()
    final = claim_position(synth_tx, "2035-12-31")
    assert (final["case_reserve"] == 0).all()


def test_accident_2025_development(synth, synth_tx):
    ids = set(synth[1].loc[synth[1]["accident_date"] <= "2025-12-31", "claim_id"])
    t = synth_tx[synth_tx["claim_id"].isin(ids)]
    cp25 = claim_position(t, "2025-12-31")
    assert len(cp25) == 218
    assert cp25.sum().round(0).tolist() == [330238, 416556, 746794]
    cp26 = claim_position(t, "2026-12-31")
    assert len(cp26) == 252
    assert cp26.sum().round(0).tolist() == [780308, 155186, 935494]
    assert claim_position(t, "2030-12-31")["incurred"].sum() == pytest.approx(948882, abs=1)
