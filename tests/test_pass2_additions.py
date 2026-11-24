"""Figures added to Chapters 1-3 and 5-8 by the Pass 2 structural edits (2026-11-23)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gidata import claim_position, exposure_table  # noqa: E402
from gidevelop import age_to_age, average_factors, cumulative_factors, develop  # noqa: E402
from giselect import capped_triangle  # noqa: E402
from gitriangle import (calendar_diagonal, exposure_by_accident_year, synthetic_history,  # noqa: E402
                        to_cumulative, to_incremental, triangle)


@pytest.fixture(scope="module")
def hist():
    return synthetic_history(2020, 2025, seed=100)


# --- Chapter 1 -------------------------------------------------------------

def test_ch1_count_factor_illustration():
    n = 52 * 1.10
    assert n == pytest.approx(57.2)
    assert n / 650 == pytest.approx(0.088)
    assert 262_500 / n == pytest.approx(4589.16, abs=0.005)


def test_ch1_exercise3a_new_data():
    inc = 190_000 + 75_000
    ult = inc * 1.05
    assert (inc, ult, ult - inc) == (265_000, pytest.approx(278_250), pytest.approx(13_250))
    assert [round(100 * x / 390_000, 1) for x in (190_000, inc, ult)] == [48.7, 67.9, 71.3]


# --- Chapter 2 -------------------------------------------------------------

def test_ch2_endorsement_segments():
    seg = pd.DataFrame({"policy_id": ["P001", "P001"], "start": ["2025-01-01", "2025-07-01"],
                        "end": ["2025-06-30", "2025-12-31"],
                        "premium": [730 * 181 / 365, 1095 * 184 / 365]})
    s = exposure_table(seg, 2025)
    assert s["premium"].round(2).tolist() == [362.0, 552.0]
    assert s["earned_premium"].sum() == pytest.approx(914.0)
    assert s["earned_exposure"].sum() == pytest.approx(1.0)


def test_ch2_reconciliation_difference():
    tx = pd.DataFrame({
        "claim_id": ["C01", "C01", "C01", "C02", "C02", "C02", "C02", "C03", "C03", "C04"],
        "date": ["2025-02-12", "2025-03-01", "2025-04-15", "2025-06-20", "2025-09-30",
                 "2025-09-30", "2026-01-20", "2025-08-22", "2025-10-05", "2025-11-20"],
        "type": ["reserve", "payment", "reserve", "reserve", "payment",
                 "reserve", "payment", "reserve", "reserve", "reserve"],
        "amount": [1500, 1200, 0, 8000, 3000, 5500, 2500, 2000, 3500, 4000]})
    paid = claim_position(tx, "2025-12-31")["paid"].sum()
    assert 6_700 - paid == 2_500
    jan = tx[(tx["claim_id"] == "C02") & (tx["date"] == "2026-01-20")]["amount"].item()
    assert jan == 2_500


# --- Chapter 3 -------------------------------------------------------------

def test_ch3_calendar_year_backlog(hist):
    pol, cl, txs = hist
    P = triangle(cl, txs, "paid", 2021, 2025)
    X = to_incremental(P)
    for ay in X.index:
        X.loc[ay, 12 * (2025 - ay + 1)] *= 1.10
    f0, f1 = age_to_age(P), age_to_age(to_cumulative(X))
    latest = {2021: "48-60", 2022: "36-48", 2023: "24-36", 2024: "12-24"}
    assert [round(f0.loc[a, c], 4) for a, c in latest.items()] == [1.0070, 1.0237, 1.1418, 1.8080]
    assert [round(f1.loc[a, c], 4) for a, c in latest.items()] == [1.0077, 1.0261, 1.1560, 1.8888]
    d = calendar_diagonal(to_incremental(P), 2025)
    shares = d / d.sum()
    assert round(shares[2025], 2) == 0.50 and round(shares[2024], 2) == 0.38
    assert round(shares[2021] + shares[2022], 2) == 0.02


# --- Chapter 5 -------------------------------------------------------------

def test_ch5_pricing_elr_after_rate_change():
    assert 0.68 / 1.05 == pytest.approx(0.648, abs=0.0005)


# --- Chapters 6-8 ----------------------------------------------------------

def test_ch7_levelled_pure_premiums(hist):
    pol, cl, txs = hist
    I = triangle(cl, txs, "incurred", 2021, 2025)
    E = exposure_by_accident_year(pol, range(2021, 2026))
    d = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    pp = d["ultimate"] / E["earned_exposure"]
    lvl = [round(pp[y] * 1.04 ** (2025 - y), 2) for y in range(2021, 2026)]
    assert lvl == [531.26, 450.62, 449.58, 510.34, 516.61]
    pct = (1 / d["cdf"]).round(3).tolist()
    assert pct[:4] == [1.0, 0.998, 1.0, 0.991] and pct[4] == 0.885


def test_ch8_mature_years_load(hist):
    pol, cl, txs = hist
    I = triangle(cl, txs, "incurred", 2021, 2025)
    Ic = capped_triangle(cl, txs, 25_000, 2021, 2025)
    dI = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    dc = develop(Ic, cumulative_factors(average_factors(Ic, "volume"), 1.0))
    xs = (dI["ultimate"] - dc["ultimate"]).loc[2021:2023].sum()
    cap = dc["ultimate"].loc[2021:2023].sum()
    assert round(xs) == 509864 and round(cap) == 6298248
    rho = xs / cap
    assert round(rho, 4) == 0.0810
    u = dc.loc[2025, "ultimate"] * (1 + rho)
    assert round(u) == 3171447
    assert u / 3325643.42 - 1 == pytest.approx(-0.046, abs=0.0005)
    c = cl.assign(ay=pd.to_datetime(cl["accident_date"]).dt.year)
    assert ((c["ultimate_amount"] > 25_000) & c["ay"].between(2021, 2023)).sum() == 23
