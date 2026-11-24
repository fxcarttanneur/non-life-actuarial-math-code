"""Unit tests for src/giexpected.py: every figure quoted in Chapter 5 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gidevelop import average_factors, cumulative_factors, develop, latest_diagonal  # noqa: E402
from giexpected import (  # noqa: E402
    actual_vs_expected,
    expected_reserves,
    expected_ultimate,
    trend_factor,
    trended_ratios,
)
from gitriangle import exposure_by_accident_year, synthetic_history, triangle  # noqa: E402


@pytest.fixture
def T():
    return pd.DataFrame([[1000, 1500, 1650, 1700], [1100, 1700, 1850, np.nan],
                         [1200, 1750, np.nan, np.nan], [1300, np.nan, np.nan, np.nan]],
                        index=pd.Index([2022, 2023, 2024, 2025], name="accident_year"),
                        columns=pd.Index([12, 24, 36, 48], name="age"))


@pytest.fixture
def EP(T):
    return pd.Series([2500, 2700, 2900, 3100.0], index=T.index)


@pytest.fixture(scope="module")
def synth():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    P = triangle(cl, txs, "paid", 2021, 2025)
    E = exposure_by_accident_year(pol, range(2021, 2026))
    dI = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    dP = develop(P, cumulative_factors(average_factors(P, "volume"), 1.0))
    mature = dI["ultimate"].loc[2021:2023]
    pp = {y: trended_ratios(mature, E["earned_exposure"].loc[2021:2023], 0.04, y).mean()
          for y in (2024, 2025)}
    eu = pd.Series({y: E.loc[y, "earned_exposure"] * pp[y] for y in (2024, 2025)})
    return dict(I=I, P=P, E=E, dI=dI, dP=dP, mature=mature, pp=pp, eu=eu, cl=cl)


# --- Section 5.1-5.2: example triangle --------------------------------------------

def test_expected_method_example(T, EP):
    eu = expected_ultimate(EP, 0.70)
    assert eu.tolist() == pytest.approx([1750, 1890, 2030, 2170])
    ibnr = eu - latest_diagonal(T)["latest"]
    assert ibnr.tolist() == pytest.approx([50, 40, 280, 870])
    assert ibnr.sum() == pytest.approx(1240)
    dev = develop(T, cumulative_factors(average_factors(T, "volume"), 1.0))
    assert (dev["ultimate"] / EP).round(3).tolist() == [0.68, 0.706, 0.68, 0.709]
    assert (eu / dev["ultimate"] - 1).abs().max() < 0.03


def test_exercise1_elr65(T, EP):
    ibnr = expected_ultimate(EP, 0.65) - latest_diagonal(T)["latest"]
    assert ibnr.tolist() == pytest.approx([-75, -95, 135, 715])
    assert ibnr.sum() == pytest.approx(680)
    assert 680 / 1240 - 1 == pytest.approx(-0.45, abs=0.005)


def test_exercise2_mature_elr(T, EP):
    dev = develop(T, cumulative_factors(average_factors(T, "volume"), 1.0))
    elr = (dev["ultimate"] / EP).loc[[2022, 2023]].mean()
    assert elr == pytest.approx(0.6930, abs=0.00005)
    eu = expected_ultimate(EP, elr)
    assert eu.loc[[2024, 2025]].round(2).tolist() == [2009.63, 2148.22]


def test_ave_example(T, EP):
    dev = develop(T, cumulative_factors(average_factors(T, "volume"), 1.0))
    ave = actual_vs_expected(expected_ultimate(EP, 0.70), latest_diagonal(T)["latest"], dev["cdf"])
    assert ave["expected_emerged"].round(2).tolist() == [1750.0, 1834.41, 1801.41, 1283.76]
    assert ave["ratio"].round(3).tolist() == [0.971, 1.008, 0.971, 1.013]


# --- Section 5.3-5.4: synthetic portfolio -----------------------------------------

def test_trend_factor():
    assert trend_factor(0.04, 2021, 2025) == pytest.approx(1.1699, abs=0.00005)


def test_pure_premiums(synth):
    E, mature = synth["E"], synth["mature"]
    assert (mature / E["earned_exposure"].loc[2021:2023]).round(2).tolist() == [454.12, 400.60, 415.67]
    pp25 = trended_ratios(mature, E["earned_exposure"].loc[2021:2023], 0.04, 2025)
    assert pp25.round(2).tolist() == [531.26, 450.62, 449.58]
    assert round(synth["pp"][2024], 2) == 458.80
    assert round(synth["pp"][2025], 2) == 477.15


def test_expected_ultimates_and_errors(synth):
    eu, cl = synth["eu"], synth["cl"]
    assert eu.round(0).tolist() == [2744727, 2964160]
    true = cl.assign(ay=pd.to_datetime(cl["accident_date"]).dt.year).groupby("ay")["ultimate_amount"].sum()
    assert (eu / true.loc[[2024, 2025]] - 1).round(3).tolist() == [-0.080, -0.109]
    pp_true = (true.loc[2023:2025] / synth["E"]["earned_exposure"].loc[2023:2025]).round(2).tolist()
    assert pp_true == [419.50, 498.76, 535.34]
    mean = cl.assign(ay=pd.to_datetime(cl["accident_date"]).dt.year).groupby("ay")["ultimate_amount"].mean()
    assert mean.loc[[2023, 2024]].round(0).tolist() == [4091, 5057]


def test_expected_reserves(synth):
    lp = latest_diagonal(synth["P"])["latest"].loc[[2024, 2025]]
    li = latest_diagonal(synth["I"])["latest"].loc[[2024, 2025]]
    r = expected_reserves(synth["eu"], lp, li)
    assert r["ibnr"].round(0).tolist() == [-163809, 125419]
    assert r["unpaid"].round(0).tolist() == [314995, 1518401]


def test_synthetic_ave(synth):
    eu = synth["eu"]
    li = latest_diagonal(synth["I"])["latest"].loc[[2024, 2025]]
    lp = latest_diagonal(synth["P"])["latest"].loc[[2024, 2025]]
    aI = actual_vs_expected(eu, li, synth["dI"]["cdf"].loc[[2024, 2025]])
    aP = actual_vs_expected(eu, lp, synth["dP"]["cdf"].loc[[2024, 2025]])
    assert aI["expected_emerged"].round(0).tolist() == [2719374, 2621902]
    assert aI["difference"].round(0).tolist() == [189162, 216839]
    assert aI["ratio"].round(3).tolist() == [1.070, 1.083]
    assert aP["expected_emerged"].round(0).tolist() == [2317830, 1393949]
    assert aP["ratio"].round(3).tolist() == [1.048, 1.037]


def test_exercise3_implied_pure_premium(synth):
    I, dI, E = synth["I"], synth["dI"], synth["E"]
    pp = I.loc[2025, 12] * dI.loc[2025, "cdf"] / E.loc[2025, "earned_exposure"]
    assert pp == pytest.approx(516.61, abs=0.005)
