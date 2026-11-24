"""Unit tests for src/gicapecod.py: every figure quoted in Chapter 7 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gicapecod import average_severity, cape_cod, cape_cod_ratio, used_up_base  # noqa: E402
from gidevelop import age_to_age, average_factors, cumulative_factors, develop  # noqa: E402
from gitriangle import exposure_by_accident_year, synthetic_history, triangle  # noqa: E402

TRUE = pd.Series([2325248.70, 2127025.45, 2372859.34, 2983764.38, 3325643.42],
                 index=range(2021, 2026))


@pytest.fixture
def ex():
    T = pd.DataFrame([[1000, 1500, 1650, 1700], [1100, 1700, 1850, np.nan],
                      [1200, 1750, np.nan, np.nan], [1300, np.nan, np.nan, np.nan]],
                     index=pd.Index([2022, 2023, 2024, 2025], name="accident_year"),
                     columns=pd.Index([12, 24, 36, 48], name="age"))
    EP = pd.Series([2500, 2700, 2900, 3100.0], index=T.index)
    return develop(T, cumulative_factors(average_factors(T, "volume"), 1.0)), EP


@pytest.fixture(scope="module")
def synth():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    P = triangle(cl, txs, "paid", 2021, 2025)
    R = triangle(cl, txs, "reported_count", 2021, 2025)
    E = exposure_by_accident_year(pol, range(2021, 2026))
    dI = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    dP = develop(P, cumulative_factors(average_factors(P, "volume"), 1.0))
    dR = develop(R, cumulative_factors(average_factors(R, "volume"), 1.0))
    level = pd.Series({y: 1.04 ** -(2025 - y) for y in range(2021, 2026)})
    return dict(I=I, R=R, E=E, dI=dI, dP=dP, dR=dR, level=level, cl=cl)


# --- Section 7.1: example triangle ------------------------------------------------

def test_used_up_and_ratio(ex):
    dev, EP = ex
    uu = used_up_base(EP, dev["cdf"])
    assert uu.round(2).tolist() == [2500.0, 2620.59, 2573.45, 1833.95]
    assert uu.sum() == pytest.approx(9527.98, abs=0.005)
    assert cape_cod_ratio(dev["latest"], EP, dev["cdf"]) == pytest.approx(0.6927, abs=0.00005)


def test_cape_cod_example(ex):
    dev, EP = ex
    cc = cape_cod(dev["latest"], EP, dev["cdf"])
    assert cc["expected_ultimate"].round(2).tolist() == [1731.74, 1870.28, 2008.82, 2147.36]
    assert cc["bf_ibnr"].round(2).tolist() == [0.0, 55.01, 226.20, 876.99]
    assert cc["bf_ultimate"].sum() == pytest.approx(7758.20, abs=0.005)
    assert cc["bf_ibnr"].sum() == pytest.approx(1158.20, abs=0.005)


def test_exercise1_on_level(ex):
    dev, EP = ex
    lvl = pd.Series([1.05 ** 2, 1.05 ** 2, 1.05, 1.0], index=EP.index)
    assert (EP * lvl).round(2).tolist() == [2756.25, 2976.75, 3045.0, 3100.0]
    assert (EP * lvl / dev["cdf"]).sum() == pytest.approx(10181.52, abs=0.005)
    cc2 = cape_cod(dev["latest"], EP, dev["cdf"], lvl)
    assert cc2["ratio"].iloc[0] == pytest.approx(0.6482, abs=0.00005)
    assert cc2["expected_ultimate"].round(2).tolist() == [1786.69, 1929.63, 1973.87, 2009.52]
    assert cc2["bf_ultimate"].round(2).tolist() == [1700.0, 1906.75, 1972.27, 2120.70]
    assert cc2["bf_ultimate"].sum() == pytest.approx(7699.72, abs=0.005)
    base = cape_cod(dev["latest"], EP, dev["cdf"])["bf_ultimate"][2025]
    assert base - cc2["bf_ultimate"][2025] == pytest.approx(56.29, abs=0.005)


# --- Section 7.2: synthetic Cape Cod ----------------------------------------------

def test_synthetic_used_up(synth):
    E, level, dI = synth["E"], synth["level"], synth["dI"]
    assert level.round(4).tolist() == [0.8548, 0.8890, 0.9246, 0.9615, 1.0]
    adj = E["earned_exposure"] * level
    assert adj.round(2).tolist() == [4376.66, 4730.84, 5229.69, 5752.30, 6212.18]
    assert (adj / dI["cdf"]).round(2).tolist() == [4376.66, 4722.10, 5228.74, 5699.17, 5494.89]
    assert (adj / dI["cdf"]).sum() == pytest.approx(25521.55, abs=0.005)


def test_synthetic_cape_cod(synth):
    E, level, dI, dP = synth["E"], synth["level"], synth["dI"], synth["dP"]
    cI = cape_cod(dI["latest"], E["earned_exposure"], dI["cdf"], level)
    cP = cape_cod(dP["latest"], E["earned_exposure"], dP["cdf"], level)
    assert cI["ratio"].iloc[0] == pytest.approx(491.78, abs=0.005)
    assert cP["ratio"].iloc[0] == pytest.approx(483.61, abs=0.005)
    assert (cI["expected_ultimate"] / E["earned_exposure"]).round(2).tolist() == [420.38, 437.19, 454.68, 472.87, 491.78]
    assert cI["bf_ultimate"].loc[[2024, 2025]].round(0).tolist() == [2934667, 3191492]
    assert cP["bf_ultimate"].loc[[2024, 2025]].round(0).tolist() == [2862404, 3037207]
    assert round(cI["bf_ultimate"].sum()) == 12934671
    assert round(cP["bf_ultimate"].sum()) == 12719657
    assert (cI["bf_ultimate"] / TRUE - 1).loc[[2024, 2025]].round(3).tolist() == [-0.016, -0.040]
    assert round(cI["bf_ultimate"].sum() - dP["latest"].sum()) == 2331608


# --- Section 7.3: frequency-severity ----------------------------------------------

def test_counts_and_frequency(synth):
    dR, E, cl = synth["dR"], synth["E"], synth["cl"]
    assert average_factors(synth["R"], "volume").round(4).tolist() == [1.0868, 1.0, 1.0, 1.0]
    assert dR["ultimate"].round(2).tolist() == [524.0, 548.0, 580.0, 590.0, 644.45]
    n_true = cl.assign(ay=pd.to_datetime(cl["accident_date"]).dt.year).groupby("ay").size().loc[2021:2025]
    assert n_true.tolist() == [524, 548, 580, 590, 654]
    assert (dR["ultimate"] / E["earned_exposure"]).round(4).tolist() == [0.1023, 0.1030, 0.1025, 0.0986, 0.1037]


def test_severity_development(synth):
    S = average_severity(synth["I"], synth["R"])
    assert S[12].round(2).tolist() == [4292.49, 3551.23, 4114.74, 4728.96, 4787.08]
    assert S.loc[2021].round(2).tolist() == [4292.49, 4314.46, 4451.99, 4429.08, 4437.27]
    fs = average_factors(S, "simple")
    assert fs.round(4).tolist() == [1.0317, 1.0094, 0.9985, 1.0019]
    cdf = cumulative_factors(fs, 1.0)
    assert cdf.loc[12] == pytest.approx(1.0417, abs=0.00005)
    dS = develop(S, cdf)
    assert dS["ultimate"].round(2).tolist() == [4437.27, 3890.14, 4054.46, 4977.92, 4986.91]
    u = synth["dR"]["ultimate"] * dS["ultimate"]
    assert u.round(0).tolist() == [2325132, 2131796, 2351589, 2936973, 3213829]
    assert (u / TRUE - 1).round(3).tolist() == [-0.0, 0.002, -0.009, -0.016, -0.034]
    assert dS["ultimate"][2024] / dS["ultimate"][2023] - 1 == pytest.approx(0.228, abs=0.0005)


def test_trended_severity(synth):
    sev = (synth["dI"]["ultimate"] / synth["dR"]["ultimate"]).loc[2021:2023]
    assert sev.round(2).tolist() == [4437.27, 3890.14, 4053.77]
    s25 = np.mean([sev[a] * 1.04 ** (2025 - a) for a in sev.index])
    assert s25 == pytest.approx(4650.47, abs=0.005)
    u25 = s25 * synth["dR"]["ultimate"][2025]
    assert u25 == pytest.approx(2997009, abs=1)
    s24 = np.mean([sev[a] * 1.04 ** (2024 - a) for a in sev.index])
    assert round(s24 * synth["dR"]["ultimate"][2024]) == 2638248
