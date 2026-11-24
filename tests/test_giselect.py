"""Unit tests for src/giselect.py: every figure quoted in Chapter 8 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gidevelop import age_to_age, average_factors, cumulative_factors, develop, latest_diagonal  # noqa: E402
from giselect import capped_triangle, expected_emergence, large_claims, selection_summary  # noqa: E402
from gitriangle import exposure_by_accident_year, synthetic_history, triangle  # noqa: E402

TRUE_TOTAL = 13134541.29


@pytest.fixture(scope="module")
def d():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    P = triangle(cl, txs, "paid", 2021, 2025)
    R = triangle(cl, txs, "reported_count", 2021, 2025)
    Ic = capped_triangle(cl, txs, 25_000, 2021, 2025)
    Ix = capped_triangle(cl, txs, 25_000, 2021, 2025, part="excess")
    dI = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    dc = develop(Ic, cumulative_factors(average_factors(Ic, "volume"), 1.0))
    load = (dI["ultimate"] - dc["ultimate"]).sum() / dc["ultimate"].sum()
    return dict(pol=pol, cl=cl, txs=txs, I=I, P=P, R=R, Ic=Ic, Ix=Ix, dI=dI, dc=dc, load=load)


def test_large_claims(d):
    big = large_claims(d["cl"], d["txs"], "2025-12-31", 25_000)
    big = big[big["accident_year"] >= 2021]
    assert len(big) == 62
    assert big.groupby("accident_year").size().tolist() == [11, 7, 5, 20, 19]
    assert big["incurred"].head(6).round(2).tolist() == [149908.18, 134712.58, 100973.25,
                                                         99648.59, 70330.62, 60792.12]
    assert big.index[2] == "2024-C000260" and big["paid"].iloc[2] == 0
    assert (big["case_reserve"] > 0).groupby(big["accident_year"]).sum().loc[[2024, 2025]].tolist() == [4, 12]


def test_capped_plus_excess_is_incurred(d):
    assert np.allclose((d["Ic"] + d["Ix"]).fillna(0), d["I"].fillna(0))
    assert d["Ix"][12].round(0).tolist() == [230101, 29427, 242211, 196780, 235209]
    assert d["Ix"].loc[2021].round(0).tolist() == [230101, 236196, 281008, 273440, 273440]


def test_capped_factors(d):
    assert average_factors(d["Ic"], "volume").round(4).tolist() == [1.1166, 1.0071, 1.0, 1.0021]
    f_c = age_to_age(d["Ic"])["12-24"].dropna()
    f_u = age_to_age(d["I"])["12-24"].dropna()
    assert (f_c.min().round(4), f_c.max().round(4)) == (1.0933, 1.1601)
    assert round(f_c.std(), 3) == 0.030 and round(f_u.std(), 3) == 0.034


def test_capped_ultimates_and_load(d):
    assert d["dc"]["ultimate"].round(0).tolist() == [2051691, 2081627, 2164930, 2627046, 2933936]
    assert d["load"] == pytest.approx(0.0922, abs=0.00005)
    xs = (d["dI"]["ultimate"] - d["dc"]["ultimate"]).sum()
    assert round(xs) == 1093840 and round(d["dc"]["ultimate"].sum()) == 11859230
    c = d["cl"].assign(ay=pd.to_datetime(d["cl"]["accident_date"]).dt.year)
    c = c[c["ay"].between(2021, 2025)]
    capped_true = c["ultimate_amount"].clip(upper=25_000).groupby(c["ay"]).sum()
    assert capped_true.round(0).tolist() == [2051808, 2076436, 2182424, 2679976, 3035214]
    true_load = (c["ultimate_amount"].sum() - capped_true.sum()) / capped_true.sum()
    assert true_load == pytest.approx(0.0922, abs=0.00005)
    u25 = d["dc"].loc[2025, "ultimate"] * (1 + d["load"])
    assert round(u25) == 3204548


def test_true_large_counts(d):
    c = d["cl"].assign(ay=pd.to_datetime(d["cl"]["accident_date"]).dt.year)
    c = c[c["ay"].between(2021, 2025)]
    for cap, counts in {20_000: [17, 12, 14, 26, 38], 25_000: [11, 7, 5, 20, 28],
                        50_000: [3, 0, 2, 3, 1]}.items():
        assert (c["ultimate_amount"] > cap).groupby(c["ay"]).sum().tolist() == counts


def test_roll_forward(d):
    I24 = triangle(d["cl"], d["txs"], "incurred", 2021, 2024)
    d24 = develop(I24, cumulative_factors(average_factors(I24, "volume"), 1.0))
    assert d24["cdf"].round(4).tolist() == [1.0, 0.9949, 1.0145, 1.1292]
    assert round(d24["ultimate"].sum()) == 9732218
    ee = expected_emergence(d24["ultimate"], d24["cdf"], d["dI"]["cdf"].loc[2021:2024])
    actual = d["dI"]["latest"].loc[2021:2024] - d24["latest"]
    assert ee.round(0).tolist() == [0, -14827, 34108, 303852]
    assert actual.round(0).tolist() == [4294, 4581, -24820, 350168]
    assert (actual - ee).round(0).tolist() == [4294, 19408, -58928, 46317]
    assert (d["dI"]["ultimate"].loc[2021:2024] - d24["ultimate"]).round(0).tolist() == [4294, 19444, -58939, 46748]


def test_selection(d):
    E = exposure_by_accident_year(d["pol"], range(2021, 2026))
    dR = develop(d["R"], cumulative_factors(average_factors(d["R"], "volume"), 1.0))
    sel = d["dI"]["ultimate"].copy()
    sel[2025] = d["dc"].loc[2025, "ultimate"] * (1 + d["load"])
    s = selection_summary(sel, latest_diagonal(d["P"])["latest"], d["dI"]["latest"],
                          E["earned_premium"], E["earned_exposure"], dR["ultimate"])
    assert s["loss_ratio"].round(3).tolist() == [0.733, 0.643, 0.670, 0.791, 0.828]
    assert s["severity"].round(0).tolist() == [4437, 3890, 4054, 4976, 4973]
    assert s["ibnr_to_case"].round(2).tolist() == [0.0, 0.11, 0.01, 0.06, 0.26]
    assert s[["case_reserve", "ibnr", "unpaid"]].sum().round(0).tolist() == [1947963, 397288, 2345251]
    assert round(s["selected"].sum()) == 12948314
    true_unpaid = TRUE_TOTAL - latest_diagonal(d["P"])["latest"].sum()
    assert s["unpaid"].sum() / true_unpaid - 1 == pytest.approx(-0.074, abs=0.0005)


def test_exercise1_december_claim(d):
    a = 150_000 * d["dI"].loc[2025, "cdf"]
    b = 25_000 * d["dc"].loc[2025, "cdf"] * (1 + d["load"])
    assert round(a) == 169581 and round(b) == 30771
