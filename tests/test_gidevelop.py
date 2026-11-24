"""Unit tests for src/gidevelop.py: every figure quoted in Chapter 4 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gidevelop import (  # noqa: E402
    age_to_age,
    average_factors,
    cumulative_factors,
    develop,
    fit_exponential_tail,
    latest_diagonal,
    reserve_summary,
)
from gitriangle import synthetic_history, to_long, triangle  # noqa: E402

TRUE = [2325248.70, 2127025.45, 2372859.34, 2983764.38, 3325643.42]


@pytest.fixture
def T():
    return pd.DataFrame([[1000, 1500, 1650, 1700], [1100, 1700, 1850, np.nan],
                         [1200, 1750, np.nan, np.nan], [1300, np.nan, np.nan, np.nan]],
                        index=pd.Index([2022, 2023, 2024, 2025], name="accident_year"),
                        columns=pd.Index([12, 24, 36, 48], name="age"))


@pytest.fixture(scope="module")
def synth():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    return (triangle(cl, txs, "incurred", 2021, 2025), triangle(cl, txs, "paid", 2021, 2025), cl)


# --- Sections 4.1-4.3: example triangle ----------------------------------------

def test_age_to_age(T):
    f = age_to_age(T).round(4)
    assert f["12-24"].dropna().tolist() == [1.5, 1.5455, 1.4583]
    assert f["24-36"].dropna().tolist() == [1.1, 1.0882]
    assert f["36-48"].dropna().tolist() == [1.0303]


def test_averages(T):
    assert average_factors(T, "simple").round(4).tolist() == [1.5013, 1.0941, 1.0303]
    assert average_factors(T, "volume").round(4).tolist() == [1.5, 1.0938, 1.0303]
    assert average_factors(T, "medial").round(4).tolist() == [1.5, 1.0941, 1.0303]
    assert average_factors(T, "geometric").round(4).tolist() == [1.5008, 1.0941, 1.0303]
    with pytest.raises(ValueError):
        average_factors(T, "harmonic")


def test_volume_is_weighted_average(T):
    f = age_to_age(T)["12-24"].dropna()
    w = T[12].iloc[:3] / T[12].iloc[:3].sum()
    assert (w.to_numpy() * f.to_numpy()).sum() == pytest.approx(average_factors(T, "volume").iloc[0])


def test_cdf_and_ultimates(T):
    cdf = cumulative_factors(average_factors(T, "volume"), 1.0)
    assert cdf.round(4).tolist() == [1.6903, 1.1269, 1.0303, 1.0]
    res = develop(T, cdf)
    assert res["ultimate"].round(2).tolist() == [1700.0, 1906.06, 1972.06, 2197.44]
    assert res["ultimate"].sum() == pytest.approx(7775.57, abs=0.005)
    assert (res["ultimate"] - res["latest"]).sum() == pytest.approx(1175.57, abs=0.005)
    assert res.loc[2025, "pct_developed"] == pytest.approx(0.592, abs=0.0005)


def test_exercise1_unpaid(T):
    ult = develop(T, cumulative_factors(average_factors(T, "volume"), 1.0))["ultimate"]
    paid = pd.DataFrame({12: [np.nan] * 4}, index=T.index)
    paid[48] = [1650, np.nan, np.nan, np.nan]
    # build a paid triangle whose latest diagonal is 1650, 1600, 1200, 600
    P = pd.DataFrame(np.nan, index=T.index, columns=T.columns)
    P.loc[2022, 48], P.loc[2023, 36], P.loc[2024, 24], P.loc[2025, 12] = 1650, 1600, 1200, 600
    P = P.bfill(axis=1).where(T.notna())
    rs = reserve_summary(P, T, ult)
    assert rs["unpaid"].round(2).tolist() == [50.0, 306.06, 772.06, 1597.44]
    assert rs["case_reserve"].tolist() == [50, 250, 550, 700]
    assert rs["unpaid"].sum() == pytest.approx(rs["case_reserve"].sum() + rs["ibnr"].sum())
    assert rs["unpaid"].sum() == pytest.approx(2725.57, abs=0.005)


def test_exercise2_tail(T):
    cdf = cumulative_factors(average_factors(T, "volume"), 1.02)
    assert cdf.round(4).tolist() == [1.7241, 1.1494, 1.0509, 1.02]
    res = develop(T, cdf)
    assert res["ultimate"].round(2).tolist() == [1734.0, 1944.18, 2011.51, 2241.39]
    ibnr = (res["ultimate"] - res["latest"]).sum()
    assert ibnr == pytest.approx(1331.08, abs=0.005)
    assert ibnr / 1175.57 - 1 == pytest.approx(0.132, abs=0.0005)


# --- Section 4.4: synthetic portfolio ------------------------------------------

def test_synthetic_factors(synth):
    I, P, _ = synth
    assert average_factors(I, "volume").round(4).tolist() == [1.1201, 1.0091, 0.9983, 1.0019]
    assert average_factors(I, "simple").round(4).tolist() == [1.1206, 1.0094, 0.9985, 1.0019]
    assert average_factors(I, "volume", 3).round(4).tolist() == [1.131, 1.0091, 0.9983, 1.0019]
    assert average_factors(P, "volume").round(4).tolist() == [1.7957, 1.1484, 1.024, 1.007]
    assert age_to_age(I)["12-24"].dropna().round(4).tolist() == [1.0859, 1.1598, 1.0997, 1.1369]


def test_synthetic_ultimates(synth):
    I, P, cl = synth
    dI = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    dP = develop(P, cumulative_factors(average_factors(P, "volume"), 1.0))
    assert dI["ultimate"].round(0).tolist() == [2325132, 2131796, 2351184, 2935653, 3209305]
    assert dP["ultimate"].round(0).tolist() == [2324983, 2105305, 2384097, 2877240, 3074330]
    assert dI.loc[2025, "cdf"] == pytest.approx(1.1305, abs=0.00005)
    assert dP.loc[2025, "cdf"] == pytest.approx(2.1264, abs=0.00005)
    true = cl.assign(ay=pd.to_datetime(cl["accident_date"]).dt.year).groupby("ay")["ultimate_amount"].sum()
    assert true.loc[2021:2025].round(2).tolist() == TRUE
    assert (dI["ultimate"].to_numpy() / TRUE - 1).round(3).tolist() == [-0.0, 0.002, -0.009, -0.016, -0.035]
    assert (I[12].to_numpy() / TRUE).round(3).tolist() == [0.895, 0.855, 0.91, 0.857, 0.854]


def test_synthetic_reserve(synth):
    I, P, _ = synth
    ult = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))["ultimate"]
    rs = reserve_summary(P, I, ult)
    assert rs[["case_reserve", "ibnr", "unpaid"]].sum().round(0).tolist() == [1947963, 402044, 2350007]
    true_unpaid = sum(TRUE) - latest_diagonal(P)["latest"].sum()
    assert true_unpaid == pytest.approx(2531478, abs=1)
    assert rs["unpaid"].sum() / true_unpaid - 1 == pytest.approx(-0.072, abs=0.0005)


def test_exercise3_latest3(synth):
    I, _, _ = synth
    cdf3 = cumulative_factors(average_factors(I, "volume", n=3), 1.0)
    assert cdf3.loc[12] == pytest.approx(1.1415, abs=0.00005)
    assert develop(I, cdf3).loc[2025, "ultimate"] == pytest.approx(3240470, abs=1)


def test_matches_chainladder(synth):
    chl = pytest.importorskip("chainladder")
    I, _, _ = synth
    tri = chl.Triangle(to_long(I, "incurred"), origin="accident_year", development="valuation",
                       columns=["incurred"], cumulative=True)
    model = chl.Chainladder().fit(chl.Development(average="volume").fit_transform(tri))
    ours = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))["ultimate"]
    assert np.allclose(model.ultimate_.to_frame().values.ravel(), ours.to_numpy())


# --- Pass 2 additions: paid route, zero cells, tail fit, new Exercise 1 ----------

def test_paid_route_example(T):
    P = pd.DataFrame([[500, 1100, 1450, 1650], [550, 1150, 1600, np.nan],
                      [560, 1200, np.nan, np.nan], [600, np.nan, np.nan, np.nan]],
                     index=T.index, columns=T.columns)
    assert average_factors(P, "volume").round(4).tolist() == [2.1429, 1.3556, 1.1379]
    cdf = cumulative_factors(average_factors(P, "volume"), tail=1700 / 1650)
    assert cdf.round(4).tolist() == [3.4056, 1.5893, 1.1724, 1.0303]
    d = develop(P, cdf)
    assert d["ultimate"].round(2).tolist() == [1700.0, 1875.86, 1907.13, 2043.35]
    assert d["ultimate"].sum() == pytest.approx(7526.34, abs=0.005)
    assert d["ultimate"].sum() - 5050 == pytest.approx(2476.34, abs=0.005)
    assert d["ultimate"].sum() - 6600 == pytest.approx(926.34, abs=0.005)


def test_zero_cell():
    X = pd.DataFrame([[800, 1040], [0, 1197], [950, np.nan]],
                     index=pd.Index([2021, 2022, 2023]), columns=pd.Index([12, 24]))
    assert np.isinf(average_factors(X, "simple").iloc[0])
    assert np.isfinite(average_factors(X, "volume").iloc[0])


def test_fit_exponential_tail():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    P = triangle(cl, txs, "paid", 2021, 2025)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    fit = fit_exponential_tail(average_factors(P, "volume"))
    assert (round(fit["a"], 3), round(fit["b"], 3), round(fit["tail"], 4)) == (3.657, 1.601, 1.0015)
    assert np.round(fit["fitted"], 4).tolist() == [1.7375, 1.1487, 1.0300, 1.0060]
    assert I.loc[2021, 60] / P.loc[2021, 60] == pytest.approx(1.0001, abs=0.00005)
    ult = develop(P, cumulative_factors(average_factors(P, "volume"), 1.0))["ultimate"][2025]
    assert ult * (fit["tail"] - 1) == pytest.approx(4700, abs=50)
    with pytest.raises(ValueError):
        fit_exponential_tail(pd.Series([1.2, 0.99]))


def test_new_exercise1():
    X = pd.DataFrame([[800, 1040, 1144, 1167, 1170], [900, 1197, 1305, 1331, np.nan],
                      [950, 1235, 1359, np.nan, np.nan], [1000, 1310, np.nan, np.nan, np.nan],
                      [1100, np.nan, np.nan, np.nan, np.nan]],
                     index=pd.Index(range(2021, 2026), name="accident_year"),
                     columns=pd.Index([12, 24, 36, 48, 60], name="age"))
    assert average_factors(X, "simple").round(4).tolist() == [1.31, 1.0969, 1.02, 1.0026]
    assert average_factors(X, "volume").round(4).tolist() == [1.3101, 1.0968, 1.02, 1.0026]
    cdf = cumulative_factors(average_factors(X, "volume"), 1.0)
    assert cdf.round(4).tolist() == [1.4694, 1.1216, 1.0226, 1.0026, 1.0]
    d = develop(X, cdf)
    assert d["ultimate"].round(2).tolist() == [1170.0, 1334.42, 1389.75, 1469.29, 1616.39]
    assert (d["pct_developed"] * 100).round(1).tolist() == [100.0, 99.7, 97.8, 89.2, 68.1]
    assert d["ultimate"].sum() == pytest.approx(6979.85, abs=0.005)
    s = develop(X, cumulative_factors(average_factors(X, "simple"), 1.0))["ultimate"].sum()
    assert s == pytest.approx(6980.00, abs=0.005)


def test_incurred_selection_paragraph():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    f = average_factors(I, "volume")
    assert round(f.iloc[2] * f.iloc[3], 4) == 1.0002
    d = develop(I, cumulative_factors(f, 1.0))
    change = d.loc[2022, "latest"] * (d.loc[2022, "cdf"] - 1) + d.loc[2023, "latest"] * (d.loc[2023, "cdf"] - 1)
    assert 0 < change < 5000
