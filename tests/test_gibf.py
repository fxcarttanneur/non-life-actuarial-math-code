"""Unit tests for src/gibf.py: every figure quoted in Chapter 6 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gibf import benktander, bornhuetter_ferguson, credibility_weight  # noqa: E402
from gidevelop import average_factors, cumulative_factors, develop  # noqa: E402
from giexpected import expected_ultimate, trended_ratios  # noqa: E402
from gitriangle import exposure_by_accident_year, synthetic_history, triangle  # noqa: E402


@pytest.fixture
def ex():
    T = pd.DataFrame([[1000, 1500, 1650, 1700], [1100, 1700, 1850, np.nan],
                      [1200, 1750, np.nan, np.nan], [1300, np.nan, np.nan, np.nan]],
                     index=pd.Index([2022, 2023, 2024, 2025], name="accident_year"),
                     columns=pd.Index([12, 24, 36, 48], name="age"))
    EP = pd.Series([2500, 2700, 2900, 3100.0], index=T.index)
    dev = develop(T, cumulative_factors(average_factors(T, "volume"), 1.0))
    return dev, EP


@pytest.fixture(scope="module")
def synth():
    pol, cl, txs = synthetic_history(2020, 2025, seed=100)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    P = triangle(cl, txs, "paid", 2021, 2025)
    E = exposure_by_accident_year(pol, range(2021, 2026))
    dI = develop(I, cumulative_factors(average_factors(I, "volume"), 1.0))
    dP = develop(P, cumulative_factors(average_factors(P, "volume"), 1.0))
    mature = dI["ultimate"].loc[2021:2023]
    pp = pd.Series({y: trended_ratios(mature, E["earned_exposure"].loc[2021:2023], 0.04, y).mean()
                    for y in range(2021, 2026)})
    return dict(dI=dI, dP=dP, pp=pp, eu=E["earned_exposure"] * pp, P=P)


# --- Section 6.2: example triangle ------------------------------------------------

def test_bf_example(ex):
    dev, EP = ex
    bf = bornhuetter_ferguson(dev["latest"], dev["cdf"], expected_ultimate(EP, 0.70))
    assert bf["pct_unreported"].round(4).tolist() == [0.0, 0.0294, 0.1126, 0.4084]
    assert bf["bf_ibnr"].round(2).tolist() == [0.0, 55.59, 228.59, 886.24]
    assert bf["bf_ultimate"].round(2).tolist() == [1700.0, 1905.59, 1978.59, 2186.24]
    assert bf["bf_ultimate"].sum() == pytest.approx(7770.41, abs=0.005)


def test_bf_is_credibility_weighting(ex):
    dev, EP = ex
    eu = expected_ultimate(EP, 0.70)
    bf = bornhuetter_ferguson(dev["latest"], dev["cdf"], eu)["bf_ultimate"]
    z = credibility_weight(dev["cdf"], 1)
    assert np.allclose(bf, z * dev["ultimate"] + (1 - z) * eu)
    gb = benktander(dev["latest"], dev["cdf"], eu, 2)
    z2 = credibility_weight(dev["cdf"], 2)
    assert np.allclose(gb, z2 * dev["ultimate"] + (1 - z2) * eu)


def test_benktander_iterations(ex):
    dev, EP = ex
    eu = expected_ultimate(EP, 0.70)
    expected = {0: [2030.0, 2170.0], 1: [1978.59, 2186.24], 2: [1972.8, 2192.87],
                3: [1972.15, 2195.57], 10: [1972.06, 2197.44]}
    for k, v in expected.items():
        assert benktander(dev["latest"], dev["cdf"], eu, k).loc[[2024, 2025]].round(2).tolist() == v
    assert benktander(dev["latest"], dev["cdf"], eu, 2).sum() == pytest.approx(7771.71, abs=0.005)
    assert np.allclose(benktander(dev["latest"], dev["cdf"], eu, 60), dev["ultimate"])


def test_credibility_weights(ex):
    dev, _ = ex
    assert credibility_weight(dev["cdf"], 1).round(4).tolist() == [1.0, 0.9706, 0.8874, 0.5916]
    assert credibility_weight(dev["cdf"], 2).round(4).tolist() == [1.0, 0.9991, 0.9873, 0.8332]


def test_elr_sensitivity(ex):
    dev, EP = ex
    bf_tot, gb_tot, bf25, gb25 = {}, {}, {}, {}
    for elr in (0.6, 0.7, 0.8):
        e = expected_ultimate(EP, elr)
        b = bornhuetter_ferguson(dev["latest"], dev["cdf"], e)["bf_ultimate"]
        g = benktander(dev["latest"], dev["cdf"], e, 2)
        bf_tot[elr], gb_tot[elr], bf25[elr], gb25[elr] = b.sum(), g.sum(), b[2025], g[2025]
    assert [round(bf_tot[e], 2) for e in (0.6, 0.7, 0.8)] == [7603.21, 7770.41, 7937.61]
    assert [round(gb_tot[e], 2) for e in (0.6, 0.7, 0.8)] == [7716.09, 7771.71, 7827.33]
    assert bf_tot[0.8] - bf_tot[0.6] == pytest.approx(334.40, abs=0.005)
    assert gb_tot[0.8] - gb_tot[0.6] == pytest.approx(111.23, abs=0.005)
    assert (bf25[0.8] - bf25[0.6]) / 620 == pytest.approx(0.408, abs=0.0005)
    assert (gb25[0.8] - gb25[0.6]) / 620 == pytest.approx(0.167, abs=0.0005)


# --- Sections 6.3-6.4: synthetic portfolio --------------------------------------

def test_priors(synth):
    assert synth["pp"].round(2).tolist() == [407.87, 424.19, 441.15, 458.80, 477.15]


def test_synthetic_results(synth):
    dI, dP, eu = synth["dI"], synth["dP"], synth["eu"]
    bI = bornhuetter_ferguson(dI["latest"], dI["cdf"], eu)
    bP = bornhuetter_ferguson(dP["latest"], dP["cdf"], eu)
    assert bI["pct_unreported"].round(4).tolist() == [0.0, 0.0018, 0.0002, 0.0092, 0.1155]
    assert bP["pct_unreported"].round(4).tolist() == [0.0, 0.0070, 0.0303, 0.1555, 0.5297]
    assert bI["bf_ultimate"].loc[[2024, 2025]].round(0).tolist() == [2933890, 3180999]
    assert bP["bf_ultimate"].loc[[2024, 2025]].round(0).tolist() == [2856630, 3015969]
    assert benktander(dI["latest"], dI["cdf"], eu, 2).loc[[2024, 2025]].round(0).tolist() == [2935637, 3206036]
    assert benktander(dP["latest"], dP["cdf"], eu, 2).round(0).tolist() == [2324983, 2105313, 2384199, 2874034, 3043414]
    paid = dP["latest"].sum()
    totals = {"exp": eu.sum(), "bfI": bI["bf_ultimate"].sum(), "bfP": bP["bf_ultimate"].sum(),
              "gbI": benktander(dI["latest"], dI["cdf"], eu, 2).sum()}
    assert {k: round(v) for k, v in totals.items()} == {"exp": 12549914, "bfI": 12923259,
                                                       "bfP": 12691411, "gbI": 12949786}
    assert round(totals["bfI"] - paid) == 2320196
    assert round(totals["bfP"] - paid) == 2088348


def test_shock_leverage(synth):
    dI, dP, eu = synth["dI"], synth["dP"], synth["eu"]
    s = dI["latest"].copy()
    s[2025] *= 1.10
    d = s[2025] - dI["latest"][2025]
    assert d == pytest.approx(283874.11, abs=0.005)
    assert s[2025] * dI["cdf"][2025] - dI["ultimate"][2025] == pytest.approx(320930.47, abs=0.01)
    bf0 = bornhuetter_ferguson(dI["latest"], dI["cdf"], eu)["bf_ultimate"][2025]
    assert bornhuetter_ferguson(s, dI["cdf"], eu)["bf_ultimate"][2025] - bf0 == pytest.approx(d)
    gb_change = benktander(s, dI["cdf"], eu, 2)[2025] - benktander(dI["latest"], dI["cdf"], eu, 2)[2025]
    assert gb_change == pytest.approx(316651.75, abs=0.01)
    sp = dP["latest"].copy()
    sp[2025] += d
    assert sp[2025] * dP["cdf"][2025] - dP["ultimate"][2025] == pytest.approx(603643.33, abs=0.01)
    gbp = benktander(sp, dP["cdf"], eu, 2)[2025] - benktander(dP["latest"], dP["cdf"], eu, 2)[2025]
    assert gbp == pytest.approx(434251.32, abs=0.01)
