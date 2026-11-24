"""Unit tests for Part V modules (Chapters 20-24): giadjust, giulae, givar, gidiscount, gipremliab."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from giadjust import (average_case, bs_case_adjust, bs_paid_adjust, changing_history,  # noqa: E402
                      disposal_ratios)
from gidata import exposure_table  # noqa: E402
from gidevelop import average_factors, cumulative_factors, develop, latest_diagonal  # noqa: E402
from gidiscount import (confidence_level, cost_of_capital_margin, discount_factors,  # noqa: E402
                        future_cash_flows, present_value)
from gipremliab import ae_review, premium_deficiency, premium_liability  # noqa: E402
from gitriangle import synthetic_history, to_long, triangle  # noqa: E402
from giulae import (calendar_activity, classical_unpaid, count_based_unpaid,  # noqa: E402
                    kittel_ratio, paid_to_paid, synthetic_ulae, unit_costs)
from givar import mack, odp_bootstrap, summarize  # noqa: E402

TRUE = pd.Series([2325249, 2127025, 2372859, 2983764, 3325643], index=range(2021, 2026))
IDX = pd.Index([2023, 2024, 2025], name="accident_year")
COLS = pd.Index([12, 24, 36], name="age")
EX4 = pd.DataFrame([[1000, 1500, 1650, 1700], [1100, 1700, 1850, np.nan], [1200, 1750, np.nan, np.nan],
                    [1300, np.nan, np.nan, np.nan]], index=[2022, 2023, 2024, 2025], columns=[12, 24, 36, 48])


def cdf(t):
    return cumulative_factors(average_factors(t, "volume"), 1.0)


@pytest.fixture(scope="module")
def hist():
    return synthetic_history()


@pytest.fixture(scope="module")
def paid(hist):
    _, C, T = hist
    return triangle(C, T, "paid", 2021, 2025)


@pytest.fixture(scope="module")
def boot(paid):
    return odp_bootstrap(paid, 10_000, seed=22)


# ------------------------------------------------------------ Chapter 20

def test_bs_hand_case():
    P = pd.DataFrame([[1000, 1800, 2000], [1100, 1950, np.nan], [1200, np.nan, np.nan]], IDX, COLS)
    O = pd.DataFrame([[40, 15, 3], [42, 16, np.nan], [45, np.nan, np.nan]], IDX, COLS)
    A = pd.DataFrame([[20, 18, 15], [25, 24, np.nan], [32, np.nan, np.nan]], IDX, COLS)
    inc = P + O * A
    assert inc.stack().dropna().tolist() == [1800, 2070, 2045, 2150, 2334, 2640]
    adj = bs_case_adjust(inc, P, O, 0.05)
    assert adj.stack().dropna().round(2).tolist() == [2161.0, 2142.86, 2045.0, 2380.0, 2334.0, 2640.0]
    u0 = develop(inc, cdf(inc))["ultimate"]
    u1 = develop(inc, cdf(adj))["ultimate"]
    assert average_factors(inc).round(4).tolist() == [1.1149, 0.9879]
    assert average_factors(adj).round(4).tolist() == [0.9859, 0.9543]
    assert u0.round(1).tolist() == [2045.0, 2305.8, 2907.9] and round(u0.sum(), 1) == 7258.7
    assert u1.round(1).tolist() == [2045.0, 2227.4, 2483.9] and round(u1.sum(), 1) == 6756.3
    adj0 = bs_case_adjust(inc, P, O, 0.0)
    assert round(develop(inc, cdf(adj0))["ultimate"][2025], 1) == 2377.8


def test_bs_hand_paid():
    P = pd.DataFrame([[600, 1500, 1900], [800, 1700, np.nan], [900, np.nan, np.nan]], IDX, COLS)
    Cc = pd.DataFrame([[30, 80, 98], [45, 90, np.nan], [55, np.nan, np.nan]], IDX, COLS)
    U = pd.Series(100.0, IDX)
    adj = bs_paid_adjust(P, Cc, U)
    assert adj.stack().dropna().round(2).tolist() == [1050.0, 1722.22, 1900.0, 1000.0, 1700.0, 900.0]
    assert average_factors(P).round(4).tolist() == [2.2857, 1.2667]
    assert average_factors(adj).round(4).tolist() == [1.6694, 1.1032]
    assert develop(P, cdf(P))["ultimate"].round(1).tolist() == [1900.0, 2153.3, 2605.7]
    assert develop(P, cdf(adj))["ultimate"].round(1).tolist() == [1900.0, 1875.5, 1657.5]


def test_bs_synthetic():
    _, C, T = changing_history()
    tri = {m: triangle(C, T, m, 2021, 2025) for m in ("paid", "incurred", "reported_count", "closed_count", "open_count")}
    n_ult = develop(tri["reported_count"], cdf(tri["reported_count"]))["ultimate"]
    avg = average_case(tri["incurred"], tri["paid"], tri["open_count"])
    assert avg[12].round(0).tolist() == [3932, 3153, 3930, 5184, 5073]
    assert avg[24].dropna().round(0).tolist() == [4829, 3107, 5763, 8424]
    assert tri["open_count"].loc[2024, 24] == 36
    d = disposal_ratios(tri["closed_count"], n_ult)
    assert d[12].round(3).tolist() == [0.416, 0.453, 0.552, 0.527, 0.560]
    assert d[24].dropna().round(3).tolist() == [0.870, 0.880, 0.952, 0.939]
    res = pd.DataFrame({
        "inc": develop(tri["incurred"], cdf(tri["incurred"]))["ultimate"],
        "paid": develop(tri["paid"], cdf(tri["paid"]))["ultimate"],
        "bs_inc": develop(tri["incurred"], cdf(bs_case_adjust(tri["incurred"], tri["paid"], tri["open_count"], 0.04)))["ultimate"],
        "bs_paid": develop(tri["paid"], cdf(bs_paid_adjust(tri["paid"], tri["closed_count"], n_ult)))["ultimate"]})
    assert res.inc.round(0).tolist() == [2325132, 2122265, 2377017, 3058215, 3400089]
    assert res.paid.round(0).tolist() == [2324983, 2098709, 2412486, 3072646, 3719538]
    assert res.bs_inc.round(0).tolist() == [2325132, 2116099, 2342082, 2808238, 3107744]
    assert res.bs_paid.round(0).tolist() == [2324983, 2109069, 2363226, 2907906, 3247057]
    assert (res.sum() - TRUE.sum()).round(0).tolist() == [148176, 493822, -435245, -182299]


# ------------------------------------------------------------ Chapter 21

def test_ulae(hist):
    _, C, T = hist
    tri = {m: triangle(C, T, m, 2020, 2025) for m in ("paid", "incurred", "reported_count", "closed_count")}
    rep, clo = calendar_activity(tri["reported_count"], 2021), calendar_activity(tri["closed_count"], 2021)
    lp, li = calendar_activity(tri["paid"], 2021), calendar_activity(tri["incurred"], 2021)
    assert rep.astype(int).tolist() == [527, 551, 561, 596, 642] and clo.astype(int).tolist() == [346, 515, 543, 585, 591]
    assert lp.round(0).tolist() == [1597280, 2028456, 2499020, 2426141, 2898624]
    assert li.round(0).tolist() == [2328995, 2020638, 2524535, 2775523, 3169501]
    u = synthetic_ulae(rep, clo)
    assert u.round(0).tolist() == [80781, 98213, 104402, 114874, 124320]
    assert paid_to_paid(u, lp).round(4).tolist() == [0.0506, 0.0484, 0.0418, 0.0473, 0.0429]
    r = u.loc[2023:].sum() / lp.loc[2023:].sum()
    k = u.loc[2023:].sum() / ((lp + li) / 2).loc[2023:].sum()
    assert (round(u.loc[2023:].sum()), round(lp.loc[2023:].sum()), round(r, 4), round(k, 4)) == (343596, 7823786, 0.0439, 0.0422)
    t5 = {m: triangle(C, T, m, 2021, 2025) for m in ("paid", "incurred", "reported_count", "closed_count")}
    d = develop(t5["incurred"], cdf(t5["incurred"]))
    case = (d.latest - latest_diagonal(t5["paid"])["latest"]).sum()
    ibnr = (d.ultimate - d.latest).sum()
    assert (round(case), round(ibnr)) == (1947963, 402044)
    cnt = develop(t5["reported_count"], cdf(t5["reported_count"]))
    fr = (cnt.ultimate - cnt.latest).sum()
    fc = (cnt.ultimate - latest_diagonal(t5["closed_count"])["latest"]).sum()
    assert (round(fr, 1), round(fc, 1)) == (51.5, 486.5)
    tf = (TRUE.rename(lambda x: x).astype(float) * 0 + pd.Series([524, 548, 580, 590, 654], index=TRUE.index))
    true_fr = (tf - cnt.latest).sum()
    true_fc = (tf - latest_diagonal(t5["closed_count"])["latest"]).sum()
    assert (true_fr, true_fc, round((120 * true_fr + 80 * true_fc) * 1.03)) == (61, 496, 48410)
    assert round(classical_unpaid(r, case, ibnr)) == 60431
    assert round(classical_unpaid(r, case, ibnr, 1.0)) == 103205
    assert round(classical_unpaid(k, case, ibnr)) == 58036
    ur, uc = unit_costs(u[2025], rep[2025], clo[2025])
    assert (round(ur, 2), round(uc, 2)) == (100.83, 100.83)
    assert round(count_based_unpaid(ur, uc, fr, fc, 0.03)) == 55863
    assert round(count_based_unpaid(120, 80, fr, fc, 0.03)) == 46443
    assert np.allclose(np.linalg.solve([[596, 585], [642, 591]], [u[2024] * 1.03, u[2025]]), [120, 80])


def test_ulae_hand():
    assert classical_unpaid(0.05, 2_000_000, 800_000) == pytest.approx(90_000)
    ur, uc = unit_costs(300_000, 1000, 900)
    assert round(ur, 2) == 157.89 and round(count_based_unpaid(ur, uc, 100, 400, 0.03)) == 81316
    assert classical_unpaid(0.05, 2_000_000, 800_000, 0.4) == pytest.approx(80_000)
    assert classical_unpaid(0.05, 2_000_000, 800_000, 0.6) == pytest.approx(100_000)


# ------------------------------------------------------------ Chapter 22

def test_mack_example():
    M = mack(EX4)
    assert M["f"].round(5).tolist() == [1.5, 1.09375, 1.0303]
    assert M["sigma2"].round(4).tolist() == [2.1780, 0.1103, 0.0056]
    assert M["reserve"].round(2).tolist()[1:] == [56.06, 222.06, 897.44]
    assert M["se"].round(2).tolist()[1:] == [4.68, 18.44, 73.53]
    assert (round(M["total_reserve"], 2), round(M["total_se"], 2)) == (1175.57, 78.08)
    f, s2, full = M["f"].to_numpy(), M["sigma2"].to_numpy(), M["full"].to_numpy()
    den = [EX4.iloc[:3 - j, j].sum() for j in range(3)]
    U = full[3, -1]
    proc = U ** 2 * sum(s2[j] / f[j] ** 2 / full[3, j] for j in range(3))
    par = U ** 2 * sum(s2[j] / f[j] ** 2 / den[j] for j in range(3))
    assert (round(np.sqrt(proc), 2), round(np.sqrt(par), 2)) == (61.93, 39.64)


def test_mack_synthetic(paid):
    M = mack(paid)
    assert M["reserve"].round(0).tolist() == [0, 14683, 72131, 447508, 1628571]
    assert M["se"].round(0).tolist() == [0, 33, 1145, 42497, 252331]
    assert (round(M["total_reserve"]), round(M["total_se"])) == (2162893, 258122)
    assert M["sigma2"].round(1).tolist()[:3] == [23105.3, 491.0, 0.4]
    assert round(M["latest"].sum()) == 10603063


def test_mack_chainladder(paid):
    cl = pytest.importorskip("chainladder")
    tri = cl.Triangle(to_long(paid, "paid"), origin="accident_year", development="valuation",
                      columns=["paid"], cumulative=True)
    mk = cl.MackChainladder().fit(cl.Development(sigma_interpolation="mack").fit_transform(tri))
    assert float(mk.total_mack_std_err_.values.ravel()[0]) == pytest.approx(mack(paid)["total_se"], rel=1e-6)


def test_bootstrap(paid, boot):
    assert round(boot["phi"]) == 9452 and len(boot["residuals"]) == 15
    s = summarize(boot["reserves"])
    assert [round(s[k]) for k in ("mean", "sd", "q50", "q75", "q90", "q95", "q99.5")] == \
        [2173623, 247953, 2163175, 2330695, 2497112, 2602680, 2877097]
    true = 13134540 - mack(paid)["latest"].sum()
    assert round(true) == 2531477 and round((boot["reserves"] <= true).mean(), 2) == 0.92
    qs = [round(np.quantile(odp_bootstrap(paid, 1000, seed=k)["reserves"], 0.995)) for k in range(1, 6)]
    assert (min(qs), max(qs)) == (2822012, 2967596)


# ------------------------------------------------------------ Chapter 23

def test_discounting(paid, boot):
    cf = future_cash_flows(paid, 2025)
    assert cf.round(0).tolist() == [1581053, 468759, 91639, 21441]
    assert [round(present_value(cf, 2025, r)) for r in (0.03, 0.05)] == [2110732, 2077820]
    assert round(present_value(cf, 2025, [0.025, 0.028, 0.030, 0.032])) == 2115706
    s = boot["reserves"]
    scr0 = np.quantile(s, 0.995) - s.mean()
    rm = cost_of_capital_margin(cf, scr0, 2025)
    assert (round(scr0), round(rm["margin"]), round(np.quantile(s, 0.75) - s.mean())) == (703474, 54073, 157072)
    pv = present_value(cf, 2025, 0.03)
    assert round(pv / cf.sum(), 4) == 0.9759 and round(pv + rm["margin"]) == 2164806
    assert round(confidence_level(s * pv / cf.sum(), pv + rm["margin"]), 3) == 0.587


def test_discount_hand():
    cf = pd.Series({2026: 600_000.0, 2027: 300_000.0, 2028: 100_000.0})
    assert discount_factors(cf.index, 2025, 0.04).round(5).tolist() == [0.98058, 0.94287, 0.90660]
    assert round(present_value(cf, 2025, 0.04)) == 961868
    assert round(cost_of_capital_margin(cf, 150_000, 2025, 0.06, 0.04)["margin"]) == 12782
    assert round(present_value(pd.Series(100_000.0, index=range(2026, 2036)), 2025, 0.04)) == 827152


# ------------------------------------------------------------ Chapter 24

def test_premium_liability(hist):
    P, _, _ = hist
    e = exposure_table(P[P.start.dt.year == 2025], 2025)
    upr = e.unearned_premium.sum()
    ue = len(e) - e.earned_exposure.sum()
    assert (round(upr), round(ue, 2), round(e.earned_exposure.sum(), 2)) == (1986887, 3192.31, 3188.69)
    pp = 564.1499927571194 / 1.04 ** 1.5
    pl = premium_liability(ue, pp, 0.1239, 0.05 * upr, 1.03 ** -0.5)
    assert (round(pp, 2), round(pl["claims"]), round(pl["lae"]), round(pl["maintenance"]), round(pl["total"])) == \
        (531.92, 1698045, 210388, 99344, 1978322)
    assert round(premium_deficiency(upr, 0.10 * upr, pl["total"])) == 190123
    assert premium_deficiency(1_000_000, 150_000, 2000 * 350 * 1.10 + 40_000) == 0.0


def test_ae(hist):
    _, C, T = hist
    r = ae_review(triangle(C, T, "paid", 2021, 2024), triangle(C, T, "paid", 2021, 2025))
    assert r.prior_ultimate.round(0).tolist() == [2308768, 2091811, 2388916, 2839082]
    assert r.expected.round(0).tolist() == [0, 49592, 307525, 1062427]
    assert r.actual.round(0).tolist() == [16215, 48404, 287211, 1085861]
    assert r.bf_update.round(0).tolist() == [2324983, 2090622, 2368603, 2862516]
    assert r.dev_update.round(0).tolist() == [2324983, 2105305, 2384097, 2877240]
    assert (round(r.expected.sum()), round(r.actual.sum())) == (1419544, 1437691)
    ri = ae_review(triangle(C, T, "incurred", 2021, 2024), triangle(C, T, "incurred", 2021, 2025))
    assert (round(ri.expected.sum()), round(ri.actual.sum()), round(ri.a_over_e[2024], 2)) == (325214, 334223, 1.21)
    assert round(16215 / 2308768 + 1, 4) == 1.0070 and round(mack(triangle(C, T, "paid", 2021, 2025))["f"].iloc[-1], 4) == 1.0070
