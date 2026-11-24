"""Figures of Chapters 25-27 (reinsurance)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gitriangle import synthetic_history, triangle, exposure_by_accident_year
from gidevelop import average_factors, cumulative_factors, develop
from gilossmodels import fit_mle
from gicoverage import lev
from gireins import (layer, quota_share, surplus_share, simulate_layers, risk_transfer, burning_cost,
                     exposure_rate, simulate_layer_years, xl_with_reinstatements, reinstatement_premium,
                     cat_event_set, cat_layer_cost, ceded_triangle, true_ceded, excess_factor)

YEARS = range(2021, 2026)


@pytest.fixture(scope="module")
def data():
    return synthetic_history()


def cdf(t):
    return cumulative_factors(average_factors(t, "volume"), 1.0)


def test_hand_examples():
    x = np.array([20_000, 80_000, 180_000, 400_000])
    assert quota_share(x, 0.4)[0].sum() == 272_000
    assert layer(x, 100_000, 150_000).sum() == 230_000
    net_qs = quota_share(x, 0.4)[1]
    assert layer(net_qs, 100_000, 150_000).sum() == 148_000
    assert np.round(surplus_share([80_000, 300_000, 800_000], 100_000, 4), 4).tolist() == [0.0, 0.6667, 0.5]
    assert layer([8.6e6, 11e6], 8e6, 2e6).tolist() == [600_000, 2_000_000]
    L = np.array([0] * 70 + [50] * 20 + [150] * 8 + [400] * 2, float)
    r = risk_transfer(L, 60)
    assert r["p_loss_10"] == pytest.approx(0.10) and round(r["erd"], 4) == 0.2333
    assert reinstatement_premium(600_000, 1e6, 1, 200_000) == pytest.approx(120_000)
    assert round(excess_factor(1.12, 1.08, 10e6, 9e6), 4) == 1.48


def test_ch25_portfolio(data):
    _, c, _ = data
    x = c[c.accident_date.dt.year == 2025].ultimate_amount.to_numpy()
    assert (round(quota_share(x, 0.3)[0].sum()), round(layer(x, 25_000, 25_000).sum())) == (997693, 270543)
    L = fit_mle(x, "lognormal")
    mu, s = float(L["mu"]), float(L["sigma"])
    sim = simulate_layers(654, lambda r, n: r.lognormal(mu, s, n), [(25_000, 25_000)], 20_000, seed=25)
    xl = sim["25,000 xs 25,000"]
    net = sim["gross"] - xl
    assert round(np.quantile(net, 0.995) - net.mean()) == 669982
    P = xl.mean() / 0.8
    r = risk_transfer(xl, P)
    assert round(P) == 276797 and round(r["p_loss_10"], 3) == 0.105 and round(r["erd"], 3) == 0.029
    assert risk_transfer(np.minimum(xl, 1.05 * P), P)["p_loss_10"] == 0.0


def test_ch26(data):
    p, c, t = data
    ex = exposure_by_accident_year(p, YEARS)["earned_exposure"]
    k = pd.Series([1.04 ** (2026 - y) for y in YEARS], index=list(YEARS))
    tt = ceded_triangle(c, t, "incurred", 25_000, 25_000, 2021, 2025, trend=0.04, base_year=2026).mul(k, axis=0)
    ult = develop(tt, cdf(tt))["ultimate"]
    assert ult.round(0).tolist() == [197290, 94047, 97028, 258546, 258645]
    assert round(ult.sum() / ex.sum(), 2) == 32.01
    bc = burning_cost(c, 25_000, 25_000, 0.04, 2026.5, YEARS, ex)
    assert round(bc["layer"].sum() / ex.sum(), 2) == 35.17
    P = {"alpha": 1.5, "theta": 50_000}
    ilf = lambda u: lev("pareto", P, u)
    sp = {250_000: 2e6, 500_000: 3e6, 1_000_000: 1.5e6}
    assert round(sum(0.65 * v * exposure_rate(ilf, k_, 250_000, 250_000) for k_, v in sp.items())) == 431099
    x = c[c.accident_date.dt.year == 2025].ultimate_amount.to_numpy()
    L = fit_mle(x, "lognormal")
    mu, s = float(L["mu"]) + np.log(1.04), float(L["sigma"])
    samp = lambda r, n: r.lognormal(mu, s, n)
    sim = simulate_layers(654 * 1.05, samp, [(25_000, 25_000), (50_000, 50_000), (100_000, 100_000)], 20_000, seed=26)
    assert sim.mean().round(0).tolist()[1:] == [251453, 105832, 33530]
    yrs = simulate_layer_years(654 * 1.05, samp, 20_000, seed=26)
    R = xl_with_reinstatements(yrs, 50_000, 50_000, 1)
    assert round((R.mean() / 0.8) / (1 + np.minimum(R, 50_000).mean() / 50_000)) == 51208
    assert round(cat_layer_cost(cat_event_set(20), 250_000, 500_000)) == 23514


def test_ch27(data):
    _, c, t = data
    ci = ceded_triangle(c, t, "incurred", 25_000, 25_000, 2021, 2025)
    cp = ceded_triangle(c, t, "paid", 25_000, 25_000, 2021, 2025)
    assert round(develop(ci, cdf(ci))["ultimate"].sum()) == 707726
    assert round(develop(cp, cdf(cp))["ultimate"].sum()) == 776659
    assert round(true_ceded(c, 25_000, 25_000, YEARS).sum()) == 781019
    gi = triangle(c, t, "incurred", 2021, 2025)
    li = ceded_triangle(c, t, "incurred", 0, 25_000, 2021, 2025)
    xi = gi - li
    fu, fl = average_factors(gi, "volume").iloc[0], average_factors(li, "volume").iloc[0]
    assert excess_factor(fu, fl, gi[12].iloc[:4].sum(), li[12].iloc[:4].sum()) == pytest.approx(
        average_factors(xi, "volume").iloc[0])
    assert round(develop(xi, cdf(xi))["ultimate"].sum()) == 1094801


def test_ch26_tail_comparison(data):
    from gireins import pareto_tail, pareto_tail_layer
    from gicoverage import layer_cost
    _, c, _ = data
    ay = c.accident_date.dt.year
    m = ay.between(2021, 2025)
    x = (c.ultimate_amount[m] * 1.04 ** (2026 - ay[m])).to_numpy()
    a = pareto_tail(x, 25_000)
    assert round(a, 2) == 2.35 and (x > 25_000).sum() == 84
    x25 = (c.ultimate_amount[ay == 2025] * 1.04).to_numpy()
    n_large = 654 * 1.05 * (x25 > 25_000).mean()
    mu = 7.6 + 5.5 * np.log(1.04)
    got = [(round(layer(x25, A, L).sum()), round(layer(x, A, L).sum() / 5),
            round(n_large * pareto_tail_layer(a, 25_000, A, L)),
            round(654 * 1.05 * layer_cost("lognormal", {"mu": mu, "sigma": 1.2}, A, L)))
           for A, L in [(25_000, 25_000), (50_000, 50_000), (100_000, 100_000)]]
    assert got == [(308328, 199002, 342782, 228385), (22712, 57770, 134565, 91327), (0, 30753, 52826, 27068)]
