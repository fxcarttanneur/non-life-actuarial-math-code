"""Unit tests for src/giruin.py: every figure quoted in Chapter 12 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from giaggregate import discretize, grid_moments, grid_quantile, panjer  # noqa: E402
from gidata import synthetic_portfolio  # noqa: E402
from gilossmodels import fit_mle, fit_summary  # noqa: E402
from giruin import (adjustment_coefficient, capital_for_ruin, exceedance,  # noqa: E402
                    expected_value_premium, exponential_R, exponential_ruin,
                    loading_for_ruin, simulate_ruin, std_premium, variance_premium)

B = 1 / 4000


@pytest.fixture(scope="module")
def dists():
    _, cl = synthetic_portfolio()
    x = cl["ultimate_amount"].to_numpy()
    fs = fit_summary(x)
    L = fs.loc["lognormal", "params"]
    X = stats.lognorm(L["sigma"], scale=np.exp(L["mu"]))
    out = {}
    for lam, h in ((48.5, 200), (485, 500)):
        g = panjer("poisson", discretize(X.cdf, h, 2 ** 13), lam=lam)
        out[lam] = (g, h) + grid_moments(g, h)
    return fs, L, out


def test_premium_principles(dists):
    _, _, out = dists
    g, h, m, v = out[48.5]
    assert (round(m), round(np.sqrt(v))) == (195919, 56213)
    w, k = 0.1 * m / v, 0.1 * m / np.sqrt(v)
    assert (round(w * 1e6, 3), round(k, 4)) == (6.200, 0.3485)
    rows = {}
    for lam in (48.5, 485):
        g, h, M, V = out[lam]
        n = lam / 0.097
        P = [expected_value_premium(M, 0.1), variance_premium(M, V, w), std_premium(M, V, k),
             grid_quantile(g, h, 0.75)]
        rows[lam] = ([round(p) for p in P], [round((p - M) / n, 2) for p in P],
                     [round(exceedance(g, h, p), 3) for p in P])
    assert rows[48.5] == ([215511, 215511, 215511, 226600], [39.18, 39.18, 39.18, 61.36],
                          [0.313, 0.313, 0.313, 0.250])
    assert rows[485] == ([2154968, 2155121, 2021039, 2071500], [39.18, 39.21, 12.40, 22.49],
                         [0.134, 0.134, 0.344, 0.250])
    assert round(out[485][2]) == 1959062 and round(np.sqrt(out[485][3])) == 177825


def test_exponential_ruin():
    R = exponential_R(B, 0.10)
    assert round(R * 1e5, 4) == 2.2727
    assert adjustment_coefficient(50, 220_000, lambda r: B / (B - r), B) == pytest.approx(R, rel=1e-6)
    assert [round(exponential_ruin(u, B, 0.10), 4) for u in (0, 25_000, 50_000, 100_000, 200_000)] == \
        [0.9091, 0.5150, 0.2918, 0.0937, 0.0097]
    assert round(np.exp(-R * 50_000), 4) == 0.3210
    assert round(np.log(2) / R) == 30498
    assert round(exponential_ruin(20_000, B, 0.10), 4) == 0.5770


def test_tradeoff():
    t5 = [round(loading_for_ruin(u, B, 0.05), 4) for u in (50_000, 100_000, 200_000)]
    t1 = [round(loading_for_ruin(u, B, 0.01), 4) for u in (50_000, 100_000, 200_000)]
    assert t5 == [0.2817, 0.1299, 0.0624] and t1 == [0.5053, 0.2142, 0.0992]
    assert [round(capital_for_ruin(B, t, 0.01)) for t in (0.1, 0.2, 0.3)] == [198434, 106148, 75275]
    assert round(exponential_ruin(100_000, B, 0.1299), 3) == 0.050
    assert capital_for_ruin(B, 0.1, 0.95) == 0.0


def test_gamma_R(dists):
    fs, _, _ = dists
    G = fs.loc["gamma", "params"]
    a, b = G["alpha"], G["beta"]
    m = a / b
    R = adjustment_coefficient(48.5, 1.1 * 48.5 * m, lambda r: (b / (b - r)) ** a, b)
    assert round(R * 1e5, 4) == 2.2176 and round(exponential_R(1 / m, 0.1) * 1e5, 4) == 2.3467


def test_simulation(dists):
    _, L, _ = dists
    sim = [simulate_ruin(50_000, 50, 220_000, lambda r, k: r.exponential(4000, k), T, 20_000)
           for T in (1, 10, 50)]
    assert [round(s * 20_000) for s in sim] == [2271, 5636, 5841]   # 11.36%, 28.18%, 29.21%
    mL = np.exp(L["mu"] + L["sigma"] ** 2 / 2)
    c = 1.1 * 48.5 * mL
    ln = [simulate_ruin(50_000, 48.5, c, lambda r, k: r.lognormal(L["mu"], L["sigma"], k), T, 20_000)
          for T in (1, 10)]
    ex = [simulate_ruin(50_000, 48.5, c, lambda r, k: r.exponential(mL, k), T, 20_000) for T in (1, 10)]
    assert [round(s, 4) for s in ln] == [0.2113, 0.4441] and [round(s * 20_000) for s in ex] == [2376, 5719]


def test_one_year(dists):
    _, _, out = dists
    g, h, m, _ = out[48.5]
    q = grid_quantile(g, h, 0.995)
    assert (q, round((q - 150_000) / m - 1, 4)) == (395600, 0.2536)
    assert round(((q - 150_000) - m) / 500, 2) == 99.36
    g, h, m, _ = out[485]
    q = grid_quantile(g, h, 0.995)
    assert (q, round((q - 300_000) / m - 1, 4)) == (2482000, 0.1138)
    assert round(((q - 300_000) - m) / 5000, 2) == 44.59
