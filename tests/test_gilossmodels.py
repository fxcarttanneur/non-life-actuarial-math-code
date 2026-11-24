"""Unit tests for src/gilossmodels.py: every figure quoted in Chapter 9 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gidata import synthetic_portfolio  # noqa: E402
from gilossmodels import (ab0_ratios, chi_square, claim_counts, count_table,  # noqa: E402
                          expected_counts, fit_mle, fit_moments, fit_summary, frozen,
                          lognormal_mean_excess, loglik, mean_excess, negbin_moments,
                          negbin_pmf, poisson_fit, top_share)


@pytest.fixture(scope="module")
def port():
    pol, cl = synthetic_portfolio()
    return pol, cl, cl["ultimate_amount"].to_numpy()


def expand(table):
    t = pd.Series(table)
    return t, t.index.repeat(t.values).to_series()


def test_portfolio_counts(port):
    pol, cl, _ = port
    n = claim_counts(pol, cl)
    assert count_table(n).tolist() == [4531, 453, 16]
    assert poisson_fit(n) == pytest.approx(0.097)
    assert n.var(ddof=1) == pytest.approx(0.0940, abs=1e-4)
    with pytest.raises(ValueError):
        negbin_moments(n)
    assert ab0_ratios(count_table(n)).round(3).tolist() == [0.100, 0.071]
    oe = expected_counts(count_table(n), lambda k: stats.poisson.pmf(k, 0.097))
    assert oe["expected"].round(2).tolist() == [4537.78, 440.16, 22.06]
    st, df, pv = chi_square(oe, 1)
    assert (round(st, 2), df, round(pv, 2)) == (2.05, 1, 0.15)


def test_fleet():
    f, c = expand([860, 110, 22, 6, 2])
    assert c.var(ddof=1) == pytest.approx(0.2519, abs=1e-4)
    r, p = negbin_moments(c)
    assert (round(p, 4), round(r, 4)) == (0.7147, 0.4509)
    assert ab0_ratios(f).round(3).tolist() == [0.128, 0.400, 0.818, 1.333]
    nb = expected_counts(f, lambda k: negbin_pmf(k, r, p))
    assert nb["expected"].round(2).tolist() == [859.45, 110.57, 22.88, 5.33, 1.76]
    po = expected_counts(f, lambda k: stats.poisson.pmf(k, 0.18))
    assert po["expected"].round(2).tolist() == [835.27, 150.35, 13.53, 0.81, 0.04]
    assert po["expected"][2:].sum() == pytest.approx(14.38, abs=0.005)
    assert round(chi_square(po, 1)[0], 2) == 28.52 and chi_square(po, 1)[1] == 1
    assert round(chi_square(nb, 2)[0], 2) == 0.15 and chi_square(nb, 2)[1] == 1
    # recursion constants
    assert (1 - p) == pytest.approx(0.2853, abs=1e-4)
    assert (r - 1) * (1 - p) == pytest.approx(-0.1566, abs=1e-4)


def test_five_claims():
    s = np.array([500, 1200, 2000, 3500, 7800.0])
    assert s.var() == 6_756_000
    g = fit_moments(s, "gamma")
    assert (round(g["alpha"], 3), round(g["beta"], 6)) == (1.332, 0.000444)
    lm = fit_moments(s, "lognormal")
    assert (round(lm["mu"], 4), round(lm["sigma"], 4)) == (7.7264, 0.7483)
    ln = fit_mle(s, "lognormal")
    assert (round(ln["mu"], 4), round(ln["sigma"], 4)) == (7.6056, 0.9325)
    gm = fit_mle(s, "gamma")
    assert (round(gm["alpha"], 3), round(gm["beta"], 6)) == (1.391, 0.000464)
    lls = [loglik(s, d, fit_mle(s, d)) for d in ("exponential", "gamma", "lognormal")]
    assert [round(v, 2) for v in lls] == [-45.03, -44.88, -44.77]
    assert frozen("exponential", fit_mle(s, "exponential")).sf(10_000) == pytest.approx(0.0357, abs=5e-5)
    assert frozen("lognormal", ln).sf(10_000) == pytest.approx(0.0426, abs=5e-5)
    assert 2 + 2 * 45.03 == pytest.approx(92.06) and round(4 - 2 * lls[2], 2) == 93.55


def test_portfolio_fits(port):
    _, _, x = port
    assert (x.size, round(x.mean(), 2), round(np.median(x), 2), round(x.std(), 2), x.max()) == \
        (485, 3873.94, 1967.71, 5197.30, 37183.80)
    fs = fit_summary(x)
    assert fs["loglik"].round(2).tolist() == [-4492.08, -4490.21, -4458.68, -4471.98, -4485.24]
    assert fs["aic"].round(2).tolist() == [8986.17, 8984.41, 8921.37, 8947.97, 8974.48]
    assert fs["ks"].round(3).tolist() == [0.115, 0.097, 0.022, 0.053, 0.074]
    assert fs["mean"].round(2).tolist() == [3873.94, 3873.94, 4039.53, 3898.85, 3838.22]
    p = fs["params"]
    assert round(p["lognormal"]["mu"], 3) == 7.611 and round(p["lognormal"]["sigma"], 3) == 1.177
    assert round(p["pareto"]["alpha"], 3) == 3.593 and round(p["pareto"]["theta"], 2) == 10109.81
    assert round(p["gamma"]["alpha"], 3) == 0.899 and round(p["weibull"]["tau"], 3) == 0.888
    over20 = [485 * frozen(d, p[d]).sf(20_000) for d in fs.index]
    assert [round(v, 1) for v in over20] == [2.8, 3.7, 12.5, 9.6, 5.1]
    over10 = [485 * frozen(d, p[d]).sf(10_000) for d in fs.index]
    assert [round(v, 1) for v in over10] == [36.7, 39.6, 42.3, 41.0, 41.2]
    assert [(x > d).sum() for d in (10_000, 20_000)] == [46, 13]


def test_tail(port):
    _, _, x = port
    me = mean_excess(x, [0, 1000, 2500, 5000, 10000, 20000])
    assert me.round(2).tolist() == [3873.94, 4077.12, 5149.52, 6444.75, 7174.65, 5692.16]
    assert [(x > d).sum() for d in (1000, 2500, 5000)] == [356, 203, 105]
    ln = fit_mle(x, "lognormal")
    th = [lognormal_mean_excess(ln["mu"], ln["sigma"], d) for d in (1000, 2500, 5000, 10000, 20000)]
    assert [round(v, 2) for v in th] == [4359.99, 5426.57, 7042.55, 9838.09, 14607.17]
    assert [round(top_share(x, q), 3) for q in (0.01, 0.05, 0.10)] == [0.087, 0.283, 0.436]


def test_scipy_conversion():
    # gamma with rate 0.5 has mean alpha / beta = 4
    assert frozen("gamma", {"alpha": 2, "beta": 0.5}).mean() == pytest.approx(4)
    assert frozen("pareto", {"alpha": 3, "theta": 2000}).mean() == pytest.approx(1000)
