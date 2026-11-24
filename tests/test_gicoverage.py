"""Unit tests for src/gicoverage.py: every figure quoted in Chapter 10 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gicoverage import (franchise_per_loss, inflate, layer_cost, ler, lev,  # noqa: E402
                        lev_empirical, payment, per_loss, per_payment, survival)
from gidata import synthetic_portfolio  # noqa: E402
from gilossmodels import fit_mle  # noqa: E402

P = {"alpha": 3, "theta": 8000}
E = {"beta": 1 / 2000}


@pytest.fixture(scope="module")
def port():
    _, cl = synthetic_portfolio()
    x = cl["ultimate_amount"].to_numpy()
    return x, fit_mle(x, "lognormal")


def r2(v):
    return round(float(v), 2)


def test_pareto_policy():
    assert r2(lev("pareto", P, 500)) == 456.75 and r2(lev("pareto", P, 25_000)) == 3764.92
    assert r2(per_loss("pareto", P, 500, 25_000, 0.8)) == 2646.54
    assert round(survival("pareto", P, 500), 6) == 0.833706
    assert r2(per_payment("pareto", P, 500, 25_000, 0.8)) == 3174.43
    assert round(ler("pareto", P, 500), 4) == 0.1142
    assert r2(0.2 * (lev("pareto", P, 25_000) - lev("pareto", P, 500))) == 661.63


def test_pareto_inflation():
    Pi = inflate("pareto", P, 0.10)
    assert Pi["theta"] == pytest.approx(8800)
    assert r2(lev("pareto", Pi, 500)) == 460.40 and r2(lev("pareto", Pi, 25_000)) == 4101.75
    a, b = per_loss("pareto", P, 500, 25_000, 0.8), per_loss("pareto", Pi, 500, 25_000, 0.8)
    assert r2(b) == 2913.08 and round(b / a - 1, 4) == 0.1007
    d0, d1 = per_loss("pareto", P, 500, c=0.8), per_loss("pareto", Pi, 500, c=0.8)
    assert (r2(d0), r2(d1), round(d1 / d0 - 1, 4)) == (2834.60, 3151.68, 0.1119)
    u0, u1 = per_loss("pareto", P, 0, 25_000, 0.8), per_loss("pareto", Pi, 0, 25_000, 0.8)
    assert (r2(u0), r2(u1), round(u1 / u0 - 1, 4)) == (3011.94, 3281.40, 0.0895)


def test_portfolio_lev(port):
    x, L = port
    assert [r2(lev_empirical(x, d)) for d in (250, 500, 1000, 2500)] == [246.54, 476.55, 881.24, 1718.57]
    assert [round(lev_empirical(x, d) / x.mean(), 4) for d in (250, 500, 1000, 2500)] == \
        [0.0636, 0.1230, 0.2275, 0.4436]
    assert [r2(lev("lognormal", L, d)) for d in (250, 500, 1000, 2500)] == [246.89, 477.72, 878.41, 1715.38]
    assert [round(ler("lognormal", L, d), 4) for d in (250, 500, 1000, 2500)] == [0.0611, 0.1183, 0.2175, 0.4246]
    assert [round(survival("lognormal", L, d), 4) for d in (250, 500, 1000, 2500)] == [0.9621, 0.8823, 0.7250, 0.4283]
    us = (5000, 10000, 25000, 50000, np.inf)
    assert [r2(lev_empirical(x, u)) for u in us] == [2478.68, 3193.45, 3792.81, 3873.94, 3873.94]
    assert [r2(lev("lognormal", L, u)) for u in us] == [2484.95, 3182.36, 3766.34, 3954.69, 4039.53]


def test_layer_inflation(port):
    _, L = port
    rows = []
    for r in (0, 0.05, 0.10):
        Li = inflate("lognormal", L, r)
        rows.append((r2(lev("lognormal", Li, np.inf)), r2(layer_cost("lognormal", Li, 0, 10_000)),
                     r2(layer_cost("lognormal", Li, 10_000, 10_000))))
    assert rows == [(4039.53, 3182.36, 481.32), (4241.50, 3296.26, 523.62), (4443.48, 3406.92, 566.65)]
    b = layer_cost("lognormal", L, 10_000, 10_000)
    assert round(layer_cost("lognormal", inflate("lognormal", L, 0.04), 10_000, 10_000) / b - 1, 2) == 0.07
    two = layer_cost("lognormal", inflate("lognormal", L, 1.05 ** 2 - 1), 10_000, 10_000)
    assert (r2(two), round(485 * b), round(485 * two), round(two / b - 1, 3)) == (568.81, 233442, 275875, 0.182)


def test_per_policy(port):
    x, L = port
    pl = per_loss("lognormal", L, 500, 25_000)
    assert r2(pl) == 3288.62 and r2(0.097 * pl) == 319.00
    assert r2(0.097 * lev("lognormal", L, np.inf)) == 391.83
    assert round(0.097 * survival("lognormal", L, 500), 4) == 0.0856
    # thinning identity: lambda E[Y^L] = lambda S(d) E[Y^P]
    assert 0.097 * pl == pytest.approx(0.097 * survival("lognormal", L, 500) * per_payment("lognormal", L, 500, 25_000))
    rho = (lev("lognormal", L, np.inf) - lev("lognormal", L, 25_000)) / lev("lognormal", L, 25_000)
    assert round(rho, 4) == 0.0725
    assert round((x.mean() - lev_empirical(x, 25_000)) / lev_empirical(x, 25_000), 4) == 0.0214
    assert payment(x, 500, 25_000).mean() == pytest.approx(lev_empirical(x, 25_000) - lev_empirical(x, 500))


def test_exercises():
    assert r2(lev("exponential", E, 250)) == 235.01 and round(ler("exponential", E, 250), 4) == 0.1175
    assert r2(per_payment("exponential", E, 250)) == 2000.00
    Ei = inflate("exponential", E, 0.10)
    assert r2(per_loss("exponential", Ei, 250)) == 1963.68
    assert round(per_loss("exponential", Ei, 250) / per_loss("exponential", E, 250) - 1, 4) == 0.1126
    assert r2(per_payment("exponential", Ei, 250)) == 2200.00
    assert r2(per_loss("pareto", P, 500)) == 3543.25
    assert r2(franchise_per_loss("pareto", P, 500)) == 3960.11
