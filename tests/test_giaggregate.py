"""Unit tests for src/giaggregate.py: every figure quoted in Chapter 11 is checked here."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from giaggregate import (ab0, compound_moments, compound_poisson_skewness, discretize,  # noqa: E402
                         fft_aggregate, grid_moments, grid_quantile, grid_tvar,
                         lognormal_approx, lognormal_raw_moment, normal_approx, panjer,
                         sample_tvar, simulate, translated_gamma)
from gidata import synthetic_portfolio  # noqa: E402
from gilossmodels import fit_mle  # noqa: E402

QS = (0.5, 0.9, 0.95, 0.99, 0.995)
NB = dict(r=10, p=10 / 58.5)


@pytest.fixture(scope="module")
def sev():
    _, cl = synthetic_portfolio()
    L = fit_mle(cl["ultimate_amount"].to_numpy(), "lognormal")
    m = [lognormal_raw_moment(L["mu"], L["sigma"], k) for k in (1, 2, 3)]
    X = stats.lognorm(L["sigma"], scale=np.exp(L["mu"]))
    return L, m, X


@pytest.fixture(scope="module")
def book(sev):
    _, _, X = sev
    fx = discretize(X.cdf, 200, 2 ** 13)
    return fx, panjer("poisson", fx, 2 ** 13, lam=48.5), panjer("negbin", fx, 2 ** 13, **NB)


def test_hand_examples():
    g = panjer("poisson", np.array([0, 0.5, 0.3, 0.2]), 8, lam=2)
    assert g[:4].round(6).tolist() == [0.135335, 0.135335, 0.148869, 0.157891]
    assert round(g[:4].sum(), 4) == 0.5774 and round(g[:3].sum(), 4) == 0.4195
    assert compound_moments(2, 2, 1.7, 3.5 - 1.7 ** 2) == pytest.approx((3.4, 7.0))
    assert ab0("negbin", r=2, p=0.5) == (0.5, 0.5)
    g2 = panjer("negbin", np.array([0, 0.6, 0.4]), 4, r=2, p=0.5)
    assert g2.round(6).tolist() == [0.25, 0.15, 0.1675, 0.117]
    assert compound_moments(2, 4, 1.4, 0.24) == pytest.approx((2.8, 8.32))
    # FFT gives the same as Panjer on a long grid
    assert np.allclose(fft_aggregate("poisson", np.array([0, 0.5, 0.3, 0.2]), 64, lam=2)[:8], g)


def test_moments(sev):
    _, (m1, m2, m3), _ = sev
    assert round(m1, 2) == 4039.53 and round(m2 / 1e7, 4) == 6.5191 and round(m3 / 1e12, 4) == 4.2031
    ES, VS = compound_moments(48.5, 48.5, m1, m2 - m1 ** 2)
    assert round(ES) == 195917 and round(VS / 1e9, 4) == 3.1618 and round(np.sqrt(VS)) == 56230
    assert round(compound_poisson_skewness(48.5, m2, m3), 3) == 1.147
    EF, VF = compound_moments(485, 485, m1, m2 - m1 ** 2)
    assert (round(EF), round(np.sqrt(VF)), round(compound_poisson_skewness(485, m2, m3), 3)) == \
        (1959170, 177814, 0.363)
    EN, VN = compound_moments(48.5, 48.5 / NB["p"], m1, m2 - m1 ** 2)
    assert round(48.5 / NB["p"], 2) == 283.73 and round(VN / 1e9, 4) == 7.0001 and round(np.sqrt(VN)) == 83667


def test_panjer_book(sev, book):
    fx, g, _ = book
    assert round(float((np.arange(fx.size) * 200 * fx).sum()), 2) == 4039.59
    assert 1 - g.sum() < 1e-6
    m, v = grid_moments(g, 200)
    assert round(m) == 195919 and round(v / 1e9, 4) == 3.1599
    assert [grid_quantile(g, 200, q) for q in QS] == [188800, 267600, 296200, 363800, 395600]
    assert round(grid_tvar(g, 200, 0.99)) == 415682


def test_fft_and_wrap(sev, book):
    fx, g, gnb = book
    gf = fft_aggregate("poisson", fx, 2 ** 14, lam=48.5)[:2 ** 13]
    assert np.abs(gf - g).max() < 1e-15
    gw = fft_aggregate("poisson", fx, 2 ** 11, lam=48.5)
    assert grid_quantile(gw, 200, 0.99) == 350400
    assert [grid_quantile(gnb, 200, q) for q in (0.9, 0.99, 0.995)] == [306400, 441800, 480600]
    gfn = fft_aggregate("negbin", fx, 2 ** 14, **NB)[:2 ** 13]
    assert [grid_quantile(gfn, 200, q) for q in (0.9, 0.99, 0.995)] == [306400, 441800, 480600]


def test_simulation(sev):
    L, _, _ = sev
    sim = simulate(100_000, lambda r, k: r.poisson(48.5, k),
                   lambda r, k: r.lognormal(L["mu"], L["sigma"], k), seed=11)
    assert round(sim.mean()) == 196102
    assert [round(np.quantile(sim, q)) for q in (0.9, 0.99, 0.995)] == [267977, 363849, 392534]
    assert round(sample_tvar(sim, 0.99)) == 412448
    nb = simulate(100_000, lambda r, k: r.negative_binomial(10, 10 / 58.5, k),
                  lambda r, k: r.lognormal(L["mu"], L["sigma"], k), seed=11)
    assert [round(np.quantile(nb, q)) for q in (0.9, 0.99, 0.995)] == [306429, 441380, 478036]


def test_approximations(sev):
    _, (m1, m2, m3), X = sev
    ES, VS = compound_moments(48.5, 48.5, m1, m2 - m1 ** 2)
    sk = compound_poisson_skewness(48.5, m2, m3)
    TG, (a, b, x0) = translated_gamma(ES, VS, sk)
    assert (round(a, 3), round(b * 1e5, 3), round(x0)) == (3.042, 3.102, 97838)
    LN, (mu, s) = lognormal_approx(ES, VS)
    assert (round(mu, 3), round(s, 4)) == (12.146, 0.2813)
    assert [round(normal_approx(ES, VS).ppf(q)) for q in QS] == [195917, 267978, 288406, 326727, 340755]
    assert [round(TG.ppf(q)) for q in QS] == [185406, 271311, 302840, 371131, 399202]
    assert [round(LN.ppf(q)) for q in QS] == [188314, 270068, 299133, 362355, 388702]


def test_full_portfolio(sev):
    _, (m1, m2, m3), X = sev
    fx = discretize(X.cdf, 500, 2 ** 13)
    g = panjer("poisson", fx, 2 ** 13, lam=485)
    m, _ = grid_moments(g, 500)
    q = grid_quantile(g, 500, 0.995)
    assert (q, round(m)) == (2482000, 1959062)
    EF, VF = compound_moments(485, 485, m1, m2 - m1 ** 2)
    TG, _ = translated_gamma(EF, VF, compound_poisson_skewness(485, m2, m3))
    assert round(normal_approx(EF, VF).ppf(0.995)) == 2417187 and round(TG.ppf(0.995)) == 2477392
    assert round(q - m) == 522938 and round((q - m) / 5000, 2) == 104.59
