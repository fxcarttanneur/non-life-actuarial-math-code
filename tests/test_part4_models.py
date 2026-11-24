"""Unit tests for src/giglm.py, src/giilf.py, src/gicred.py and src/giexperience.py
(figures quoted in Chapters 16-19)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gicoverage import inflate, ler  # noqa: E402
from gicred import buhlmann, buhlmann_straub, full_credibility_claims, square_root_z  # noqa: E402
from gidata import synthetic_portfolio  # noqa: E402
from giexperience import (efficiency, experience_mod, mean_relativity, retro_premium,  # noqa: E402
                          simulate_bm, simulate_fleet_losses, split_plan_mod, stationary,
                          bm_transition)
from giglm import (design, double_lift, gini, glm_fit, lift_table, lr_test,  # noqa: E402
                   policy_data, relativities)
from giilf import deductible_relativity, ilf_consistent, ilf_table, layer_premium  # noqa: E402
from gilossmodels import fit_mle  # noqa: E402
from gipricing import policy_year_experience  # noqa: E402
from gitriangle import synthetic_history  # noqa: E402

B = {"region": "South", "vehicle": "A", "color": "black"}
REL = [0.60, 0.70, 0.80, 1.00, 1.20, 1.50]


@pytest.fixture(scope="module")
def hist():
    P, C, _ = synthetic_history()
    return P, C


@pytest.fixture(scope="module")
def glmdata(hist):
    return policy_data(*hist, [2023, 2024, 2025])


@pytest.fixture(scope="module")
def lognormal():
    _, c = synthetic_portfolio()
    return fit_mle(c["ultimate_amount"].to_numpy(), "lognormal")


def r4(s):
    return [round(float(v), 4) for v in s]


# ------------------------------------------------------------ Chapter 16

def test_glm_frequency(glmdata):
    pol, cl = glmdata
    assert (len(pol), len(cl), round(cl.amount.mean(), 2)) == (18247, 1872, 5498.96)
    f = glm_fit(design(pol, ["region", "vehicle"], B), pol.n_claims, "poisson", offset=np.log(pol.exposure))
    assert r4(f["coef"]) == [-2.3846, 0.1553, -0.1305, 0.1024, 0.3620]
    assert r4(f["se"]) == [0.0466, 0.0575, 0.0552, 0.0547, 0.0573]
    assert [round(v, 2) for v in f["coef"][1:] / f["se"][1:]] == [2.70, -2.37, 1.87, 6.32]
    assert [round(v, 3) for v in relativities(f)] == [0.092, 1.168, 0.878, 1.108, 1.436]
    assert round(np.exp(0.1553 - 1.96 * 0.0575), 2) == 1.04 and round(np.exp(0.1553 + 1.96 * 0.0575), 2) == 1.31
    f2 = glm_fit(design(pol, ["region", "vehicle", "color"], B), pol.n_claims, "poisson")
    st, df, p = lr_test(f, f2)
    assert (round(st, 2), df, round(p, 2)) == (3.57, 3, 0.31)
    assert [round(v, 3) for v in relativities(f2)[-3:]] == [1.032, 1.111, 0.999]
    r = relativities(f)
    assert round(r.iloc[0] * r["region:City"] * r["vehicle:C"] * cl.amount.mean(), 2) == 849.76
    assert round(r.iloc[0] * r["region:North"] * cl.amount.mean(), 2) == 444.58
    assert [round(np.exp(0.3620 + s * 1.96 * 0.0573), 3) for s in (-1, 1)] == [1.284, 1.607]
    assert [round(np.exp(0.1024 + s * 1.96 * 0.0547), 3) for s in (-1, 1)] == [0.995, 1.233]


def test_glm_cells(glmdata):
    pol, _ = glmdata
    c = pol.groupby(["region", "vehicle"]).agg(exposure=("exposure", "sum"), n=("n_claims", "sum")).reset_index()
    f = glm_fit(design(c, ["region", "vehicle"], B), c.n, "poisson", offset=np.log(c.exposure))
    assert [round(v, 3) for v in relativities(f)] == [0.092, 1.168, 0.878, 1.108, 1.436]
    c["fit"] = f["mu"]
    assert c.groupby("region").fit.sum().round(6).tolist() == [558, 652, 662]
    assert c.groupby("vehicle").fit.sum().round(6).tolist() == [831, 559, 482]
    assert (round(f["deviance"], 2), f["df_resid"]) == (5.56, 4)
    assert round(c.fit[2] / c.exposure[2], 4) == 0.1545 and round(c.fit[4] / c.exposure[4], 4) == 0.0896


def test_glm_severity_tweedie(glmdata):
    pol, cl = glmdata
    s = glm_fit(design(cl, ["region", "vehicle"], B), cl.amount, "gamma")
    assert [round(v, 3) for v in relativities(s)[1:]] == [1.072, 0.995, 1.062, 0.926]
    assert round(s["phi"], 2) == 2.55
    s0 = glm_fit(design(cl, [], B), cl.amount, "gamma")
    st, df, p = lr_test(s0, s, s["phi"])
    assert (round(st, 2), df, round(p, 2)) == (2.78, 4, 0.60)
    t = glm_fit(design(pol, ["region", "vehicle"], B), pol.losses, "tweedie", p=1.5)
    assert [round(v, 3) for v in relativities(t)[1:]] == [1.251, 0.862, 1.173, 1.338]


def test_glm_validation(glmdata):
    pol, _ = glmdata
    tr, te = pol[pol.policy_year < 2025], pol[pol.policy_year == 2025]
    assert (len(te), int(te.n_claims.sum())) == (6381, 665)
    fa = glm_fit(design(tr, ["region", "vehicle"], B), tr.n_claims, "poisson")
    fb = glm_fit(design(tr, ["region"], B), tr.n_claims, "poisson")
    fc = glm_fit(design(tr, ["region", "vehicle", "color"], B), tr.n_claims, "poisson")
    assert [round(v, 3) for v in relativities(fa)[1:]] == [1.117, 0.829, 1.167, 1.558]

    def pred(f, fs):
        return np.exp(design(te, fs, B)[f["coef"].index] @ f["coef"])
    pa, pb, pc = pred(fa, ["region", "vehicle"]), pred(fb, ["region"]), pred(fc, ["region", "vehicle", "color"])
    lt = lift_table(pa, te.n_claims, te.exposure)
    assert r4(lt.predicted) == [0.0755, 0.0892, 0.0967, 0.1105, 0.1379]
    assert r4(lt.actual) == [0.0980, 0.0823, 0.1003, 0.1160, 0.1245]
    assert [round(gini(p, te.n_claims, te.exposure), 3) for p in (pa, pb, pc)] == [0.071, 0.055, 0.071]
    dl = double_lift(pa, pb, te.n_claims, te.exposure)
    assert dl.ratio_a_b.round(3).tolist() == [0.859, 0.861, 0.934, 1.012, 1.341]
    assert dl.actual_over_a.round(3).tolist() == [1.201, 1.139, 1.002, 0.943, 0.904]
    assert dl.actual_over_b.round(3).tolist() == [1.031, 0.981, 0.936, 0.954, 1.212]


# ------------------------------------------------------------ Chapter 17

def test_ilf_pareto():
    BI = {"alpha": 2, "theta": 20_000}
    t = ilf_table("pareto", BI, [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], 25_000)
    assert t["lev"].round(2).tolist() == [6666.67, 11111.11, 14285.71, 16666.67, 18518.52, 19230.77, 19607.84]
    assert t["ilf"].round(4).tolist() == [0.6, 1.0, 1.2857, 1.5, 1.6667, 1.7308, 1.7647]
    assert t["marginal_per_1000"].round(4).tolist() == [0.06, 0.0267, 0.0114, 0.0043, 0.0011, 0.0003, 0.0001]
    ilf = t["ilf"].to_dict()
    lay = [layer_premium(350, ilf, a, b) for a, b in ((25_000, 100_000), (100_000, 250_000), (250_000, 1_000_000))]
    assert [round(v, 2) for v in lay] == [175.00, 58.33, 34.31] and round(350 * ilf[1_000_000], 2) == 617.65
    ti = ilf_table("pareto", inflate("pareto", BI, 0.10), [25_000, 100_000, 1_000_000], 25_000)
    assert ti["lev"].round(2).tolist() == [11702.13, 18032.79, 21526.42]
    assert ti["ilf"].round(4).tolist() == [1.0, 1.5410, 1.8395]


def test_ilf_consistency_and_deductibles(lognormal):
    bad = ilf_consistent(pd.Series([0.80, 1.00, 1.20, 1.65, 1.80], index=[10_000, 25_000, 50_000, 100_000, 250_000]))
    assert bad.increase.all() and bad.per_unit_falls.tolist() == [True, True, True, False, True]
    assert ilf_consistent(pd.Series([0.80, 1.00, 1.20, 1.60, 1.80],
                                    index=[10_000, 25_000, 50_000, 100_000, 250_000])).per_unit_falls.all()
    t = ilf_table("lognormal", lognormal, [10_000, 25_000, 50_000, 100_000, 1_000_000], 25_000)
    assert t["ilf"].round(4).tolist() == [0.8449, 1.0, 1.05, 1.0673, 1.0725]
    ds = (250, 500, 1000, 2500)
    assert [round(ler("lognormal", lognormal, d), 4) for d in ds] == [0.0611, 0.1183, 0.2175, 0.4246]
    assert [round(deductible_relativity("lognormal", lognormal, d), 4) for d in ds] == [0.9389, 0.8817, 0.7825, 0.5754]
    assert [round(deductible_relativity("lognormal", lognormal, d, pure_premium=609.28, fixed=30), 4)
            for d in ds] == [0.9417, 0.8873, 0.7928, 0.5953]
    assert round(0.7825 / 0.8817, 4) == 0.8875
    assert round(800 * (1 - 0.5754 / 0.8817), 1) == 277.9 and round(800 * (1 - 0.5953 / 0.8873), 1) == 263.3


# ------------------------------------------------------------ Chapter 18

def test_limited_fluctuation():
    nf = full_credibility_claims(cv_severity=1.34)
    assert round(full_credibility_claims()) == 1082 and round(nf) == 3025
    assert round(full_credibility_claims(0.95, 0.05)) == 1537 and round(full_credibility_claims(0.90, 0.10, 1.34)) == 756
    z1, z2 = square_root_z(153, nf), square_root_z(178, nf)
    assert (round(z1, 4), round(z2, 4)) == (0.2249, 0.2426)
    assert round(z1 * 684.1180 + (1 - z1) * 743.22, 2) == 729.93
    assert round(z2 * 381.8289 + (1 - z2) * 451.96, 2) == 434.95
    assert round(square_root_z(1872, nf), 2) == 0.79 and round(square_root_z(665, 1082.2174), 3) == 0.784


def test_buhlmann():
    b = buhlmann(pd.DataFrame([[3, 5, 4], [7, 5, 6], [2, 3, 4]], index=list("ABC")))
    assert (b["epv"], round(b["vhm"], 6), round(b["k"], 6), round(b["z"], 3)) == (1.0, 2.0, 0.5, 0.857)
    assert b["premium"].round(3).tolist() == [4.048, 5.762, 3.190]
    assert round(buhlmann(pd.DataFrame([[4, 6, 5], [5, 4, 6], [6, 5, 4]]))["vhm"], 3) == -0.333


def test_buhlmann_straub(hist):
    e = policy_year_experience(*hist, by=["region", "vehicle"]).reset_index()
    e["pp"] = e.losses * 1.04 ** (2028 - (e.policy_year + 1)) / e.exposure
    e["cell"] = e.region + " " + e.vehicle
    X = e.pivot(index="cell", columns="policy_year", values="pp")
    M = e.pivot(index="cell", columns="policy_year", values="exposure")
    bs = buhlmann_straub(X, M)
    assert (round(bs["sigma2"], -2), round(bs["a"]), round(bs["k"]), round(bs["mu"], 2)) == (11909700, 17185, 693, 600.71)
    assert bs["m_i"].astype(int).tolist() == [4266, 2621, 1748, 6733, 4022, 2660, 6049, 3581, 2329]
    assert bs["xbar_i"].round(2).tolist() == [703.72, 765.65, 848.93, 442.29, 415.73, 633.19, 462.77, 599.61, 605.31]
    assert bs["z"].round(3).tolist() == [0.860, 0.791, 0.716, 0.907, 0.853, 0.793, 0.897, 0.838, 0.771]
    assert bs["premium"].round(2).tolist() == [689.32, 731.16, 778.46, 457.08, 442.91, 626.48, 476.95, 599.79, 604.26]
    sev = np.exp(7.6 + 0.72) * 1.04 ** 8
    rel = {"City": 1.3, "North": 0.9, "South": 1.0}
    vr = {"A": 0.9, "B": 1.0, "C": 1.25}
    true = pd.Series({c: 0.1 * rel[c.split()[0]] * vr[c.split()[1]] * sev for c in X.index})
    assert round(np.sqrt(bs["a"])) == 131 and round(true.std(ddof=0)) == 134
    rmse = lambda s: round(float(np.sqrt(((s - true) ** 2).mean())))  # noqa: E731
    assert (rmse(bs["premium"]), rmse(bs["xbar_i"]), rmse(bs["xbar"] + 0 * true)) == (62, 56, 149)


# ------------------------------------------------------------ Chapter 19

def test_experience_rating():
    assert experience_mod(35_000, 50_000, 0.25) == pytest.approx(0.925)
    losses = [1200, 3500, 8000, 22000, 600, 4000]
    s = split_plan_mod(losses, 5000, 27_000, 18_000, 0.3, 20_000)
    assert (s["A_p"], s["A_e"], round(s["mod"], 4)) == (19300, 20000, 0.8908)
    assert round(experience_mod(39_300, 45_000, 45_000 / 65_000), 4) == 0.9123
    assert round(split_plan_mod(losses, 5000, 27_000, 18_000, 0.3, 5_000)["mod"], 4) == 0.858
    assert round(split_plan_mod(losses, 5000, 27_000, 18_000, 0.3, 60_000)["mod"], 4) == 0.9324


def test_retro(lognormal):
    assert retro_premium([40_000, 90_000, 150_000], 26_000, 1.10, 1.04, 78_000, 182_000).round(2).tolist() == \
        [78_000, 130_000, 182_000]
    L = simulate_fleet_losses(100_000, 25, lognormal["mu"], lognormal["sigma"])
    R = retro_premium(L, 26_000, 1.10, 1.04, 78_000, 182_000)
    assert (round(L.mean()), round(R.mean()), round((R == 78_000).mean(), 3), round((R == 182_000).mean(), 3)) == \
        (100968, 136512, 0.031, 0.164)
    b = optimize.brentq(lambda b: retro_premium(L, b, 1.10, 1.04, 78_000, 182_000).mean() - 130_000, -50_000, 50_000)
    Rb = retro_premium(L, b, 1.10, 1.04, 78_000, 182_000)
    assert (round(b), round((Rb == 182_000).mean(), 3), round((Rb == 78_000).mean(), 3)) == (18232, 0.133, 0.063)
    assert round((175_000 - 26_000) / 1.10) == 135455


def test_bonus_malus():
    T = bm_transition(0.10, 6, 2)
    assert round(T[3, 2], 4) == 0.9048 and round(T[3, 5], 4) == 0.0952
    pis = {lam: stationary(bm_transition(lam, 6, 2)) for lam in (0.05, 0.10, 0.15, 0.20)}
    assert [round(v, 4) for v in pis[0.10]] == [0.7829, 0.0823, 0.0910, 0.0223, 0.0164, 0.0051]
    assert [round(v, 4) for v in pis[0.05]] == [0.8953, 0.0459, 0.0483, 0.0060, 0.0040, 0.0006]
    assert [round(mean_relativity(lam, REL, 2), 3) for lam in pis] == [0.620, 0.650, 0.691, 0.742]
    assert [round(efficiency(lam, REL, 2), 2) for lam in pis] == [0.04, 0.11, 0.20, 0.30]
    s = simulate_bm(0.10, 6, 2, 3, 20, 20_000)
    assert round(s.loc[20, 0], 3) == 0.788 and round(s.loc[5, 0], 3) == 0.602
    p3 = stationary(bm_transition(0.10, 6, 3))
    assert [round(v, 4) for v in p3] == [0.7004, 0.0737, 0.0814, 0.0900, 0.0294, 0.0251]
    assert (round(mean_relativity(0.10, REL, 3), 3), round(efficiency(0.10, REL, 3), 3)) == (0.700, 0.175)
