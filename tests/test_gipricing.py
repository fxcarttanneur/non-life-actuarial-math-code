"""Unit tests for src/gipricing.py: figures quoted in Chapters 13-15."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gipricing import (add_ratios, average_accident_date, claims_made_step_factors,  # noqa: E402
                       earned_share_before, fit_trend, loss_ratio_change, one_way,
                       parallelogram, policy_year_experience, pure_premium_rate, trend_factor)
from gitriangle import synthetic_history  # noqa: E402


@pytest.fixture(scope="module")
def hist():
    P, C, _ = synthetic_history()
    return P, C, policy_year_experience(P, C)


@pytest.fixture(scope="module")
def cells(hist):
    P, C, _ = hist
    e = policy_year_experience(P, C, by=["region", "vehicle"]).loc[2023:2025]
    return add_ratios(e.groupby(["region", "vehicle"])[["exposure", "claims", "losses"]].sum())


def test_policy_years(hist):
    py = hist[2]
    assert py.exposure.tolist() == [5000, 5250, 5512, 5788, 6078, 6381]
    assert py.claims.astype(int).tolist() == [520, 554, 560, 569, 638, 665]
    assert py.losses.round().tolist() == [2258559, 2366584, 2402247, 2584580, 3013034, 3588385]
    assert py.frequency.round(4).tolist() == [0.1040, 0.1055, 0.1016, 0.0983, 0.1050, 0.1042]
    assert py.severity.round(2).tolist() == [4343.38, 4271.81, 4289.73, 4542.32, 4722.62, 5396.07]
    assert py.pure_premium.round(2).tolist() == [451.71, 450.78, 435.82, 446.54, 495.73, 562.35]


def test_ch13_classes():
    exp, n, L = np.array([800, 2400, 1000]), np.array([120, 168, 90]), np.array([540000, 638400, 360000])
    assert (L / exp).tolist() == [675, 266, 360]
    assert round(L.sum() / n.sum(), 2) == 4069.84 and round(L.sum() / exp.sum(), 2) == 366.29
    assert round(675 / 266, 3) == 2.538 and round(360 / 266, 3) == 1.353
    assert round(998400 / 3400, 2) == 293.65 and round(675 / 293.65, 3) == 2.299


def test_ch13_cells(cells):
    tot = cells[["exposure", "claims", "losses"]].sum()
    assert (tot.exposure, tot.claims, round(tot.losses)) == (18247, 1872, 9186000)
    assert round(tot.claims / tot.exposure, 4) == 0.1026 and round(tot.losses / tot.claims, 2) == 4907.05
    assert round(tot.losses / tot.exposure, 2) == 503.43
    assert cells.claims.min() == 153 and cells.claims.max() == 304
    assert cells.pure_premium.round(2).tolist() == [577.77, 662.54, 684.12, 390.28, 381.83, 602.61,
                                                    427.31, 595.52, 529.62]
    r = one_way(cells.reset_index(), "region", "South")
    v = one_way(cells.reset_index(), "vehicle", "A")
    assert r.relativity.round(3).tolist() == [1.257, 0.864, 1.000]
    assert v.relativity.round(3).tolist() == [1.000, 1.177, 1.330]
    assert r.pure_premium.round(2).tolist() == [625.80, 429.97, 497.81]
    assert round(r.frequency["City"], 4) == 0.1208 and round(r.severity["City"], 2) == 5181.39
    d = cells.reset_index()
    d["rel"] = d.region.map(r.relativity) * d.vehicle.map(v.relativity)
    off = (d.exposure * d.rel).sum() / d.exposure.sum()
    base = tot.losses / (d.exposure * d.rel).sum()
    assert (round(off, 4), round(base, 2)) == (1.1324, 444.58)
    ind = (base * d.rel).round(2).tolist()
    assert ind[2] == 743.22 and ind[4] == 451.96


def test_ch14_trend(hist):
    py = hist[2]
    assert [round(fit_trend(py[c])[0], 4) for c in ("frequency", "severity", "pure_premium")] == \
        [-0.0011, 0.0421, 0.0410]
    assert [round(fit_trend(py.loc[2020:2024, c])[0], 4) for c in ("frequency", "severity", "pure_premium")] == \
        [-0.0052, 0.0231, 0.0178]
    ap = (py.premium / py.exposure).round(2)
    assert (ap[2020], ap[2025], round(fit_trend(py.premium / py.exposure)[0], 4)) == (620.77, 623.52, 0.0006)
    e = py.loc[2023:2025]
    f = [trend_factor(0.04, average_accident_date(2027) - average_accident_date(y)) for y in e.index]
    assert [round(x, 4) for x in f] == [1.1699, 1.1249, 1.0816]
    tl = e.losses * f
    assert tl.round().tolist() == [3023594, 3389254, 3881198] and round(tl.sum()) == 10294045
    assert (tl / e.exposure).round(2).tolist() == [522.39, 557.63, 608.24]
    assert round(tl.sum() / e.exposure.sum(), 2) == 564.15
    for g, v in ((0.0231, 537.87), (0.0421, 567.48)):
        assert round((e.losses * (1 + g) ** (2028 - (e.index + 1))).sum() / e.exposure.sum(), 2) == v
    assert round(1.04 ** 3.5, 4) == 1.1471 and average_accident_date(2027.5) == 2028.5


def test_ch14_parallelogram():
    assert earned_share_before(2023.5, 2023) == pytest.approx(0.875)
    assert earned_share_before(2023.5, 2024) == pytest.approx(0.125)
    p = parallelogram({2023.5: 0.05, 2025.25: 0.03}, range(2022, 2027))
    assert p.average_level.round(4).tolist() == [1.0000, 1.0062, 1.0438, 1.0589, 1.0805]
    assert p.on_level_factor.round(4).tolist() == [1.0815, 1.0748, 1.0362, 1.0214, 1.0009]
    q = parallelogram({2024.0: 0.06}, [2023, 2024, 2025], term=0.5)
    assert q.average_level.round(4).tolist() == [1.0, 1.045, 1.06]
    assert q.on_level_factor.round(4).tolist() == [1.06, 1.0144, 1.0]


def test_ch15_rates():
    assert round(pure_premium_rate(440, 50, 0.20, 0.05), 2) == 653.33
    assert round(loss_ratio_change(440 / 600, 50 / 600, 0.20, 0.05), 4) == 0.0889
    assert round(pure_premium_rate(440, 90, 0.05, 0.05), 2) == 588.89
    pp = 564.1499927571194 * 1.08
    assert round(pp, 2) == 609.28 and round(pure_premium_rate(pp, 30, 0.15, 0.04), 2) == 789.24
    avg = 11353458 / 18247
    assert round(avg, 2) == 622.21
    lr = 10294044.917839158 * 1.08 / 11353458
    assert (round(lr, 4), round(30 / avg, 4)) == (0.9792, 0.0482)
    assert round(loss_ratio_change(lr, 30 / avg, 0.15, 0.04), 4) == 0.2684
    olf = parallelogram({2023.5: 0.05, 2025.25: 0.03}, [2024, 2025]).on_level_factor
    olp = 1_000_000 * olf[2024] + 1_100_000 * olf[2025]
    assert round(olp) == 2159688
    assert round(loss_ratio_change(1_490_000 / olp, 0.07, 0.18, 0.05), 4) == -0.0131
    assert round(loss_ratio_change(1_490_000 / 2_100_000, 0.07, 0.18, 0.05), 4) == 0.0124
    assert round(0.10 * 0.5 - 0.02 * 0.9, 3) == 0.032


def test_ch15_claims_made():
    s = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    assert claims_made_step_factors(s).round(2).tolist() == [0.40, 0.65, 0.80, 0.90, 0.96, 1.00]
    assert round((s * np.arange(6)).sum(), 2) == 1.29 and round((s * 1.04 ** np.arange(6)).sum(), 4) == 1.0536


def test_ch15_base_rate():
    assert round(789.2370273798631 / 1.132364447715739, 2) == 696.98
