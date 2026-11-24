"""Figures of Chapter 28 (worked case)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gicase import run_case, liability, indicated_rate, reserve_review, variability
from gitriangle import synthetic_history


@pytest.fixture(scope="module")
def case():
    return run_case()


def test_case_figures(case):
    s = case["selection"]
    assert round(s["selected"].sum()) == 12948314 and round(s["unpaid"].sum()) == 2345251
    assert case["data"].attrs["failed_checks"] == 0
    assert round(case["variability"]["mack_cv"], 3) == 0.119 and round(case["variability"]["boot_cv"], 3) == 0.114
    assert round(case["variability"]["q75"]) == 2514725
    assert round(case["ulae"]) == 55863
    L = case["liability"]
    assert (round(L["pv"]), round(L["risk_margin"]), round(L["lic"])) == (2343208, 59732, 2402941)
    assert round(L["confidence"], 3) == 0.605
    r = case["reinsurance"]
    assert round(r["ceded_ultimate"] - r["ceded_paid"]) == 270000
    assert round(indicated_rate(), 2) == 789.24 and round(indicated_rate(reins_cost=52_194 / 6_522.79), 2) == 799.12


def test_exercise_discount():
    p, c, t = synthetic_history()
    rv = reserve_review(p, c, t)
    var = variability(rv["paid_triangle"], 2_345_251)
    assert round(liability(rv["paid_triangle"], 2_345_251, 55_863, var["sims"], rate=0.04)["lic"]) == 2383734
