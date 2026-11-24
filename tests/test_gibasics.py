"""Unit tests for src/gibasics.py: every figure quoted in Chapter 1 is checked here."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gibasics import (  # noqa: E402
    earned_fraction,
    indicated_premium,
    months_between,
    premium_summary,
    ultimate_from_factor,
)


@pytest.fixture
def portfolio():
    df = pd.DataFrame({
        "start": ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"],
        "term_months": 12,
        "units": [300, 250, 200, 250],
    })
    df["written_premium"] = 600 * df["units"]
    return df


# --- Section 1.2 -----------------------------------------------------------

def test_months_between():
    assert months_between("2025-04-01", "2025-12-31") == 9
    assert months_between("2025-01-01", "2025-12-31") == 12


def test_earned_fraction_bounds():
    assert earned_fraction("2026-03-01", "2025-12-31") == 0.0      # not yet started
    assert earned_fraction("2024-01-01", "2025-12-31") == 1.0      # fully earned


def test_single_policy_earned_premium():
    assert 600 * earned_fraction("2025-04-01", "2025-12-31") == pytest.approx(450.0)


# --- Section 1.4 -----------------------------------------------------------

def test_portfolio_totals(portfolio):
    s = premium_summary(portfolio, "2025-12-31")
    assert s["earned_exposure"].sum() == pytest.approx(650.0)
    assert s["earned_premium"].sum() == pytest.approx(390_000)
    assert s["unearned_premium"].sum() == pytest.approx(210_000)
    assert (s["earned_premium"] + s["unearned_premium"]).sum() == pytest.approx(600_000)


def test_ultimates_and_ratios():
    ult = ultimate_from_factor(210_000, 1.25)
    count = ultimate_from_factor(52, 1.25)
    assert ult == pytest.approx(262_500)
    assert count == pytest.approx(65)
    assert count / 650 == pytest.approx(0.100)
    assert ult / count == pytest.approx(4038.46, abs=0.01)
    assert ult / 650 == pytest.approx(403.85, abs=0.01)
    assert ult / 390_000 == pytest.approx(0.673, abs=0.0005)
    assert 98_000 / 390_000 == pytest.approx(0.251, abs=0.0005)


def test_indicated_premium_and_change():
    p = indicated_premium(262_500 / 650, variable_expense=0.25, profit=0.05)
    assert p == pytest.approx(576.92, abs=0.01)
    assert p / 600 - 1 == pytest.approx(-0.038, abs=0.0005)


# --- Section 1.6 exercises -------------------------------------------------

def test_exercise1_semi_annual_policy():
    ex1 = pd.DataFrame({"start": ["2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01"],
                        "term_months": [12, 12, 6, 12], "units": 1,
                        "written_premium": [480, 720, 300, 960]})
    s = premium_summary(ex1, "2025-12-31")
    assert s["earned_premium"].tolist() == pytest.approx([440, 480, 250, 160])
    assert s["unearned_premium"].sum() == pytest.approx(1130)
    assert s["earned_exposure"].iloc[2] == pytest.approx(5 / 12)
    assert s["earned_exposure"].sum() == pytest.approx(26 / 12)


def test_exercise2_factor_140():
    ult = ultimate_from_factor(210_000, 1.40)
    assert ult - 210_000 == pytest.approx(84_000)
    assert ult / 390_000 == pytest.approx(0.754, abs=0.0005)
    p = indicated_premium(ult / 650, 0.25, 0.05)
    assert p == pytest.approx(646.15, abs=0.01)
    assert p / 600 - 1 == pytest.approx(0.077, abs=0.0005)


def test_exercise3_rate_cut():
    assert (262_500 / 650) / 480 == pytest.approx(0.841, abs=0.0005)
    assert 164_500 / 262_500 == pytest.approx(0.63, abs=0.005)
