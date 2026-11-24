"""Unit tests for src/gitriangle.py: every figure quoted in Chapter 3 is checked here."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gitriangle import (  # noqa: E402
    calendar_diagonal,
    development_age,
    exposure_by_accident_year,
    synthetic_history,
    to_cumulative,
    to_incremental,
    to_long,
    triangle,
)


@pytest.fixture
def six():
    claims = pd.DataFrame({
        "claim_id": ["K1", "K2", "K3", "K4", "K5", "K6"],
        "accident_date": ["2023-03-10", "2023-11-20", "2023-07-01", "2024-02-14", "2024-12-20", "2025-05-05"],
        "report_date": ["2023-03-15", "2024-01-10", "2023-07-05", "2024-02-20", "2025-01-08", "2025-05-06"],
    })
    rows = [("K1", "2023-03-15", "reserve", 5000), ("K1", "2023-09-01", "payment", 2000),
            ("K1", "2023-09-01", "reserve", 3500), ("K1", "2024-05-01", "payment", 3800),
            ("K1", "2024-05-01", "reserve", 0),
            ("K2", "2024-01-10", "reserve", 4000), ("K2", "2024-06-30", "payment", 1500),
            ("K2", "2024-06-30", "reserve", 3000), ("K2", "2025-04-01", "payment", 3000),
            ("K2", "2025-04-01", "reserve", 0),
            ("K3", "2023-07-05", "reserve", 1000), ("K3", "2023-08-01", "payment", 900),
            ("K3", "2023-08-01", "reserve", 0),
            ("K4", "2024-02-20", "reserve", 6000), ("K4", "2024-10-01", "payment", 2500),
            ("K4", "2024-10-01", "reserve", 4000), ("K4", "2025-06-01", "reserve", 4500),
            ("K5", "2025-01-08", "reserve", 2000), ("K5", "2025-03-01", "payment", 2200),
            ("K5", "2025-03-01", "reserve", 0),
            ("K6", "2025-05-06", "reserve", 3000), ("K6", "2025-11-15", "payment", 1000),
            ("K6", "2025-11-15", "reserve", 2500)]
    return claims, pd.DataFrame(rows, columns=["claim_id", "date", "type", "amount"])


@pytest.fixture(scope="module")
def hist():
    return synthetic_history(2020, 2025, seed=100)


def rows(tri):
    return [[v for v in r if pd.notna(v)] for r in tri.values.tolist()]


# --- Section 3.1 -----------------------------------------------------------

def test_development_age():
    assert development_age(2023, "2025-12-31") == 36
    assert development_age(2025, "2025-06-30") == 6


# --- Section 3.2-3.3: six-claim example --------------------------------------

def test_six_claim_triangles(six):
    c, t = six
    assert rows(triangle(c, t, "paid", 2023, 2025)) == [[2900, 8200, 11200], [2500, 4700], [1000]]
    assert rows(triangle(c, t, "incurred", 2023, 2025)) == [[6400, 11200, 11200], [6500, 9200], [3500]]
    assert rows(triangle(c, t, "case_reserve", 2023, 2025)) == [[3500, 3000, 0], [4000, 4500], [2500]]
    assert rows(triangle(c, t, "reported_count", 2023, 2025)) == [[2, 3, 3], [1, 2], [1]]
    assert rows(triangle(c, t, "closed_count", 2023, 2025)) == [[1, 2, 3], [0, 1], [0]]
    assert rows(triangle(c, t, "open_count", 2023, 2025)) == [[1, 1, 0], [1, 1], [1]]


def test_incremental_and_diagonals(six):
    c, t = six
    P = triangle(c, t, "paid", 2023, 2025)
    X = to_incremental(P)
    assert rows(X) == [[2900, 5300, 3000], [2500, 2200], [1000]]
    assert to_cumulative(X).equals(P)
    assert calendar_diagonal(X, 2025).sum() == 6200
    assert calendar_diagonal(X, 2024).sum() == 7800
    assert P.loc[2023, 24] / P.loc[2023, 12] == pytest.approx(2.83, abs=0.005)
    assert P.loc[2024, 24] / P.loc[2024, 12] == pytest.approx(1.88, abs=0.005)


def test_bad_measure(six):
    with pytest.raises(ValueError):
        triangle(*six, "ultimate", 2023, 2025)


# --- Section 3.4: synthetic portfolio --------------------------------------

def test_history_sizes(hist):
    pol, cl, txs = hist
    assert (len(pol), len(cl), len(txs)) == (34009, 3506, 21298)


def test_paid_incurred(hist):
    pol, cl, txs = hist
    P = triangle(cl, txs, "paid", 2021, 2025).round(0)
    I = triangle(cl, txs, "incurred", 2021, 2025).round(0)
    assert P.loc[2021].tolist() == [1032119, 1932401, 2254032, 2308768, 2324983]
    assert P[12].tolist() == [1032119, 924838, 1256832, 1343872, 1445759]
    assert I.loc[2021].tolist() == [2081859, 2260779, 2332841, 2320837, 2325132]
    assert I[12].tolist() == [2081859, 1818229, 2160241, 2558368, 2838741]
    assert I.loc[2024, 24] == 2908537


def test_diagnostics(hist):
    pol, cl, txs = hist
    P = triangle(cl, txs, "paid", 2021, 2025)
    I = triangle(cl, txs, "incurred", 2021, 2025)
    R = triangle(cl, txs, "reported_count", 2021, 2025)
    C = triangle(cl, txs, "closed_count", 2021, 2025)
    O = triangle(cl, txs, "open_count", 2021, 2025)
    CR = triangle(cl, txs, "case_reserve", 2021, 2025)
    assert (P / I)[12].round(3).tolist() == [0.496, 0.509, 0.582, 0.525, 0.509]
    assert (C / R)[12].round(3).tolist() == [0.449, 0.469, 0.469, 0.468, 0.460]
    assert (CR / O)[12].round(0).tolist() == [3932, 3285, 3238, 4217, 4353]
    assert R[12].tolist() == [485, 512, 525, 541, 593]
    assert R.loc[2021].tolist() == [485, 524, 524, 524, 524]
    assert O.loc[2021].tolist() == [267, 68, 17, 4, 1]


def test_exposure(hist):
    pol, cl, txs = hist
    E = exposure_by_accident_year(pol, range(2021, 2026))
    assert E["earned_exposure"].round(2).tolist() == [5120.07, 5321.55, 5656.43, 5982.39, 6212.18]
    R = triangle(cl, txs, "reported_count", 2021, 2025)
    assert (R[12] / E["earned_exposure"]).round(4).tolist() == [0.0947, 0.0962, 0.0928, 0.0904, 0.0955]


def test_calendar_2025_matches_transactions(hist):
    pol, cl, txs = hist
    X = to_incremental(triangle(cl, txs, "paid", 2021, 2025))
    diag = calendar_diagonal(X, 2025).sum()
    assert diag == pytest.approx(2883450, abs=1)
    ay = pd.to_datetime(cl.set_index("claim_id")["accident_date"]).dt.year
    pay = txs[(txs["type"] == "payment") & (pd.to_datetime(txs["date"]).dt.year == 2025)]
    pay = pay[ay.loc[pay["claim_id"]].between(2021, 2025).to_numpy()]
    assert pay["amount"].sum() == pytest.approx(diag)


def test_to_long(hist):
    pol, cl, txs = hist
    long = to_long(triangle(cl, txs, "paid", 2021, 2025), "paid")
    assert len(long) == 15
    assert long["paid"].head(3).round(2).tolist() == [1032119.41, 1932400.71, 2254031.80]
    assert long["valuation"].iloc[2] == pd.Timestamp("2023-12-31")


def test_chainladder_roundtrip(hist):
    chl = pytest.importorskip("chainladder")
    import numpy as np
    pol, cl, txs = hist
    P = triangle(cl, txs, "paid", 2021, 2025)
    tri = chl.Triangle(to_long(P, "paid"), origin="accident_year", development="valuation",
                       columns=["paid"], cumulative=True)
    frame = tri.to_frame()
    assert list(frame.columns) == [12, 24, 36, 48, 60]
    assert np.allclose(frame.values, P.values, equal_nan=True)
