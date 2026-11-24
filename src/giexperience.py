"""Chapter 19: experience rating, retrospective rating and bonus-malus systems."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------- experience rating

def experience_mod(actual: float, expected: float, z: float) -> float:
    """Credibility-weighted modification: Z (A / E) + (1 - Z)."""
    return z * actual / expected + (1 - z)


def split_plan_mod(losses, split: float, exp_primary: float, exp_excess: float,
                   w: float, ballast: float) -> dict:
    """Split-rating plan in the NCCI style.

    Each loss is split into a primary part min(x, split) and an excess part. The mod is
    (A_p + w A_e + (1 - w) E_e + B) / (E_p + E_e + B).
    """
    x = np.asarray(losses, float)
    ap = float(np.minimum(x, split).sum())
    ae = float(np.maximum(x - split, 0).sum())
    num = ap + w * ae + (1 - w) * exp_excess + ballast
    den = exp_primary + exp_excess + ballast
    return {"A_p": ap, "A_e": ae, "mod": num / den}


# ------------------------------------------------------ retrospective rating

def retro_premium(losses, basic: float, lcf: float, tax: float, minimum: float, maximum: float):
    """R = (b + c L) T, bounded by the minimum and maximum premiums."""
    raw = (basic + lcf * np.asarray(losses, float)) * tax
    return np.clip(raw, minimum, maximum)


def simulate_fleet_losses(n_sims: int, lam: float, mu: float, sigma: float, seed: int = 19) -> np.ndarray:
    """Annual aggregate losses of a fleet: Poisson count, lognormal severities."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(lam, n_sims)
    x = rng.lognormal(mu, sigma, int(n.sum()))
    return np.bincount(np.repeat(np.arange(n_sims), n), weights=x, minlength=n_sims)


# ----------------------------------------------------------- bonus-malus

def bm_transition(lam: float, n_classes: int, up_per_claim: int, down: int = 1,
                  max_claims: int = 20) -> np.ndarray:
    """Transition matrix of a bonus-malus scale with Poisson(lam) claims per year.

    Class 0 is the best. A claim-free year moves `down` classes toward 0; each claim moves
    `up_per_claim` classes up; the scale is bounded at 0 and n_classes - 1.
    """
    from scipy import stats
    p = stats.poisson.pmf(np.arange(max_claims + 1), lam)
    p[-1] += 1 - p.sum()
    T = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        T[i, max(i - down, 0)] += p[0]
        for k in range(1, max_claims + 1):
            T[i, min(i + k * up_per_claim, n_classes - 1)] += p[k]
    return T


def stationary(T: np.ndarray) -> np.ndarray:
    """Stationary distribution pi with pi T = pi and sum(pi) = 1."""
    n = T.shape[0]
    A = np.vstack([T.T - np.eye(n), np.ones(n)])
    b = np.concatenate([np.zeros(n), [1.0]])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def mean_relativity(lam: float, relativities, up_per_claim: int, down: int = 1) -> float:
    pi = stationary(bm_transition(lam, len(relativities), up_per_claim, down))
    return float(pi @ np.asarray(relativities, float))


def efficiency(lam: float, relativities, up_per_claim: int, down: int = 1, h: float = 1e-4) -> float:
    """Loimaranta efficiency: d ln(mean relativity) / d ln(lam), by central difference."""
    up = mean_relativity(lam * np.exp(h), relativities, up_per_claim, down)
    lo = mean_relativity(lam * np.exp(-h), relativities, up_per_claim, down)
    return float((np.log(up) - np.log(lo)) / (2 * h))


def simulate_bm(lam: float, n_classes: int, up_per_claim: int, start: int, years: int,
                n_drivers: int, seed: int = 19, down: int = 1) -> pd.DataFrame:
    """Share of drivers in each class after each year, starting in class `start`."""
    rng = np.random.default_rng(seed)
    c = np.full(n_drivers, start)
    rows = []
    for _ in range(years):
        k = rng.poisson(lam, n_drivers)
        c = np.where(k == 0, np.maximum(c - down, 0), np.minimum(c + k * up_per_claim, n_classes - 1))
        rows.append(np.bincount(c, minlength=n_classes) / n_drivers)
    return pd.DataFrame(rows, index=range(1, years + 1))
