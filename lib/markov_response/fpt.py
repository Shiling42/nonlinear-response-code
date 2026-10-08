"""markov_response.fpt

First-passage time (FPT) utilities

IMPORTANT NOTATION
------------------
Throughout markov_response we follow the paper + Shiling's note convention:
`t_{ij}` means *first passage from state j to state i* (source j → target i).
So arguments are always (i=target, j=source). built on top of :class:`markov_response.core.CTMC`.

Design goal
-----------
Provide a clean *OO-first* API via ``CTMC.fpt_*`` helpers, while keeping
verification / Monte Carlo code outside of the core package surface.

This module is intentionally lightweight: it delegates the actual moment/cumulant
computations to CTMC methods (spectral recursion based on the attached note
"Spectral Theory of First Passage Times in CTMCs").

Public API (preferred)
----------------------
- ``CTMC.fpt_moments(i, j, max_k=4)``
- ``CTMC.fpt_cumulants(i, j, max_k=4)``
- ``CTMC.fpt_stats(i, j, max_k=4)``

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .core import CTMC


@dataclass(frozen=True)
class FPTStats:
    """Convenient container for FPT moments/cumulants (up to 4th order)."""

    i: int
    j: int

    # Raw moments
    m1: float
    m2: float
    m3: float
    m4: float

    # Cumulants
    kappa1: float
    kappa2: float
    kappa3: float
    kappa4: float

    # Derived standardized stats
    variance: float
    skewness: float
    kurtosis_excess: float
    cv: float


    # Friendly aliases
    @property
    def mean(self) -> float:
        return self.m1

    @property
    def var(self) -> float:
        return self.variance

    @property
    def kurtosis(self) -> float:
        """Non-excess kurtosis = excess + 3."""
        return self.kurtosis_excess + 3.0


def fpt_moments(ctmc: CTMC, i: int, j: int, max_k: int = 4) -> Dict[int, float]:
    """Return raw moments E[t_{ij}^k] for k=1..max_k, where t_{ij} is FPT j→i."""

    if max_k < 1 or max_k > 4:
        raise ValueError("max_k must be 1..4")
    return {k: ctmc.higher_order_moment(i, j, k) for k in range(1, max_k + 1)}


def fpt_cumulants(ctmc: CTMC, i: int, j: int, max_k: int = 4) -> Dict[int, float]:
    """Return cumulants κ_k(t_{ij}) for k=1..max_k, where t_{ij} is FPT j→i."""

    if max_k < 1 or max_k > 4:
        raise ValueError("max_k must be 1..4")
    return {k: ctmc.mfpt_cumulant(i, j, k) for k in range(1, max_k + 1)}


def fpt_stats(ctmc: CTMC, i: int, j: int, max_k: int = 4) -> FPTStats:
    """Return a structured summary of moments/cumulants/stats for t_{ij} (FPT j→i)."""

    if max_k < 1 or max_k > 4:
        raise ValueError("max_k must be 1..4")

    m1 = ctmc.higher_order_moment(i, j, 1)
    m2 = ctmc.higher_order_moment(i, j, 2) if max_k >= 2 else np.nan
    m3 = ctmc.higher_order_moment(i, j, 3) if max_k >= 3 else np.nan
    m4 = ctmc.higher_order_moment(i, j, 4) if max_k >= 4 else np.nan

    k1 = ctmc.mfpt_cumulant(i, j, 1)
    k2 = ctmc.mfpt_cumulant(i, j, 2) if max_k >= 2 else np.nan
    k3 = ctmc.mfpt_cumulant(i, j, 3) if max_k >= 3 else np.nan
    k4 = ctmc.mfpt_cumulant(i, j, 4) if max_k >= 4 else np.nan

    var = float(k2) if max_k >= 2 else np.nan

    # Reuse CTMC helpers (computed from moments)
    skew = ctmc.mfpt_skewness(i, j) if max_k >= 3 else np.nan
    kurt_ex = ctmc.mfpt_kurtosis(i, j, excess=True) if max_k >= 4 else np.nan
    cv = ctmc.mfpt_cv(i, j) if max_k >= 2 else np.nan

    return FPTStats(
        i=i,
        j=j,
        m1=float(m1),
        m2=float(m2),
        m3=float(m3),
        m4=float(m4),
        kappa1=float(k1),
        kappa2=float(k2),
        kappa3=float(k3),
        kappa4=float(k4),
        variance=float(var),
        skewness=float(skew),
        kurtosis_excess=float(kurt_ex),
        cv=float(cv),
    )
