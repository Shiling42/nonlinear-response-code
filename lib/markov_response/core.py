"""
core.py - Continuous-Time Markov Chain Fundamentals

Implements the CTMC class with full spectral analysis, mean first-passage times,
and steady-state observables.

Architecture:
    Layer 1: Rate perturbations  - perturb_rate(), perturb_rates()
    Layer 2: Parametrized perturbations - perturb_activation_energy/barrier/energy/force()

Key equations:
    - Steady-state: W pi = 0
    - MFPT: <t_ij> = sum_k (A^k_ij - A^k_ii) / (lam_k pi_i)  for i != j
    - Flux: phi_ij = W_ij pi_j
    - Current: j_ij = phi_ij - phi_ji
    - Activity: a_ij = phi_ij + phi_ji

Notation:
    t_{ij} means first passage from j -> i (source j, target i).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig, expm
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# CTMC Class
# =============================================================================

class CTMC:
    """Continuous-Time Markov Chain with full spectral decomposition.

    Provides computation of:
    - Steady-state distribution pi
    - Mean first-passage times (MFPTs) and higher-order FPT moments
    - FPT cumulants (up to 4th order)
    - Fluxes, currents, and activities
    - Observable statistics
    - Propagator P(t)
    - Rate perturbations (fundamental operation)

    Parameters
    ----------
    W : array_like
        Transition rate matrix. Off-diagonal W[i,j] is rate from j to i.
        Diagonal elements will be recomputed as -sum_j W[j,i].

    Examples
    --------
    >>> W = np.array([[0, 2, 0], [1, 0, 3], [0, 1, 0]])
    >>> ctmc = CTMC(W)
    >>> ctmc.steady_state
    array([0.4, 0.2, 0.4])
    >>> ctmc_pert = ctmc.perturb_rate(0, 1, delta_W=0.5)
    """

    def __init__(self, W: NDArray[np.float64]):
        W = np.asarray(W, dtype=np.float64)
        self._size = W.shape[0]

        # Ensure proper diagonal (column sums = 0)
        self._W = self._normalize_rate_matrix(W)

        # Spectral decomposition
        (self._eigenvalues,
         self._left_eigenvectors,
         self._right_eigenvectors,
         self._A) = self._spectral_decomposition()

        # Steady state is the right eigenvector for lambda = 0
        self._steady_state = self._right_eigenvectors[:, 0].real.copy()
        self._steady_state /= self._steady_state.sum()

        # Lazy-computed MFPT matrix
        self._mfpt_matrix: Optional[NDArray[np.float64]] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Number of states."""
        return self._size

    @property
    def W(self) -> NDArray[np.float64]:
        """Transition rate matrix (copy)."""
        return self._W.copy()

    @property
    def steady_state(self) -> NDArray[np.float64]:
        """Stationary distribution pi (copy)."""
        return self._steady_state.copy()

    @property
    def eigenvalues(self) -> NDArray[np.complex128]:
        """Eigenvalues of W, sorted by real part (descending)."""
        return self._eigenvalues.copy()

    def __repr__(self) -> str:
        return f"CTMC(size={self._size}, sigma={self.entropy_production_rate():.4f})"

    # =========================================================================
    # Layer 1: Fundamental Rate Perturbations
    # =========================================================================

    def perturb_rate(self, m: int, n: int, delta_W: float) -> CTMC:
        """Perturb a single transition rate: W'[m,n] = W[m,n] + delta_W.

        This is the most fundamental perturbation operation.
        All parameter-based perturbations (A, B, E, F) are built on this.

        Parameters
        ----------
        m : int
            Target state (rate W[m,n] is from n to m)
        n : int
            Source state
        delta_W : float
            Change in rate

        Returns
        -------
        CTMC
            New CTMC with perturbed rate matrix
        """
        W_new = self._W.copy()
        W_new[m, n] += delta_W
        return CTMC(W_new)

    def perturb_rates(self, delta_W: NDArray[np.float64]) -> CTMC:
        """Perturb multiple transition rates: W' = W + delta_W.

        Only off-diagonal elements are used (diagonal is recomputed).

        Parameters
        ----------
        delta_W : ndarray
            Matrix of rate changes

        Returns
        -------
        CTMC
            New CTMC with perturbed rate matrix
        """
        return CTMC(self._W + delta_W)

    def with_rates(self, W_new: NDArray[np.float64]) -> CTMC:
        """Create new CTMC with completely new rate matrix."""
        return CTMC(W_new)

    # =========================================================================
    # Internal: Spectral Decomposition
    # =========================================================================

    def _normalize_rate_matrix(self, W: NDArray[np.float64]) -> NDArray[np.float64]:
        """Set diagonal so that columns sum to zero."""
        W = W - np.diag(np.diag(W))
        W = W - np.diag(np.sum(W, axis=0))
        return W

    def _spectral_decomposition(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Compute eigenvalue decomposition with biorthonormal projectors.

        Returns
        -------
        eigenvalues : ndarray
            Sorted eigenvalues (lam_0 = 0, others negative)
        u : ndarray
            Left eigenvectors (columns)
        v : ndarray
            Right eigenvectors (columns)
        A : ndarray
            Projector matrices A^k = |v_k><u_k|
        """
        eigenvalues, u, v = eig(self._W, left=True)

        # Sort by real part (descending: 0, -lam_1, -lam_2, ...)
        sort_idx = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        v = v[:, sort_idx]
        u = u[:, sort_idx].conj()

        # Build biorthonormal left basis via V^{-1}.
        # This is robust under degeneracies: U @ V = I exactly (up to
        # numerical error) even when eigenvalues are degenerate.
        V = v
        U = np.linalg.inv(V)  # rows are <u_m| (biorthonormal)

        # Store left eigenvectors as columns for compatibility (u[:,m])
        u = U.T.conj()

        # Projectors Lambda^{(m)} = |v_m><u_m|
        A = np.array([np.outer(V[:, m], U[m, :]) for m in range(self._size)])

        return eigenvalues, u, v, A

    # =========================================================================
    # Mean First-Passage Times
    # =========================================================================

    def mfpt(self, i: int, j: int, return_time: bool = False) -> float:
        """Mean first-passage time from state j to state i.

        Uses the spectral formula:
            <t_ij> = sum_{k>0} (A^k_ij - A^k_ii) / (lam_k pi_i)

        Parameters
        ----------
        i : int
            Target state
        j : int
            Initial state
        return_time : bool
            If True and i == j, return mean return time (MFRT).
            If False and i == j, return 0.

        Returns
        -------
        float
            <t_ij>
        """
        if i == j:
            if return_time:
                return -1.0 / (self._W[i, i] * self._steady_state[i])
            return 0.0

        result = np.sum(
            (self._A[1:, i, j] - self._A[1:, i, i]) / self._eigenvalues[1:]
        ) / self._steady_state[i]

        return float(np.real(result))

    def mfpt_matrix(self) -> NDArray[np.float64]:
        """Full MFPT matrix. T[i,j] = <t_ij>, diagonal is zero."""
        if self._mfpt_matrix is None:
            self._mfpt_matrix = np.array([
                [self.mfpt(i, j) for j in range(self._size)]
                for i in range(self._size)
            ])
        return self._mfpt_matrix.copy()

    def mfrt(self, i: int) -> float:
        """Mean first return time to state i: MFRT_i = 1/pi_i."""
        return 1.0 / self._steady_state[i]

    # =========================================================================
    # Higher-Order FPT Moments (Spectral Recursion)
    #
    # Based on "Spectral Theory of First Passage Times" (Shiling Liang).
    # Computes spectral moments a_{ij}^{(k)}, then uses the ratio-formula
    # recursion to obtain raw moments <t_{ij}^k>.
    # =========================================================================

    def spectral_moment(self, i: int, j: int, k: int) -> complex:
        """Spectral moment a_{ij}^{(k)}.

        a_{ij}^{(k)} = (1/pi_i) sum_{m>=1} Lambda^{(m)}_{ij} / lam_m^k

        Uses the biorthonormal spectral projectors Lambda^{(m)} = |v_m><u_m|.
        Returns complex in general; for real generators the final FPT moments
        are real up to numerical noise.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        pi_i = self._steady_state[i]
        lam = self._eigenvalues[1:]
        proj = self._A[1:, i, j]
        return (proj / (lam ** k)).sum() / pi_i

    def spectral_difference(self, i: int, j: int, k: int) -> complex:
        """Spectral difference Delta_{ij}^{(k)} = a_{ii}^{(k)} - a_{ij}^{(k)}."""
        return self.spectral_moment(i, i, k) - self.spectral_moment(i, j, k)

    def _fpt_moment_coeffs(self, i: int, j: int, K: int) -> NDArray[np.complex128]:
        """Taylor coefficients f_k for the FPT MGF via the note's recursion.

        F~(s) = sum_{k>=0} f_k s^k  with f_0 = 1 and
            f_k = Delta_{ij}^{(k)} + sum_{l=1}^{k-1} a_{ii}^{(k-l)} f_l.
        """
        if K < 0:
            raise ValueError("K must be >= 0")
        f = np.zeros(K + 1, dtype=np.complex128)
        f[0] = 1.0
        if K == 0:
            return f

        # Precompute diagonal a_{ii}^{(k)} up to K
        aii = np.zeros(K + 1, dtype=np.complex128)
        for k in range(1, K + 1):
            aii[k] = self.spectral_moment(i, i, k)

        for k in range(1, K + 1):
            acc = self.spectral_difference(i, j, k)
            for l in range(1, k):
                acc += aii[k - l] * f[l]
            f[k] = acc
        return f

    def higher_order_moment(self, i: int, j: int, k: int) -> float:
        """k-th raw moment <t_{ij}^k> using the spectral recursion.

        Parameters
        ----------
        i : int
            Target state
        j : int
            Source state
        k : int
            Moment order (k >= 1)

        Returns
        -------
        float
            <t_{ij}^k>; returns 0 for i == j.
        """
        if i == j:
            return 0.0
        if k < 1:
            raise ValueError("Moment order k must be >= 1")
        f = self._fpt_moment_coeffs(i, j, k)
        moment = ((-1) ** k) * float(math.factorial(k)) * f[k]
        return float(np.real(moment))

    # --- Convenience aliases for specific moment orders ---

    def fpt_moment(self, i: int, j: int, k: int) -> float:
        """Alias for higher_order_moment: k-th raw moment <t_{ij}^k>."""
        return float(self.higher_order_moment(i, j, k))

    def mfpt_second_moment(self, i: int, j: int) -> float:
        """Second moment <t_{ij}^2>."""
        return self.higher_order_moment(i, j, 2)

    def mfpt_third_moment(self, i: int, j: int) -> float:
        """Third moment <t_{ij}^3>."""
        return self.higher_order_moment(i, j, 3)

    def mfpt_fourth_moment(self, i: int, j: int) -> float:
        """Fourth moment <t_{ij}^4>."""
        return self.higher_order_moment(i, j, 4)

    # =========================================================================
    # FPT Derived Statistics
    # =========================================================================

    def mfpt_variance(self, i: int, j: int) -> float:
        """Variance Var[t_{ij}] = <t^2> - <t>^2."""
        if i == j:
            return 0.0
        m1 = self.higher_order_moment(i, j, 1)
        m2 = self.higher_order_moment(i, j, 2)
        return float(max(0.0, np.real(m2 - m1 * m1)))

    def mfpt_skewness(self, i: int, j: int) -> float:
        """Skewness of FPT distribution from j to i.

        Skewness = E[(t - mu)^3] / sigma^3

        Returns
        -------
        float
            Positive = right-tailed, negative = left-tailed.
        """
        if i == j:
            return 0.0
        m1 = self.higher_order_moment(i, j, 1)
        m2 = self.higher_order_moment(i, j, 2)
        m3 = self.higher_order_moment(i, j, 3)
        var = m2 - m1 ** 2
        if var <= 0:
            return 0.0
        mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        return float(np.real(mu3) / (var ** 1.5))

    def mfpt_kurtosis(self, i: int, j: int, excess: bool = True) -> float:
        """Kurtosis of FPT distribution from j to i.

        Parameters
        ----------
        i, j : int
            Target and source states
        excess : bool
            If True (default), return excess kurtosis (subtract 3).

        Returns
        -------
        float
            Kurtosis (excess kurtosis = 0 for normal distribution).
        """
        if i == j:
            return 0.0
        m1 = self.higher_order_moment(i, j, 1)
        m2 = self.higher_order_moment(i, j, 2)
        m3 = self.higher_order_moment(i, j, 3)
        m4 = self.higher_order_moment(i, j, 4)
        var = m2 - m1 ** 2
        if var <= 0:
            return 0.0
        mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        kurt = float(np.real(mu4) / (var ** 2))
        return kurt - 3.0 if excess else kurt

    def mfpt_cv(self, i: int, j: int) -> float:
        """Coefficient of variation of FPT: CV = sigma / mu."""
        if i == j:
            return 0.0
        m1 = self.higher_order_moment(i, j, 1)
        if m1 <= 0:
            return 0.0
        return float(np.sqrt(self.mfpt_variance(i, j)) / m1)

    # =========================================================================
    # FPT Cumulants (up to 4th order)
    # =========================================================================

    def mfpt_cumulant(self, i: int, j: int, k: int) -> float:
        """k-th cumulant kappa_k of the first-passage time t_{ij}.

        Computed from raw moments via standard formulas:
            kappa_1 = m1
            kappa_2 = m2 - m1^2
            kappa_3 = m3 - 3 m2 m1 + 2 m1^3
            kappa_4 = m4 - 4 m3 m1 - 3 m2^2 + 12 m2 m1^2 - 6 m1^4

        Parameters
        ----------
        i : int
            Target state
        j : int
            Initial state
        k : int
            Order (1..4)

        Returns
        -------
        float
            kappa_k(t_{ij})
        """
        if i == j:
            return 0.0
        if k < 1 or k > 4:
            raise ValueError('mfpt_cumulant currently supports k=1..4')

        m1 = self.higher_order_moment(i, j, 1)
        if k == 1:
            return float(m1)

        m2 = self.higher_order_moment(i, j, 2)
        if k == 2:
            return float(np.real(m2 - m1 * m1))

        m3 = self.higher_order_moment(i, j, 3)
        if k == 3:
            return float(np.real(m3 - 3 * m2 * m1 + 2 * m1 ** 3))

        m4 = self.higher_order_moment(i, j, 4)
        return float(np.real(
            m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
        ))

    def fpt_cumulant(self, i: int, j: int, k: int) -> float:
        """Alias for mfpt_cumulant: k-th cumulant kappa_k(t_{ij})."""
        return float(self.mfpt_cumulant(i, j, k))

    def mfpt_cumulants(self, i: int, j: int, max_k: int = 4) -> dict:
        """Return cumulants kappa_1..kappa_{max_k} for FPT t_{ij}."""
        if max_k < 1 or max_k > 4:
            raise ValueError('max_k must be 1..4')
        return {k: self.mfpt_cumulant(i, j, k) for k in range(1, max_k + 1)}

    # =========================================================================
    # FPT Summary
    # =========================================================================

    def mfpt_statistics(self, i: int, j: int) -> Dict[str, float]:
        """All first-passage time statistics from j to i.

        Returns
        -------
        dict
            Keys: 'mean', 'second_moment', 'variance', 'third_moment',
            'fourth_moment', 'skewness', 'excess_kurtosis', 'cv'
        """
        if i == j:
            return {
                'mean': 0.0, 'second_moment': 0.0, 'variance': 0.0,
                'third_moment': 0.0, 'fourth_moment': 0.0,
                'skewness': 0.0, 'excess_kurtosis': 0.0, 'cv': 0.0,
            }
        return {
            'mean': self.higher_order_moment(i, j, 1),
            'second_moment': self.mfpt_second_moment(i, j),
            'variance': self.mfpt_variance(i, j),
            'third_moment': self.mfpt_third_moment(i, j),
            'fourth_moment': self.mfpt_fourth_moment(i, j),
            'skewness': self.mfpt_skewness(i, j),
            'excess_kurtosis': self.mfpt_kurtosis(i, j, excess=True),
            'cv': self.mfpt_cv(i, j),
        }

    # =========================================================================
    # FPT Convenience OO API (delegates to markov_response.fpt)
    # =========================================================================

    def fpt_moments(self, i: int, j: int, max_k: int = 4):
        """Return raw FPT moments up to order max_k as a dict."""
        from . import fpt as _fpt
        return _fpt.fpt_moments(self, i=i, j=j, max_k=max_k)

    def fpt_cumulants(self, i: int, j: int, max_k: int = 4):
        """Return FPT cumulants up to order max_k as a dict."""
        from . import fpt as _fpt
        return _fpt.fpt_cumulants(self, i=i, j=j, max_k=max_k)

    def fpt_stats(self, i: int, j: int, max_k: int = 4):
        """Return a structured FPTStats object."""
        from . import fpt as _fpt
        return _fpt.fpt_stats(self, i=i, j=j, max_k=max_k)

    # =========================================================================
    # FPT Matrix-Level Computations
    # =========================================================================

    def _off_diagonal_matrix(self, func) -> NDArray[np.float64]:
        """Helper: build N x N matrix with func(i, j) for i != j, 0 on diagonal."""
        M = np.zeros((self._size, self._size))
        for i in range(self._size):
            for j in range(self._size):
                if i != j:
                    M[i, j] = func(i, j)
        return M

    def higher_order_moments_matrix(self, k: int) -> NDArray[np.float64]:
        """k-th moment matrix: M[i,j] = <t_{ij}^k>."""
        return self._off_diagonal_matrix(lambda i, j: self.higher_order_moment(i, j, k))

    def mfpt_variance_matrix(self) -> NDArray[np.float64]:
        """Variance matrix: V[i,j] = Var(t_{ij})."""
        return self._off_diagonal_matrix(self.mfpt_variance)

    def mfpt_skewness_matrix(self) -> NDArray[np.float64]:
        """Skewness matrix: S[i,j] = Skewness(t_{ij})."""
        return self._off_diagonal_matrix(self.mfpt_skewness)

    def mfpt_kurtosis_matrix(self, excess: bool = True) -> NDArray[np.float64]:
        """Kurtosis matrix: K[i,j] = Kurtosis(t_{ij})."""
        return self._off_diagonal_matrix(lambda i, j: self.mfpt_kurtosis(i, j, excess=excess))

    def mfpt_cv_matrix(self) -> NDArray[np.float64]:
        """CV matrix: CV[i,j] = sigma_{ij} / mu_{ij}."""
        return self._off_diagonal_matrix(self.mfpt_cv)

    def mfpt_cumulant_matrix(self, k: int) -> NDArray[np.float64]:
        """Cumulant matrix: K[i,j] = kappa_k(t_{ij})."""
        if k < 1 or k > 4:
            raise ValueError('k must be 1..4')
        return self._off_diagonal_matrix(lambda i, j: self.mfpt_cumulant(i, j, k))

    # =========================================================================
    # Fluxes, Currents, Activities
    # =========================================================================

    def flux(self, i: int, j: int) -> float:
        """Directed flux from j to i: phi_ij = W_ij pi_j."""
        return self._W[i, j] * self._steady_state[j]

    def current(self, i: int, j: int) -> float:
        """Net probability current from j to i: j_ij = phi_ij - phi_ji."""
        return self.flux(i, j) - self.flux(j, i)

    def activity(self, i: int, j: int) -> float:
        """Activity (total traffic) between i and j: a_ij = phi_ij + phi_ji."""
        return self.flux(i, j) + self.flux(j, i)

    def flux_matrix(self) -> NDArray[np.float64]:
        """Full flux matrix: Phi[i,j] = W[i,j] * pi[j]."""
        return self._W * self._steady_state

    def current_matrix(self) -> NDArray[np.float64]:
        """Full current matrix: J = Phi - Phi^T."""
        phi = self.flux_matrix()
        return phi - phi.T

    def activity_matrix(self) -> NDArray[np.float64]:
        """Full activity matrix: A = Phi + Phi^T."""
        phi = self.flux_matrix()
        return phi + phi.T

    # =========================================================================
    # Observables
    # =========================================================================

    def observable_mean(self, O: NDArray[np.float64]) -> float:
        """Steady-state average: <O> = sum_i O_i pi_i."""
        return float(np.dot(O, self._steady_state))

    def observable_variance(self, O: NDArray[np.float64]) -> float:
        """Steady-state variance: Var(O) = <O^2> - <O>^2."""
        mean = self.observable_mean(O)
        mean_sq = self.observable_mean(O ** 2)
        return mean_sq - mean ** 2

    # =========================================================================
    # Thermodynamics
    # =========================================================================

    def entropy_production_rate(self) -> float:
        """Total entropy production rate.

        sigma = sum_{i<j} (phi_ij - phi_ji) ln(phi_ij / phi_ji)
        """
        sigma = 0.0
        for i in range(self._size):
            for j in range(i + 1, self._size):
                phi_ij = self.flux(i, j)
                phi_ji = self.flux(j, i)
                if phi_ij > 0 and phi_ji > 0:
                    sigma += (phi_ij - phi_ji) * np.log(phi_ij / phi_ji)
        return sigma

    def cycle_affinity(self, cycle: list[int]) -> float:
        """Cycle affinity F_c = sum_{edges} ln(W_ij / W_ji).

        Parameters
        ----------
        cycle : list
            States forming a cycle, e.g. [0, 1, 2] for 0->1->2->0.
        """
        affinity = 0.0
        for k in range(len(cycle)):
            i = cycle[k]
            j = cycle[(k + 1) % len(cycle)]
            if self._W[j, i] > 0 and self._W[i, j] > 0:
                affinity += np.log(self._W[j, i] / self._W[i, j])
        return affinity

    # =========================================================================
    # Propagator
    # =========================================================================

    def propagator(self, t: float, method: str = 'spectral') -> NDArray[np.float64]:
        """Transition probability matrix P(t) = exp(W t).

        P[i,j](t) = P(X(t)=i | X(0)=j)

        Parameters
        ----------
        t : float
            Time
        method : str
            'spectral' (default) or 'expm'
        """
        if method == 'spectral':
            return np.real(np.sum(
                np.exp(self._eigenvalues * t)[:, None, None] * self._A,
                axis=0,
            ))
        else:
            return expm(self._W * t)


# =============================================================================
# Layer 2: Parametrized Perturbations
# =============================================================================

def perturb_activation_energy(ctmc: CTMC, m: int, n: int, delta_A: float) -> CTMC:
    """Apply activation energy perturbation dA_mn.

    W'_mn = W_mn exp(-dA_mn).  Only affects edge m<-n, not n<-m.

    Parameters
    ----------
    ctmc : CTMC
        Original system
    m, n : int
        Target and source states
    delta_A : float
        Perturbation strength

    Returns
    -------
    CTMC
        Perturbed system
    """
    W = ctmc.W
    delta_W = np.zeros_like(W)
    delta_W[m, n] = W[m, n] * (np.exp(-delta_A) - 1)
    return ctmc.perturb_rates(delta_W)


def perturb_barrier(ctmc: CTMC, m: int, n: int, delta_B: float) -> CTMC:
    """Apply symmetric barrier perturbation dB_mn.

    W'_mn = W_mn exp(-dB),  W'_nm = W_nm exp(-dB).

    Parameters
    ----------
    ctmc : CTMC
        Original system
    m, n : int
        Edge endpoints
    delta_B : float
        Perturbation strength

    Returns
    -------
    CTMC
        Perturbed system
    """
    W = ctmc.W
    factor = np.exp(-delta_B) - 1
    delta_W = np.zeros_like(W)
    delta_W[m, n] = W[m, n] * factor
    delta_W[n, m] = W[n, m] * factor
    return ctmc.perturb_rates(delta_W)


def perturb_energy(ctmc: CTMC, m: int, delta_E: float) -> CTMC:
    """Apply vertex energy perturbation dE_m.

    W'_im = W_im exp(dE_m) for all i != m.
    Higher E_m -> faster escape from m.

    Parameters
    ----------
    ctmc : CTMC
        Original system
    m : int
        State to perturb
    delta_E : float
        Perturbation strength

    Returns
    -------
    CTMC
        Perturbed system
    """
    W = ctmc.W
    factor = np.exp(delta_E) - 1
    delta_W = np.zeros_like(W)
    for i in range(ctmc.size):
        if i != m:
            delta_W[i, m] = W[i, m] * factor
    return ctmc.perturb_rates(delta_W)


def perturb_force(ctmc: CTMC, m: int, n: int, delta_F: float) -> CTMC:
    """Apply antisymmetric force perturbation dF_mn.

    W'_mn = W_mn exp(-dF/2),  W'_nm = W_nm exp(+dF/2).

    Parameters
    ----------
    ctmc : CTMC
        Original system
    m, n : int
        Edge endpoints
    delta_F : float
        Perturbation strength

    Returns
    -------
    CTMC
        Perturbed system
    """
    W = ctmc.W
    delta_W = np.zeros_like(W)
    delta_W[m, n] = W[m, n] * (np.exp(-delta_F / 2) - 1)
    delta_W[n, m] = W[n, m] * (np.exp(delta_F / 2) - 1)
    return ctmc.perturb_rates(delta_W)


# =============================================================================
# Factory Functions & Parameters
# =============================================================================

@dataclass
class CTMCParameters:
    """Parameters for rate matrix: W_ij = exp(-(B_ij - E_j - F_ij/2))."""
    E: NDArray[np.float64]  # Energy levels (vertex)
    B: NDArray[np.float64]  # Barrier heights (symmetric edge)
    F: NDArray[np.float64]  # Driving forces (antisymmetric edge)


def rates_from_parameters(
    E: NDArray[np.float64],
    B: NDArray[np.float64],
    F: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute transition rate matrix from physical parameters.

    W_ij = exp(-(B_ij - E_j - F_ij/2))  for i != j

    Parameters
    ----------
    E : ndarray (N,)
        Energy levels
    B : ndarray (N, N)
        Barrier heights (symmetric: B_ij = B_ji)
    F : ndarray (N, N)
        Driving forces (antisymmetric: F_ij = -F_ji)

    Returns
    -------
    ndarray (N, N)
        Transition rate matrix
    """
    N = len(E)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] = np.exp(-(B[i, j] - E[j] - F[i, j] / 2))
    W -= np.diag(np.sum(W, axis=0))
    return W


def generate_random_ctmc(
    size: int,
    scale: float = 10.0,
    energy_scale: Optional[float] = None,
    barrier_scale: Optional[float] = None,
    force_scale: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[CTMC, CTMCParameters]:
    """Generate a random fully-connected CTMC.

    Parameters
    ----------
    size : int
        Number of states
    scale : float
        Unified scale for all parameters (default 10.0)
    energy_scale, barrier_scale, force_scale : float, optional
        Override individual scales
    seed : int, optional
        Random seed

    Returns
    -------
    ctmc : CTMC
    params : CTMCParameters
    """
    if seed is not None:
        np.random.seed(seed)

    e_scale = energy_scale if energy_scale is not None else scale
    b_scale = barrier_scale if barrier_scale is not None else scale
    f_scale = force_scale if force_scale is not None else scale

    E = (np.random.rand(size) - 0.5) * e_scale
    B = (np.random.rand(size, size) - 0.5) * b_scale
    B = (B + B.T) / 2
    F = (np.random.rand(size, size) - 0.5) * f_scale
    F = (F - F.T) / 2

    W = rates_from_parameters(E, B, F)
    params = CTMCParameters(E=E, B=B, F=F)
    return CTMC(W), params


def ctmc_from_parameters(params: CTMCParameters) -> CTMC:
    """Create CTMC from E, B, F parameters."""
    return CTMC(rates_from_parameters(params.E, params.B, params.F))
