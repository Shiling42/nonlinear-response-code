"""markov_response

General-purpose toolkit for **continuous-time Markov chains (CTMCs)**.

Core capabilities
-----------------
- Spectral decomposition of rate matrices (eigenvalues, projectors, propagator)
- First-passage time (FPT) moments and cumulants up to 4th order
- Two-layer perturbation interface (direct rate changes + parametrized: A, B, E, F)
- Steady-state analysis: flux, current, activity, observable mean/variance
- Thermodynamics: entropy production rate, cycle affinity

Notation
--------
Throughout the package: **t_{ij} means first passage from j → i** (source j, target i).
"""

from .core import (
    CTMC,
    CTMCParameters,
    perturb_activation_energy,
    perturb_barrier,
    perturb_energy,
    perturb_force,
    rates_from_parameters,
    generate_random_ctmc,
    ctmc_from_parameters,
)

from .fpt import FPTStats

__version__ = "0.1.0"
