from __future__ import annotations

"""
FOSMC (First-Order Sliding Mode Control) System Definition.

The simplest SMC benchmark:
    ṡ = -k·sign(s) + δ

This is r-homogeneous with:
    - weights r = (1,)
    - degree κ = -1
    - Lyapunov degree: m = 1

The exact Lyapunov function is V(s) = |s|.
"""

from typing import Tuple, List
from fractions import Fraction
import numpy as np

from gf_framework.algebra.core import GeneralizedForm, GFTerm


class FOSMCSystem:
    """
    First-Order Sliding Mode Control system.

    ṡ = -k·sign(s) + δ(t)

    where |δ(t)| ≤ δ₀ < k for stability.
    """

    def __init__(self, k: float = 1.0, delta_bound: float = 0.5):
        """
        Initialize FOSMC system.

        Args:
            k: Switching gain (must be > delta_bound)
            delta_bound: Disturbance bound
        """
        assert k > delta_bound, "Gain must exceed disturbance bound"
        self.k = k
        self.delta_bound = delta_bound

    @property
    def weights(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def kappa(self) -> float:
        return -1.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics: ṡ = -k·sign(s)

        Args:
            x: State tensor of shape (batch, 1) or (1,) or scalar

        Returns:
            ds/dt
        """
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)

        return -self.k * np.sign(x)

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when s > 0 (sign(s) = +1)."""
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)

        return -self.k * np.ones_like(x)

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when s < 0 (sign(s) = -1)."""
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)

        return self.k * np.ones_like(x)


def create_fosmc_dynamics(k: float = 1.0):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = FOSMCSystem(k)
    return system.dynamics_plus, system.dynamics_minus


# Default configuration
FOSMC_WEIGHTS = (1,)
FOSMC_KAPPA = -1.0
FOSMC_DEGREE = 1

# Known exact Lyapunov function: V(s) = |s|
# dV/dt = sign(s) · ṡ = sign(s) · (-k·sign(s)) = -k < 0


def get_fosmc_dynamics_gf(k: float = 1.0) -> List[GeneralizedForm]:
    """
    Get FOSMC dynamics as Generalized Forms.

    For FOSMC: ṡ = -k·sign(s) = -k⌈s⌋^0

    The dynamics is a GF of degree 0 (sign function).

    Args:
        k: Switching gain

    Returns:
        List with single GF for f₁(s) = -k·sign(s)
    """
    # f₁(s) = -k·sign(s) = -k⌈s⌋^0
    # Weighted degree: 1 * 0 = 0
    f1_term = GFTerm(
        coefficient=-k,
        exponents=(Fraction(0),),
        term_types=('signed',)  # ⌈s⌋^0 = sign(s)
    )

    f1 = GeneralizedForm(
        degree_m=0,
        weights_r=(1,),
        terms=[f1_term]
    )

    return [f1]


def get_fosmc_known_lyapunov(coef: float = 1.0) -> GeneralizedForm:
    """
    Get the known FOSMC Lyapunov function V(s) = coef * |s|.

    Args:
        coef: Coefficient (default 1.0)

    Returns:
        GeneralizedForm for V(s) = coef * |s|
    """
    term = GFTerm(
        coefficient=coef,
        exponents=(Fraction(1),),
        term_types=('abs',)
    )

    return GeneralizedForm(
        degree_m=1,
        weights_r=(1,),
        terms=[term]
    )
