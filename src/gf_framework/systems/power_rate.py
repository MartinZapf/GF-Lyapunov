from __future__ import annotations

"""
Power-Rate Reaching Law System Definition.

The simplest 1D finite-time/exponential reaching law:
    ṡ = -k⌈s⌋^α

This is r-homogeneous with:
    - weights r = (1,)
    - degree κ = α - 1
    - Lyapunov degree: m = 1

For α = 0: Standard sign reaching (κ = -1, same as FOSMC)
For 0 < α < 1: Finite-time convergence (κ < 0)
For α = 1: Exponential convergence (κ = 0)
For α > 1: Polynomial convergence (κ > 0)

The exact Lyapunov function is V(s) = |s|.
"""

from typing import Tuple, List
from fractions import Fraction
import numpy as np

from gf_framework.algebra.core import GeneralizedForm, GFTerm


def sgnpow(x: np.ndarray, p: float) -> np.ndarray:
    """Signed power: sign(x) * |x|^p"""
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


def _exact_alpha(alpha) -> Fraction:
    """Preserve rationals; interpret other inputs by their decimal spelling.

    No denominator cap is applied: callers needing 1/3 should pass Fraction(1, 3).
    Nonnegative exponents include the sign and exponential boundary cases.
    """
    value = alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))
    if value < 0:
        raise ValueError("Power-rate alpha must be nonnegative")
    return value


class PowerRateSystem:
    """
    Power-Rate Reaching Law system.

    ṡ = -k⌈s⌋^α

    where:
    - k > 0 is the reaching gain
    - α is the power exponent (typically 0 < α < 1 for finite-time)
    """

    def __init__(self, k: float = 1.5, alpha: float = 0.5):
        """
        Initialize Power-Rate system.

        Args:
            k: Reaching gain (must be positive)
            alpha: Power exponent (0 < α < 1 for finite-time)
        """
        assert k > 0, "Gain must be positive"
        self.k = k
        self.alpha_exact = _exact_alpha(alpha)
        self.alpha = float(self.alpha_exact)

    @property
    def weights(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def kappa(self) -> float:
        return float(self.alpha_exact - 1)

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics: ṡ = -k⌈s⌋^α

        Args:
            x: State tensor of shape (batch, 1) or (1,) or scalar

        Returns:
            ds/dt
        """
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)

        return -self.k * sgnpow(x, self.alpha)

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when s > 0."""
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)

        # For s > 0: ⌈s⌋^α = |s|^α
        return -self.k * np.abs(x) ** self.alpha

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when s < 0."""
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = np.expand_dims(x, 0)

        # For s < 0: ⌈s⌋^α = -|s|^α
        return self.k * np.abs(x) ** self.alpha


def create_power_rate_dynamics(k: float = 1.5, alpha: float = 0.5):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = PowerRateSystem(k, alpha)
    return system.dynamics_plus, system.dynamics_minus


# Default configuration (finite-time with α = 0.5)
POWER_RATE_WEIGHTS = (1,)
POWER_RATE_ALPHA = 0.5
POWER_RATE_KAPPA = POWER_RATE_ALPHA - 1.0  # -0.5
POWER_RATE_DEGREE = 1

# Known exact Lyapunov function: V(s) = |s|
# dV/dt = sign(s) · ṡ = sign(s) · (-k⌈s⌋^α) = -k|s|^α < 0


def get_power_rate_dynamics_gf(k: float = 1.5, alpha: float = 0.5) -> List[GeneralizedForm]:
    """
    Get Power-Rate dynamics as Generalized Forms.

    For Power-Rate: ṡ = -k⌈s⌋^α

    The dynamics is a GF of degree α.

    Args:
        k: Reaching gain
        alpha: Power exponent

    Returns:
        List with single GF for f₁(s) = -k⌈s⌋^α
    """
    # f₁(s) = -k⌈s⌋^α
    # Weighted degree: 1 * α = α
    alpha_frac = _exact_alpha(alpha)

    f1_term = GFTerm(
        coefficient=-k,
        exponents=(alpha_frac,),
        term_types=('signed',)  # ⌈s⌋^α
    )

    # Weighted degree of dynamics: r₁ * α = 1 * α = α
    # But for homogeneous systems, f should have degree κ + r₁ = (α-1) + 1 = α
    f1 = GeneralizedForm(
        degree_m=alpha_frac,
        weights_r=(1,),
        terms=[f1_term]
    )

    return [f1]


def get_power_rate_known_lyapunov(coef: float = 1.0) -> GeneralizedForm:
    """
    Get the known Power-Rate Lyapunov function V(s) = coef * |s|.

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
