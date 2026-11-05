from __future__ import annotations

"""
STA (Super-Twisting Algorithm) System Definition.

From Moreno-Osorio (2012):
    ṡ = -k₁|s|^(1/2)·sign(s) + v
    v̇ = -k₂·sign(s) + δ

This is r-homogeneous with:
    - weights r = (2, 1)
    - degree κ = -1
    - Lyapunov degree: m = 2

The known Lyapunov function (Moreno-Osorio) is quadratic in lifted coords:
    V(s,v) = p₁₁|s| + 2p₁₂·sign(s)·√|s|·v + p₂₂·v²

where P = [[p₁₁, p₁₂], [p₁₂, p₂₂]] solves a specific ALE.
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


class STASystem:
    """
    Super-Twisting Algorithm closed-loop system.

    Default gains: k₁ = 1.5, k₂ = 1.1. Positive gains give nominal decay.
    """

    def __init__(
        self,
        k1: float = 1.5,
        k2: float = 1.1,
    ):
        """
        Initialize STA system.

        Args:
            k1, k2: Positive controller gains for the nominal system
        """
        self.k1 = k1
        self.k2 = k2

        # Nominal decay follows from the positive-gain P/Q identity below.
        # Disturbance robustness needs its own gain-dependent proof.

    @property
    def weights(self) -> Tuple[int, ...]:
        return (2, 1)

    @property
    def kappa(self) -> float:
        return -1.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics (no disturbance).

        ṡ = -k₁⌈s⌋^(1/2) + v
        v̇ = -k₂·sign(s)

        Args:
            x: State tensor [s, v] of shape (batch, 2) or (2,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        s, v = x[:, 0], x[:, 1]

        ds = -self.k1 * sgnpow(s, 0.5) + v
        dv = -self.k2 * np.sign(s)

        return np.stack([ds, dv], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when s > 0 (sign(s) = +1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        s, v = x[:, 0], x[:, 1]

        ds = -self.k1 * sgnpow(s, 0.5) + v
        dv = -self.k2 * np.ones_like(s)

        return np.stack([ds, dv], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when s < 0 (sign(s) = -1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        s, v = x[:, 0], x[:, 1]

        ds = -self.k1 * sgnpow(s, 0.5) + v
        dv = self.k2 * np.ones_like(s)

        return np.stack([ds, dv], axis=1).squeeze()


def create_sta_dynamics(k1: float = 1.5, k2: float = 1.1):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = STASystem(k1, k2)
    return system.dynamics_plus, system.dynamics_minus


def moreno_osorio_lyapunov_matrix(k1: float, k2: float) -> np.ndarray:
    """
    Compute Moreno-Osorio Lyapunov matrix P.

    The Lyapunov function in lifted coordinates ξ = [|s|^(1/2)·sign(s), v]
    is V(ξ) = ξᵀPξ.

    P is found by solving a specific algebraic Lyapunov equation.

    Args:
        k1, k2: STA gains

    Returns:
        2x2 positive definite matrix P
    """
    # For positive gains, A = [[-k1/2, 1/2], [-k2, 0]] and
    # -(A.T @ P + P @ A) = [[k1*(2*k2+k1**2), -k1**2],
    #                       [-k1**2, k1]] is positive definite.
    # In either open half-plane, Vdot = -xi.T @ Q @ xi / sqrt(abs(s)).
    p11 = 4 * k2 + k1 ** 2
    p12 = -k1
    p22 = 2

    P = np.array([[p11, p12], [p12, p22]])
    return P


# Default configuration
STA_WEIGHTS = (2, 1)
STA_KAPPA = -1.0
STA_DEGREE = 2

# Known Lyapunov function structure:
# V(s,v) = (4k₂ + k₁²)|s| - 2k₁·sign(s)·√|s|·v + 2v²
#
# In GF notation with m=2, r=(2,1):
# V(x₁,x₂) = α₁|x₁|^1 + α₁₂⌈x₁⌋^(1/2)x₂ + α₂|x₂|^2


def get_sta_dynamics_gf(k1: float = 1.5, k2: float = 1.1) -> List[GeneralizedForm]:
    """
    Get STA dynamics as Generalized Forms.

    For STA:
        f₁ = ẋ₁ = -k₁⌈x₁⌋^(1/2) + x₂
        f₂ = ẋ₂ = -k₂·sign(x₁) = -k₂⌈x₁⌋^0

    Args:
        k1, k2: Controller gains

    Returns:
        List of GFs [f₁, f₂]
    """
    # f₁ = -k₁⌈x₁⌋^(1/2) + x₂
    # Term 1: -k₁⌈x₁⌋^(1/2), weighted degree: 2*(1/2) + 1*0 = 1
    # Term 2: x₂ = ⌈x₂⌋^1, weighted degree: 2*0 + 1*1 = 1
    f1_terms = [
        GFTerm(
            coefficient=-k1,
            exponents=(Fraction(1, 2), Fraction(0)),
            term_types=('signed', 'abs')
        ),
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1)),
            term_types=('abs', 'signed')  # x₂ needs 'signed' to allow negative values
        ),
    ]
    f1 = GeneralizedForm(degree_m=1, weights_r=(2, 1), terms=f1_terms)

    # f₂ = -k₂⌈x₁⌋^0 (= -k₂·sign(x₁))
    # Weighted degree: 2*0 + 1*0 = 0
    f2_terms = [
        GFTerm(
            coefficient=-k2,
            exponents=(Fraction(0), Fraction(0)),
            term_types=('signed', 'abs')  # sign(x₁) comes from 'signed' type with exp=0
        ),
    ]
    f2 = GeneralizedForm(degree_m=0, weights_r=(2, 1), terms=f2_terms)

    return [f1, f2]


def get_sta_known_lyapunov(k1: float = 1.5, k2: float = 1.1) -> GeneralizedForm:
    """
    Get the known Moreno-Osorio Lyapunov function for STA.

    V(s,v) = (4k₂ + k₁²)|s| - 2k₁·sign(s)·√|s|·v + 2v²

    Args:
        k1, k2: Controller gains

    Returns:
        GeneralizedForm for Moreno-Osorio Lyapunov
    """
    p11 = 4 * k2 + k1 ** 2
    p12 = -k1
    p22 = 2.0

    terms = [
        # α₁|x₁|^1 (degree 2, since 2*1 = 2)
        GFTerm(
            coefficient=p11,
            exponents=(Fraction(1), Fraction(0)),
            term_types=('abs', 'abs')
        ),
        # α₁₂⌈x₁⌋^(1/2)x₂ (degree 2, since 2*(1/2) + 1*1 = 2)
        # Note: x₂ must be 'signed' type to represent actual x₂ (not |x₂|)
        # because ⌈x₂⌋^1 = sign(x₂)|x₂| = x₂
        GFTerm(
            coefficient=2 * p12,
            exponents=(Fraction(1, 2), Fraction(1)),
            term_types=('signed', 'signed')  # x₂ needs 'signed' to allow negative values
        ),
        # α₂|x₂|^2 (degree 2, since 1*2 = 2)
        GFTerm(
            coefficient=p22,
            exponents=(Fraction(0), Fraction(2)),
            term_types=('abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=terms)
