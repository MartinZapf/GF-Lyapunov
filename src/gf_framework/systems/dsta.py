from __future__ import annotations

"""
DSTA (Damped Super-Twisting Algorithm) System Definition.

NOVEL CONTROLLER — extends STA with velocity damping.

Combines Super-Twisting continuous feedback with Twisting relay damping:
    ẋ₁ = -k₁⌈x₁⌋^(1/2) + x₂
    ẋ₂ = -k₂⌈x₁⌋^0 - k₃⌈x₂⌋^0

This is r-homogeneous with:
    - weights r = (2, 1)
    - degree κ = -1
    - Lyapunov degree: m = 2

Stability requires k₂ > k₃ > 0 and k₁ > 0.
"""

from typing import List, Tuple
from fractions import Fraction
import numpy as np

from gf_framework.algebra.core import GeneralizedForm, GFTerm


def sgnpow(x: np.ndarray, p: float) -> np.ndarray:
    """Signed power: sign(x) * |x|^p"""
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


DSTA_WEIGHTS = (2, 1)
DSTA_KAPPA = -1.0
DSTA_DEGREE = 2

DSTA_DEFAULT_GAINS = {'k1': 1.5, 'k2': 1.1, 'k3': 0.5}


class DSTASystem:
    """
    Damped Super-Twisting Algorithm closed-loop system.

    Default gains: k₁ = 1.5, k₂ = 1.1, k₃ = 0.5  (k₂ > k₃ > 0).
    """

    def __init__(self, k1: float = 1.5, k2: float = 1.1, k3: float = 0.5):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        if not (k2 > k3 > 0):
            print(f"Warning: DSTA stability requires k₂ > k₃ > 0 (got k₂={k2}, k₃={k3})")

    @property
    def weights(self) -> Tuple[int, ...]:
        return DSTA_WEIGHTS

    @property
    def kappa(self) -> float:
        return DSTA_KAPPA

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        x1, x2 = x[:, 0], x[:, 1]
        d1 = -self.k1 * sgnpow(x1, 0.5) + x2
        d2 = -self.k2 * np.sign(x1) - self.k3 * np.sign(x2)
        return np.stack([d1, d2], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics for x₁ > 0 octant (sign(x₁)=+1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        x1, x2 = x[:, 0], x[:, 1]
        d1 = -self.k1 * sgnpow(x1, 0.5) + x2
        # Sign of x₂ remains as in nominal — switching surface is x₁=0 only
        d2 = -self.k2 * np.ones_like(x1) - self.k3 * np.sign(x2)
        return np.stack([d1, d2], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics for x₁ < 0 octant (sign(x₁)=-1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        x1, x2 = x[:, 0], x[:, 1]
        d1 = -self.k1 * sgnpow(x1, 0.5) + x2
        d2 = self.k2 * np.ones_like(x1) - self.k3 * np.sign(x2)
        return np.stack([d1, d2], axis=1).squeeze()


def create_dsta_dynamics(k1: float = 1.5, k2: float = 1.1, k3: float = 0.5):
    """Create (dynamics_plus, dynamics_minus) for numerical simulation and tests."""
    system = DSTASystem(k1, k2, k3)
    return system.dynamics_plus, system.dynamics_minus


def get_dsta_dynamics_gf(k1: float = 1.5, k2: float = 1.1,
                          k3: float = 0.5) -> List[GeneralizedForm]:
    """
    DSTA dynamics as Generalized Forms.

    f₁ = -k₁⌈x₁⌋^{1/2} + x₂        (weighted degree 1)
    f₂ = -k₂⌈x₁⌋^0 - k₃⌈x₂⌋^0      (weighted degree 0)
    """
    f1_terms = [
        GFTerm(coefficient=-k1, exponents=(Fraction(1, 2), Fraction(0)),
               term_types=('signed', 'abs')),
        GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(1)),
               term_types=('abs', 'signed')),
    ]
    f1 = GeneralizedForm(degree_m=1, weights_r=(2, 1), terms=f1_terms)

    f2_terms = [
        GFTerm(coefficient=-k2, exponents=(Fraction(0), Fraction(0)),
               term_types=('signed', 'abs')),
        GFTerm(coefficient=-k3, exponents=(Fraction(0), Fraction(0)),
               term_types=('abs', 'signed')),
    ]
    f2 = GeneralizedForm(degree_m=0, weights_r=(2, 1), terms=f2_terms)

    return [f1, f2]


def get_dsta_known_lyapunov(k1: float = 1.5, k2: float = 1.1,
                             k3: float = 0.5) -> GeneralizedForm:
    """
    Reference DSTA Lyapunov function (degree-2 r-homogeneous, r=(2,1)).

    Built by direct extension of the Moreno-Osorio STA Lyapunov, with
    the integral gain k₂ replaced by the damping-effective (k₂-k₃):

        V(x) = (4(k₂-k₃) + k₁²)|x₁| + 2k₁·sign(x₁)·√|x₁|·x₂ + 2|x₂|²

    Note: the original 2026 publication treats DSTA as a NOVEL controller
    with no closed-form reference; this expression is provided as a
    discovery seed only.
    """
    eff_k2 = max(k2 - k3, 0.1 * k2)
    p11 = 4 * eff_k2 + k1 ** 2
    p12 = k1
    p22 = 2.0

    terms = [
        GFTerm(coefficient=p11, exponents=(Fraction(1), Fraction(0)),
               term_types=('abs', 'abs')),
        GFTerm(coefficient=2 * p12, exponents=(Fraction(1, 2), Fraction(1)),
               term_types=('signed', 'signed')),
        GFTerm(coefficient=p22, exponents=(Fraction(0), Fraction(2)),
               term_types=('abs', 'abs')),
    ]
    return GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=terms)
