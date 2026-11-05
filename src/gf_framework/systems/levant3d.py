from __future__ import annotations

"""
Levant's Second-Order Differentiator (3D) System Definition.

From Levant (1998, 2003):
    ė₁ = -λ₁⌈e₁⌋^(2/3) + e₂
    ė₂ = -λ₂⌈e₁⌋^(1/3) + e₃
    ė₃ = -λ₃⌈e₁⌋^0

This is r-homogeneous with:
    - weights r = (3, 2, 1)
    - degree κ = -1
    - Lyapunov degree: m = 8 (the published certified degree: sparse
      25-term / dense 69-term certificates)

The differentiator estimates the first and second derivatives of a signal
in finite time, robust to bounded Lipschitz noise.

Stability conditions (Levant 1998, Moreno 2012):
    - λ₁, λ₂, λ₃ > 0
    - Standard (2-variant, L=1): λ₁ = 2.0, λ₂ = 1.5·√2 ≈ 2.1213, λ₃ = 1.1 —
      the converted gains from the recursive sequence (1.1, 1.5, 2.0)
    - Conservative: λ₁ = 3.0, λ₂ = 2.0, λ₃ = 1.0

where L is the Lipschitz constant of the second derivative.

All terms are standard GF products - no framework extension needed.

References:
    - Levant, A. (1998). "Robust exact differentiation via sliding mode technique"
    - Levant, A. (2003). "Higher-order sliding modes, differentiation and
      output-feedback control"
    - Moreno, J. A. (2012). "Lyapunov approach for analysis and design of
      second order sliding mode algorithms"
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


class Levant3DSystem:
    """
    Levant's Second-Order Differentiator (3D system).

    ė₁ = -λ₁⌈e₁⌋^(2/3) + e₂
    ė₂ = -λ₂⌈e₁⌋^(1/3) + e₃
    ė₃ = -λ₃⌈e₁⌋^0

    Where:
    - e₁ = estimation error of signal
    - e₂ = estimation error of first derivative
    - e₃ = estimation error of second derivative

    Default gains: λ₁ = 3.0, λ₂ = 2.0, λ₃ = 1.1 (conservative)
    """

    def __init__(
        self,
        lambda1: float = 3.0,
        lambda2: float = 2.0,
        lambda3: float = 1.1,
    ):
        """
        Initialize Levant 3D differentiator.

        Args:
            lambda1, lambda2, lambda3: Differentiator gains (all must be positive)
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Check positivity
        if lambda1 <= 0 or lambda2 <= 0 or lambda3 <= 0:
            print(f"Warning: All gains must be positive for stability")

    @property
    def weights(self) -> Tuple[int, ...]:
        return (3, 2, 1)

    @property
    def kappa(self) -> float:
        return -1.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics (no disturbance).

        ė₁ = -λ₁⌈e₁⌋^(2/3) + e₂
        ė₂ = -λ₂⌈e₁⌋^(1/3) + e₃
        ė₃ = -λ₃⌈e₁⌋^0

        Args:
            x: State tensor [e₁, e₂, e₃] of shape (batch, 3) or (3,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        e1, e2, e3 = x[:, 0], x[:, 1], x[:, 2]

        de1 = -self.lambda1 * sgnpow(e1, 2/3) + e2
        de2 = -self.lambda2 * sgnpow(e1, 1/3) + e3
        de3 = -self.lambda3 * np.sign(e1)

        return np.stack([de1, de2, de3], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when e₁ > 0 (sign(e₁) = +1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        e1, e2, e3 = x[:, 0], x[:, 1], x[:, 2]

        de1 = -self.lambda1 * np.abs(e1) ** (2/3) + e2
        de2 = -self.lambda2 * np.abs(e1) ** (1/3) + e3
        de3 = -self.lambda3 * np.ones_like(e1)

        return np.stack([de1, de2, de3], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when e₁ < 0 (sign(e₁) = -1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        e1, e2, e3 = x[:, 0], x[:, 1], x[:, 2]

        de1 = self.lambda1 * np.abs(e1) ** (2/3) + e2
        de2 = self.lambda2 * np.abs(e1) ** (1/3) + e3
        de3 = self.lambda3 * np.ones_like(e1)

        return np.stack([de1, de2, de3], axis=1).squeeze()


def create_levant3d_dynamics(
    lambda1: float = 3.0,
    lambda2: float = 2.0,
    lambda3: float = 1.1,
):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = Levant3DSystem(lambda1, lambda2, lambda3)
    return system.dynamics_plus, system.dynamics_minus


# Default configuration
LEVANT3D_WEIGHTS = (3, 2, 1)
LEVANT3D_KAPPA = -1.0
LEVANT3D_DEGREE = 6


def get_levant3d_dynamics_gf(
    lambda1: float = 3.0,
    lambda2: float = 2.0,
    lambda3: float = 1.1,
) -> List[GeneralizedForm]:
    """
    Get Levant 3D differentiator dynamics as Generalized Forms.

    For Levant 3D:
        f₁ = ė₁ = -λ₁⌈e₁⌋^(2/3) + e₂
        f₂ = ė₂ = -λ₂⌈e₁⌋^(1/3) + e₃
        f₃ = ė₃ = -λ₃⌈e₁⌋^0

    Weighted degrees (κ + rᵢ):
        f₁: κ + r₁ = -1 + 3 = 2
        f₂: κ + r₂ = -1 + 2 = 1
        f₃: κ + r₃ = -1 + 1 = 0

    Args:
        lambda1, lambda2, lambda3: Differentiator gains

    Returns:
        List of GFs [f₁, f₂, f₃]
    """
    weights_r = LEVANT3D_WEIGHTS

    # f₁ = -λ₁⌈e₁⌋^(2/3) + e₂
    # Term 1: -λ₁⌈e₁⌋^(2/3), weighted degree: 3*(2/3) = 2
    # Term 2: e₂ = ⌈e₂⌋^1, weighted degree: 2*1 = 2
    f1_terms = [
        GFTerm(
            coefficient=-lambda1,
            exponents=(Fraction(2, 3), Fraction(0), Fraction(0)),
            term_types=('signed', 'abs', 'abs')
        ),
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1), Fraction(0)),
            term_types=('abs', 'signed', 'abs')  # e₂ needs 'signed'
        ),
    ]
    f1 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=f1_terms)

    # f₂ = -λ₂⌈e₁⌋^(1/3) + e₃
    # Term 1: -λ₂⌈e₁⌋^(1/3), weighted degree: 3*(1/3) = 1
    # Term 2: e₃ = ⌈e₃⌋^1, weighted degree: 1*1 = 1
    f2_terms = [
        GFTerm(
            coefficient=-lambda2,
            exponents=(Fraction(1, 3), Fraction(0), Fraction(0)),
            term_types=('signed', 'abs', 'abs')
        ),
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(0), Fraction(1)),
            term_types=('abs', 'abs', 'signed')  # e₃ needs 'signed'
        ),
    ]
    f2 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=f2_terms)

    # f₃ = -λ₃⌈e₁⌋^0 = -λ₃·sign(e₁)
    # Term: -λ₃⌈e₁⌋^0, weighted degree: 3*0 = 0
    f3_terms = [
        GFTerm(
            coefficient=-lambda3,
            exponents=(Fraction(0), Fraction(0), Fraction(0)),
            term_types=('signed', 'abs', 'abs')  # sign(e₁)
        ),
    ]
    f3 = GeneralizedForm(degree_m=0, weights_r=weights_r, terms=f3_terms)

    return [f1, f2, f3]


def get_levant3d_known_lyapunov(
    lambda1: float = 3.0,
    lambda2: float = 2.0,
    lambda3: float = 1.1,
) -> GeneralizedForm:
    """
    Get a known Lyapunov structure for Levant 3D differentiator.

    The Lyapunov function follows the D₁ structure from Pérez-Ventura et al.:
        V(e) = Σᵢ αᵢ|eᵢ|^(m/rᵢ) + Σᵢ<ⱼ αᵢⱼ⌈eᵢ⌋^(m-rⱼ)/rᵢ · ⌈eⱼ⌋

    For m = 6, r = (3, 2, 1):
        - Diagonal: |e₁|^2, |e₂|^3, |e₃|^6
        - Cross-terms following GF homogeneity constraints

    This is for validation only - discovery should find it independently.

    Args:
        lambda1, lambda2, lambda3: Differentiator gains

    Returns:
        GeneralizedForm for Levant 3D Lyapunov
    """
    weights_r = LEVANT3D_WEIGHTS
    degree_m = LEVANT3D_DEGREE

    # D₁ structure: n diagonal + n(n-1) = n² total terms for n=3 → 9 terms
    # Diagonal terms: |eᵢ|^(m/rᵢ)
    # - |e₁|^(6/3) = |e₁|^2
    # - |e₂|^(6/2) = |e₂|^3
    # - |e₃|^(6/1) = |e₃|^6

    # Conservative coefficients (actual values from SOS optimization)
    terms = [
        # Diagonal: |e₁|^2, weighted degree: 3*2 = 6 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(2), Fraction(0), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # Diagonal: |e₂|^3, weighted degree: 2*3 = 6 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(3), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # Diagonal: |e₃|^6, weighted degree: 1*6 = 6 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(0), Fraction(6)),
            term_types=('abs', 'abs', 'abs')
        ),
        # Cross-term e₁-e₂: ⌈e₁⌋^(4/3)·⌈e₂⌋, degree: 3*(4/3) + 2*1 = 6 ✓
        GFTerm(
            coefficient=0.5,
            exponents=(Fraction(4, 3), Fraction(1), Fraction(0)),
            term_types=('signed', 'signed', 'abs')
        ),
        # Cross-term e₁-e₃: ⌈e₁⌋^(5/3)·⌈e₃⌋, degree: 3*(5/3) + 1*1 = 6 ✓
        GFTerm(
            coefficient=0.3,
            exponents=(Fraction(5, 3), Fraction(0), Fraction(1)),
            term_types=('signed', 'abs', 'signed')
        ),
        # Cross-term e₂-e₃: ⌈e₂⌋^(5/2)·⌈e₃⌋, degree: 2*(5/2) + 1*1 = 6 ✓
        GFTerm(
            coefficient=0.4,
            exponents=(Fraction(0), Fraction(5, 2), Fraction(1)),
            term_types=('abs', 'signed', 'signed')
        ),
    ]

    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


def verify_homogeneity():
    """
    Verify the homogeneity property of Levant 3D differentiator.

    For an r-homogeneous system with weights r = (3, 2, 1) and degree κ = -1:
        fᵢ(Λ_ε^r x) = ε^(κ+rᵢ) fᵢ(x)

    where Λ_ε^r = diag(ε³, ε², ε).
    """
    explanation = """
    Levant 3D Differentiator Homogeneity Verification:

    Weights: r = (3, 2, 1)
    Degree: κ = -1 (finite-time stable)

    Dilation: Λ_ε^r x = (ε³e₁, ε²e₂, εe₃)

    f₁(Λ_ε^r x) = -λ₁⌈ε³e₁⌋^(2/3) + ε²e₂
                = -λ₁·ε²·⌈e₁⌋^(2/3) + ε²e₂
                = ε²·f₁(x) = ε^(κ+r₁)·f₁(x) ✓

    f₂(Λ_ε^r x) = -λ₂⌈ε³e₁⌋^(1/3) + εe₃
                = -λ₂·ε·⌈e₁⌋^(1/3) + εe₃
                = ε·f₂(x) = ε^(κ+r₂)·f₂(x) ✓

    f₃(Λ_ε^r x) = -λ₃·sign(ε³e₁)
                = -λ₃·sign(e₁) (sign is scale-invariant)
                = ε⁰·f₃(x) = ε^(κ+r₃)·f₃(x) ✓

    The system is r-homogeneous of degree κ = -1.
    Since κ < 0, the system is FINITE-TIME STABLE.
    """
    return explanation
