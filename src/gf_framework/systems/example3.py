from __future__ import annotations

"""
Example 3 from Sanchez-Moreno 2019: Double Integrator with
Continuous Homogeneous Feedback (eq. 7).

This system demonstrates FINITE-TIME STABILITY with κ = -1 < 0.

System Definition (page 663, equation 7):
    ẋ₁ = x₂
    ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2)

Homogeneity Properties:
    - Weights: r = (3, 2)
    - Degree: κ = -1 (NEGATIVE → finite-time stable!)
    - Lyapunov degree: m = 5

Key Property:
    Since κ < 0, this system is FINITE-TIME STABLE (not just asymptotically).
    The convergence rate is: V̇ ≤ -c·V^((m+κ)/m) = -c·V^(4/5)

Known Lyapunov (paper eq. 8):
    V(x) = α₁|x₁|^(5/3) + α₁₂x₁x₂ + α₂|x₂|^(5/2)

    Specific coefficients from Algorithm 6: k₁ ≈ 1.8, k₂ ≈ 4.2

Reference:
    Sanchez, T., & Moreno, J. A. (2019). Design of Lyapunov functions for a
    class of homogeneous systems: Generalized forms approach.
    Int J Robust Nonlinear Control, 29(3), 661-681, Section 5.3.
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


class Example3System:
    """
    Double integrator with continuous homogeneous feedback.

    ẋ₁ = x₂
    ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2)

    This is r-homogeneous with r=(3,2), κ=-1.

    FINITE-TIME STABLE since κ < 0.
    """

    def __init__(self, k1: float = 1.8, k2: float = 4.2):
        """
        Initialize Example 3 system.

        Default gains k₁≈1.8, k₂≈4.2 are from Algorithm 6 in the paper.

        Args:
            k1, k2: Positive controller gains (stability requires specific conditions)
        """
        self.k1 = k1
        self.k2 = k2

        if k1 <= 0 or k2 <= 0:
            print(f"Warning: Gains must be positive for stability (k1={k1}, k2={k2})")

    @property
    def weights(self) -> Tuple[int, ...]:
        return (3, 2)

    @property
    def kappa(self) -> float:
        return -1.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics.

        ẋ₁ = x₂
        ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2)

        Args:
            x: State tensor [x₁, x₂] of shape (batch, 2) or (2,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2 = x[:, 0], x[:, 1]

        dx1 = x2
        dx2 = -self.k1 * sgnpow(x1, 1/3) - self.k2 * sgnpow(x2, 1/2)

        return np.stack([dx1, dx2], axis=1).squeeze()


# Default configuration
EXAMPLE3_WEIGHTS = (3, 2)
EXAMPLE3_KAPPA = -1
EXAMPLE3_DEGREE = 5


def get_example3_dynamics_gf(k1: float = 1.8, k2: float = 4.2) -> List[GeneralizedForm]:
    """
    Get Example 3 dynamics as Generalized Forms.

    For Example 3:
        f₁ = ẋ₁ = x₂
        f₂ = ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2)

    Weighted degrees (κ + rᵢ):
        f₁: κ + r₁ = -1 + 3 = 2  →  term x₂ has weighted degree 2
        f₂: κ + r₂ = -1 + 2 = 1  →  terms have weighted degree 1

    Args:
        k1, k2: Controller gains

    Returns:
        List of GFs [f₁, f₂]
    """
    weights_r = EXAMPLE3_WEIGHTS

    # f₁ = x₂
    # Weighted degree: 2*1 = 2
    f1_terms = [
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1)),
            term_types=('abs', 'signed')  # x₂ needs 'signed' to allow negative values
        ),
    ]
    f1 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=f1_terms)

    # f₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2)
    # Term 1: -k₁⌈x₁⌋^(1/3), weighted degree: 3*(1/3) = 1
    # Term 2: -k₂⌈x₂⌋^(1/2), weighted degree: 2*(1/2) = 1
    f2_terms = [
        GFTerm(
            coefficient=-k1,
            exponents=(Fraction(1, 3), Fraction(0)),
            term_types=('signed', 'abs')
        ),
        GFTerm(
            coefficient=-k2,
            exponents=(Fraction(0), Fraction(1, 2)),
            term_types=('abs', 'signed')
        ),
    ]
    f2 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=f2_terms)

    return [f1, f2]


def get_example3_known_lyapunov(
    alpha1: float = 7.0,
    alpha12: float = 1.0,
    alpha2: float = 5.0
) -> GeneralizedForm:
    """
    Get the known Lyapunov function for Example 3.

    V(x) = α₁|x₁|^(5/3) + α₁₂x₁x₂ + α₂|x₂|^(5/2)

    From paper equation (8).

    The default coefficients (7, 1, 5) provide a positive definite V
    for the default gains k₁≈1.8, k₂≈4.2. The exact values come from
    the LMI solution in Algorithm 6.

    Args:
        alpha1: Coefficient for |x₁|^(5/3) term
        alpha12: Coefficient for x₁x₂ cross-term
        alpha2: Coefficient for |x₂|^(5/2) term

    Returns:
        GeneralizedForm for Example 3 Lyapunov
    """
    weights_r = EXAMPLE3_WEIGHTS
    degree_m = EXAMPLE3_DEGREE

    # All terms have weighted degree m = 5
    # |x₁|^(5/3): 3*(5/3) = 5 ✓
    # x₁x₂: 3*1 + 2*1 = 5 ✓
    # |x₂|^(5/2): 2*(5/2) = 5 ✓
    terms = [
        # α₁|x₁|^(5/3)
        GFTerm(
            coefficient=alpha1,
            exponents=(Fraction(5, 3), Fraction(0)),
            term_types=('abs', 'abs')
        ),
        # α₁₂·x₁·x₂
        # x₁ = ⌈x₁⌋^1 and x₂ = ⌈x₂⌋^1 (both signed with odd exponent)
        GFTerm(
            coefficient=alpha12,
            exponents=(Fraction(1), Fraction(1)),
            term_types=('signed', 'signed')  # x₁x₂ preserves combined sign
        ),
        # α₂|x₂|^(5/2)
        GFTerm(
            coefficient=alpha2,
            exponents=(Fraction(0), Fraction(5, 2)),
            term_types=('abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


def verify_finite_time_property():
    """
    Document the finite-time stability property of Example 3.

    For a homogeneous system with degree κ < 0:

        V̇ ≤ -c·W(x)

    where W is a positive definite function of degree m+κ.

    Since V has degree m and W has degree m+κ:
        V̇ ≤ -c·V^((m+κ)/m)

    For Example 3: m=5, κ=-1
        V̇ ≤ -c·V^(4/5)

    This sub-linear decay rate (exponent < 1) implies FINITE-TIME convergence.

    Settling time bound (Bhat & Bernstein 2000):
        T ≤ V(x₀)^(1/5) / (c · (1 - 4/5)) = 5·V(x₀)^(1/5) / c

    Returns:
        str: Explanation of finite-time stability property
    """
    explanation = """
    Example 3 demonstrates FINITE-TIME STABILITY (κ = -1 < 0):

    System: ẋ₁ = x₂,  ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2)

    Lyapunov properties:
        V(x) = α₁|x₁|^(5/3) + α₁₂x₁x₂ + α₂|x₂|^(5/2)
        degree m = 5

    Key result for homogeneous systems with κ < 0:
        V̇ ≤ -c·V^((m+κ)/m) = -c·V^(4/5)

    This decay rate V̇ ~ -V^(4/5) (exponent < 1) implies:
        • Trajectories reach origin in FINITE TIME
        • Not just asymptotic convergence

    Settling time bound:
        T ≤ V(x₀)^(1/5) / (c · (1 - 4/5)) = 5·V(x₀)^(1/5) / c

    Comparison:
        • Asymptotic (κ=0): V̇ ~ -V    → t → ∞
        • Finite-time (κ<0): V̇ ~ -V^p (p<1) → t = T_finite

    This is why the double integrator with signed power feedback
    achieves finite-time regulation, used in sliding mode control.
    """
    return explanation
