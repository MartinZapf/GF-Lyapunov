from __future__ import annotations

"""
Example 1 from Sanchez-Moreno 2019: Polynomial System (eq. 4).

This system demonstrates why the Generalized Forms extension is necessary:
NO strict polynomial Lyapunov function exists (Theorem 1 in paper).

System Definition (page 663, equation 4):
    ẋ₁ = -k₁x₁³ + x₂
    ẋ₂ = -k₂x₁⁵

Homogeneity Properties:
    - Weights: r = (1, 3)
    - Degree: κ = 2
    - Lyapunov degree: m = 6

Key Property:
    This system is ASYMPTOTICALLY STABLE for k₁, k₂ > 0, but there exists
    NO strict polynomial Lyapunov function of the form V(x) = Σᵢ αᵢ|xᵢ|^(m/rᵢ).
    The GF extension with signed cross-terms is REQUIRED.

Known Lyapunov (paper eq. 5):
    V(x) = α₁x₁⁶ - α₁₂x₁⌈x₂⌋^(5/3) + α₂x₂²

    Note the signed power ⌈x₂⌋^(5/3) in the cross-term.

Reference:
    Sanchez, T., & Moreno, J. A. (2019). Design of Lyapunov functions for a
    class of homogeneous systems: Generalized forms approach.
    Int J Robust Nonlinear Control, 29(3), 661-681, Section 5.1.
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


class Example1System:
    """
    Polynomial system from Sanchez-Moreno 2019, Example 1.

    ẋ₁ = -k₁x₁³ + x₂
    ẋ₂ = -k₂x₁⁵

    This is r-homogeneous with r=(1,3), κ=2.
    """

    def __init__(self, k1: float = 1.0, k2: float = 1.0):
        """
        Initialize Example 1 system.

        Args:
            k1, k2: Positive controller gains (stability requires k1, k2 > 0)
        """
        self.k1 = k1
        self.k2 = k2

        if k1 <= 0 or k2 <= 0:
            print(f"Warning: Gains must be positive for stability (k1={k1}, k2={k2})")

    @property
    def weights(self) -> Tuple[int, ...]:
        return (1, 3)

    @property
    def kappa(self) -> float:
        return 2.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics.

        ẋ₁ = -k₁x₁³ + x₂
        ẋ₂ = -k₂x₁⁵

        Args:
            x: State tensor [x₁, x₂] of shape (batch, 2) or (2,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2 = x[:, 0], x[:, 1]

        dx1 = -self.k1 * x1 ** 3 + x2
        dx2 = -self.k2 * x1 ** 5

        return np.stack([dx1, dx2], axis=1).squeeze()


# Default configuration
EXAMPLE1_WEIGHTS = (1, 3)
EXAMPLE1_KAPPA = 2
EXAMPLE1_DEGREE = 6


def get_example1_dynamics_gf(k1: float = 1.0, k2: float = 1.0) -> List[GeneralizedForm]:
    """
    Get Example 1 dynamics as Generalized Forms.

    For Example 1:
        f₁ = ẋ₁ = -k₁x₁³ + x₂
        f₂ = ẋ₂ = -k₂x₁⁵

    Weighted degrees (κ + rᵢ):
        f₁: κ + r₁ = 2 + 1 = 3  →  terms: x₁³ (deg=3), x₂ (deg=3)
        f₂: κ + r₂ = 2 + 3 = 5  →  terms: x₁⁵ (deg=5)

    Args:
        k1, k2: Controller gains

    Returns:
        List of GFs [f₁, f₂]
    """
    weights_r = EXAMPLE1_WEIGHTS

    # f₁ = -k₁x₁³ + x₂
    # Term 1: -k₁x₁³, weighted degree: 1*3 = 3
    # Term 2: x₂, weighted degree: 3*1 = 3 (x₂ needs 'signed' to allow negative)
    f1_terms = [
        GFTerm(
            coefficient=-k1,
            exponents=(Fraction(3), Fraction(0)),
            term_types=('signed', 'abs')  # x₁³ preserves sign
        ),
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1)),
            term_types=('abs', 'signed')  # x₂ needs 'signed' to allow negative values
        ),
    ]
    f1 = GeneralizedForm(degree_m=3, weights_r=weights_r, terms=f1_terms)

    # f₂ = -k₂x₁⁵
    # Weighted degree: 1*5 = 5
    f2_terms = [
        GFTerm(
            coefficient=-k2,
            exponents=(Fraction(5), Fraction(0)),
            term_types=('signed', 'abs')  # x₁⁵ preserves sign
        ),
    ]
    f2 = GeneralizedForm(degree_m=5, weights_r=weights_r, terms=f2_terms)

    return [f1, f2]


def get_example1_known_lyapunov(
    alpha1: float = 1.0,
    alpha12: float = 1.0,
    alpha2: float = 1.0
) -> GeneralizedForm:
    """
    Get the known Lyapunov function for Example 1.

    V(x) = α₁x₁⁶ - α₁₂x₁⌈x₂⌋^(5/3) + α₂x₂²

    From paper equation (5). Uses signed power ⌈x₂⌋^(5/3) for cross-term.

    Note: The cross-term has a NEGATIVE coefficient in the paper form
    (the minus sign is explicit in the formula). The actual coefficient
    values depend on the gains and can be found via SOS optimization.

    For default gains k₁=k₂=1, a valid solution exists with:
        α₁ > 0, α₂ > 0, and α₁₂ satisfying certain LMI conditions.

    Args:
        alpha1: Coefficient for x₁⁶ term (must be > 0)
        alpha12: Coefficient magnitude for cross-term
        alpha2: Coefficient for x₂² term (must be > 0)

    Returns:
        GeneralizedForm for Example 1 Lyapunov
    """
    weights_r = EXAMPLE1_WEIGHTS
    degree_m = EXAMPLE1_DEGREE

    # All terms have weighted degree m = 6
    # x₁⁶: 1*6 = 6 ✓
    # x₁·⌈x₂⌋^(5/3): 1*1 + 3*(5/3) = 1 + 5 = 6 ✓
    # x₂²: 3*2 = 6 ✓
    terms = [
        # α₁|x₁|^6 (we use |x₁|^6 = x₁⁶ since even power)
        GFTerm(
            coefficient=alpha1,
            exponents=(Fraction(6), Fraction(0)),
            term_types=('abs', 'abs')
        ),
        # -α₁₂·x₁·⌈x₂⌋^(5/3)
        # Note: x₁ = ⌈x₁⌋^1 (signed with exp=1)
        # The overall term x₁·⌈x₂⌋^(5/3) has both variables signed
        GFTerm(
            coefficient=-alpha12,  # Negative per paper formula
            exponents=(Fraction(1), Fraction(5, 3)),
            term_types=('signed', 'signed')  # Both preserve sign
        ),
        # α₂|x₂|^2 = α₂x₂² (even power)
        GFTerm(
            coefficient=alpha2,
            exponents=(Fraction(0), Fraction(2)),
            term_types=('abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


def no_polynomial_lyapunov_motivation():
    """
    Document why no polynomial Lyapunov exists for Example 1.

    From Sanchez-Moreno 2019, Theorem 1 (paraphrased):

    For the polynomial system:
        ẋ₁ = -k₁x₁³ + x₂
        ẋ₂ = -k₂x₁⁵

    There exists NO polynomial function V(x) of the form:
        V(x) = Σᵢ αᵢ|xᵢ|^(m/rᵢ)  (diagonal only)

    that is both:
        1. Positive definite (V > 0 for x ≠ 0)
        2. Has strictly negative derivative (V̇ < 0)

    The obstruction is that the cross-term structure in the dynamics
    prevents finding positive diagonal coefficients that also make
    V̇ negative definite.

    The SOLUTION is to allow cross-terms in V with signed powers,
    leading to the GF Lyapunov:
        V(x) = α₁x₁⁶ - α₁₂x₁⌈x₂⌋^(5/3) + α₂x₂²

    Returns:
        str: Explanation of why polynomial Lyapunov doesn't exist
    """
    explanation = """
    Example 1 (Sanchez-Moreno 2019) demonstrates a key limitation:

    System: ẋ₁ = -k₁x₁³ + x₂,  ẋ₂ = -k₂x₁⁵

    Attempt 1: Diagonal polynomial V = α₁x₁⁶ + α₂x₂²

    Compute V̇:
        V̇ = 6α₁x₁⁵·ẋ₁ + 2α₂x₂·ẋ₂
          = 6α₁x₁⁵(-k₁x₁³ + x₂) + 2α₂x₂(-k₂x₁⁵)
          = -6α₁k₁x₁⁸ + 6α₁x₁⁵x₂ - 2α₂k₂x₁⁵x₂
          = -6α₁k₁x₁⁸ + (6α₁ - 2α₂k₂)x₁⁵x₂

    For V̇ < 0, we need:
        1. -6α₁k₁x₁⁸ < 0  →  α₁ > 0, k₁ > 0 ✓
        2. The cross-term (6α₁ - 2α₂k₂)x₁⁵x₂ must not dominate

    PROBLEM: The cross-term x₁⁵x₂ has indefinite sign and CANNOT be
    eliminated by any choice of positive α₁, α₂.

    SOLUTION: Add a cross-term to V that cancels the problematic term:
        V = α₁x₁⁶ - α₁₂x₁⌈x₂⌋^(5/3) + α₂x₂²

    This introduces additional terms in V̇ that can be balanced to
    achieve V̇ < 0 for appropriate coefficient choices.
    """
    return explanation
