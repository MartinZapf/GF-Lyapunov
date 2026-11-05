from __future__ import annotations

"""
Twisting Algorithm System Definition.

From Levant (1993):
    ẋ₁ = x₂
    ẋ₂ = -r₁·sign(x₁) - r₂·sign(x₂)

This is r-homogeneous with:
    - weights r = (2, 1)
    - degree κ = -1
    - Lyapunov degree: m = 2

The twisting algorithm is a second-order sliding mode controller that
achieves finite-time convergence with bounded control.

Stability conditions (for bounded disturbance |d| ≤ C):
    - Km(r₁ + r₂) - C > KM(r₁ - r₂) + C
    - Km(r₁ - r₂) > C

where Km, KM are bounds on the control effectiveness.

For nominal case (Km = KM = 1, C = 0), any r₁ > r₂ > 0 works.

Known Lyapunov function (Moreno structure):
    V(x) = p₁₁|x₁| + 2p₁₂·sign(x₁)·x₂ + p₂₂·|x₂|²

References:
    - Levant, A. (1993). "Sliding order and sliding accuracy in sliding mode control"
    - Moreno, J. A., & Osorio, M. (2012). "Strict Lyapunov Functions for
      the Super-Twisting Algorithm"
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


class TwistingSystem:
    """
    Twisting Algorithm closed-loop system.

    ẋ₁ = x₂
    ẋ₂ = -r₁·sign(x₁) - r₂·sign(x₂)

    Default gains: r₁ = 2.0, r₂ = 1.5 (satisfy r₁ > r₂ > 0)
    """

    def __init__(
        self,
        r1: float = 2.0,
        r2: float = 1.5,
    ):
        """
        Initialize Twisting system.

        Args:
            r1, r2: Controller gains (stability requires r₁ > r₂ > 0)
        """
        self.r1 = r1
        self.r2 = r2

        # Check stability condition for nominal case
        if not (r1 > r2 > 0):
            print(f"Warning: Gains may not satisfy stability (need r1 > r2 > 0, got r1={r1}, r2={r2})")

    @property
    def weights(self) -> Tuple[int, ...]:
        return (2, 1)

    @property
    def kappa(self) -> float:
        return -1.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics.

        ẋ₁ = x₂
        ẋ₂ = -r₁·sign(x₁) - r₂·sign(x₂)

        Args:
            x: State tensor [x₁, x₂] of shape (batch, 2) or (2,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2 = x[:, 0], x[:, 1]

        dx1 = x2
        dx2 = -self.r1 * np.sign(x1) - self.r2 * np.sign(x2)

        return np.stack([dx1, dx2], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when x₁ > 0 (sign(x₁) = +1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2 = x[:, 0], x[:, 1]

        dx1 = x2
        dx2 = -self.r1 * np.ones_like(x1) - self.r2 * np.sign(x2)

        return np.stack([dx1, dx2], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when x₁ < 0 (sign(x₁) = -1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2 = x[:, 0], x[:, 1]

        dx1 = x2
        dx2 = self.r1 * np.ones_like(x1) - self.r2 * np.sign(x2)

        return np.stack([dx1, dx2], axis=1).squeeze()


def create_twisting_dynamics(r1: float = 2.0, r2: float = 1.5):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = TwistingSystem(r1, r2)
    return system.dynamics_plus, system.dynamics_minus


# Default configuration
TWISTING_WEIGHTS = (2, 1)
TWISTING_KAPPA = -1.0
TWISTING_DEGREE = 2

# Known Lyapunov function structure (similar to STA):
# V(x₁, x₂) = p₁₁|x₁| + 2p₁₂·sign(x₁)·x₂ + p₂₂·|x₂|²
#
# In GF notation with m=2, r=(2,1):
# V(x₁,x₂) = α₁|x₁|^1 + α₁₂⌈x₁⌋^0·x₂ + α₂|x₂|^2


def get_twisting_dynamics_gf(r1: float = 2.0, r2: float = 1.5) -> List[GeneralizedForm]:
    """
    Get Twisting dynamics as Generalized Forms.

    For Twisting:
        f₁ = ẋ₁ = x₂
        f₂ = ẋ₂ = -r₁·sign(x₁) - r₂·sign(x₂)

    Weighted degrees (κ + rᵢ):
        f₁: κ + r₁ = -1 + 2 = 1  →  term x₂ has weighted degree 1
        f₂: κ + r₂ = -1 + 1 = 0  →  sign terms have weighted degree 0

    Args:
        r1, r2: Controller gains

    Returns:
        List of GFs [f₁, f₂]
    """
    # f₁ = x₂
    # Weighted degree of x₂: r₂ * 1 = 1 * 1 = 1
    f1_terms = [
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1)),
            term_types=('abs', 'signed')  # x₂ needs 'signed' to allow negative values
        ),
    ]
    f1 = GeneralizedForm(degree_m=1, weights_r=(2, 1), terms=f1_terms)

    # f₂ = -r₁·sign(x₁) - r₂·sign(x₂)
    # Term 1: -r₁·sign(x₁) = -r₁⌈x₁⌋^0, weighted degree: 2*0 = 0
    # Term 2: -r₂·sign(x₂) = -r₂⌈x₂⌋^0, weighted degree: 1*0 = 0
    f2_terms = [
        GFTerm(
            coefficient=-r1,
            exponents=(Fraction(0), Fraction(0)),
            term_types=('signed', 'abs')  # sign(x₁) comes from 'signed' type with exp=0
        ),
        GFTerm(
            coefficient=-r2,
            exponents=(Fraction(0), Fraction(0)),
            term_types=('abs', 'signed')  # sign(x₂) comes from 'signed' type with exp=0
        ),
    ]
    f2 = GeneralizedForm(degree_m=0, weights_r=(2, 1), terms=f2_terms)

    return [f1, f2]


def get_twisting_lyapunov_seed(r1: float = 2.0, r2: float = 1.5) -> GeneralizedForm:
    """Return a homogeneous discovery seed, NOT a verified Lyapunov function.

    V = (2*r1+r2**2)|x1| + r2*signed_sqrt(x1)*x2 + x2**2.
    Positive definiteness does not imply decay: at default gains and x=(1,10),
    Vdot=62.25 > 0. The candidate needs independent discovery/verification.
    """
    p11 = 2 * r1 + r2 ** 2  # Ensures positive definiteness
    p12 = r2 / 2
    p22 = 1.0

    terms = [
        # p₁₁|x₁|^1 (degree 2, since 2*1 = 2)
        GFTerm(
            coefficient=p11,
            exponents=(Fraction(1), Fraction(0)),
            term_types=('abs', 'abs')
        ),
        # Homogeneous cross term: 2p12*signed_sqrt(x1)*x2.
        GFTerm(
            coefficient=2 * p12,
            exponents=(Fraction(1, 2), Fraction(1)),
            term_types=('signed', 'signed')
        ),
        # p₂₂|x₂|^2 (degree 2, since 1*2 = 2)
        GFTerm(
            coefficient=p22,
            exponents=(Fraction(0), Fraction(2)),
            term_types=('abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=2, weights_r=(2, 1), terms=terms)


def get_twisting_known_lyapunov(r1: float = 2.0, r2: float = 1.5) -> GeneralizedForm:
    """Deprecated compatibility name for an unverified discovery seed."""
    import warnings
    warnings.warn(
        "get_twisting_known_lyapunov returns an unverified seed, not a Lyapunov "
        "certificate; use get_twisting_lyapunov_seed",
        DeprecationWarning, stacklevel=2,
    )
    return get_twisting_lyapunov_seed(r1, r2)


def verify_homogeneity():
    """
    Verify the homogeneity property of the Twisting algorithm.

    For an r-homogeneous system with weights r = (r₁, r₂) and degree κ:
        fᵢ(Λ_ε^r x) = ε^(κ+rᵢ) fᵢ(x)

    where Λ_ε^r = diag(ε^r₁, ε^r₂).

    For Twisting with r = (2, 1), κ = -1:
        f₁(ε²x₁, εx₂) = εx₂ = ε^(−1+2) f₁(x) = ε¹ f₁(x) ✓
        f₂(ε²x₁, εx₂) = −r₁·sign(ε²x₁) − r₂·sign(εx₂)
                       = −r₁·sign(x₁) − r₂·sign(x₂) (sign is scale-invariant)
                       = ε^(−1+1) f₂(x) = ε⁰ f₂(x) ✓
    """
    explanation = """
    Twisting Algorithm Homogeneity Verification:

    Weights: r = (2, 1)
    Degree: κ = -1 (finite-time stable)

    Dilation: Λ_ε^r x = (ε²x₁, εx₂)

    f₁(Λ_ε^r x) = εx₂ = ε^(κ+r₁) x₂ = ε^1 · x₂ ✓
    f₂(Λ_ε^r x) = -r₁·sign(ε²x₁) - r₂·sign(εx₂)
                = -r₁·sign(x₁) - r₂·sign(x₂)  (sign is homogeneous of degree 0)
                = ε^0 · f₂(x) = ε^(κ+r₂) · f₂(x) ✓

    The system is r-homogeneous of degree κ = -1.
    Since κ < 0, the system is FINITE-TIME STABLE.
    """
    return explanation
