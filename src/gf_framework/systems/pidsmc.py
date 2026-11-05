from __future__ import annotations

"""
PID-SMC (PID-like Continuous Sliding Mode Controller) System Definition.

From Pérez-Ventura et al. (2021):
    ẋ₁ = x₂
    ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃ + disturbance
    ẋ₃ = -k₃⌈x₁⌋^0  (discontinuous integral)

This is r-homogeneous with:
    - weights r = (3, 2, 1)
    - degree κ = -1
    - Expected Lyapunov degree: m = 7

The known Lyapunov function (Pérez-Ventura) has 9 terms.

Relationship to Sanchez-Moreno 2019:
------------------------------------
This system is a 3D extension of the second-order differentiator discussed in
Example 2 (eq. 6) of Sanchez & Moreno (2019), Int J Robust Nonlinear Control.
The differentiator has the same r=(3,2,1) weight structure but with different
dynamics formulation. The Generalized Forms framework applies identically.

Validation Status:
------------------
- System dynamics: Verified against Pérez-Ventura formulation
- Known Lyapunov: Coefficients from SOSTOOLS (see docs/archive/reference/pidlike_sos_reference.md)
- Framework discovery: EXPERIMENTAL - requires validation against known solution
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


class PIDSMCSystem:
    """
    PID-SMC closed-loop system.

    Default gains from Pérez-Ventura: λ₁=2.7, λ₂=5.345, λ₃=1.1
    (scaled by Δ for disturbance bound)
    """

    def __init__(
        self,
        lambda1: float = 2.7,
        lambda2: float = 5.345,
        lambda3: float = 1.1,
        delta: float = 1.0,
        L: float = 1.0,  # Lipschitz bound of disturbance derivative
    ):
        """
        Initialize PID-SMC system.

        Args:
            lambda1, lambda2, lambda3: Base controller gains
            delta: Scaling factor (set > L̄/L for stability)
            L: Nominal Lipschitz constant
        """
        self.k1 = lambda1 * delta ** (2/3)
        self.k2 = lambda2 * delta ** (1/2)
        self.k3 = lambda3 * delta
        self.delta = delta
        self.L = L

    @property
    def weights(self) -> Tuple[int, ...]:
        return (3, 2, 1)

    @property
    def kappa(self) -> float:
        return -1.0

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics (no disturbance).

        ẋ₁ = x₂
        ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
        ẋ₃ = -k₃⌈x₁⌋^0

        Args:
            x: State tensor of shape (batch, 3) or (3,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = -self.k1 * sgnpow(x1, 1/3) - self.k2 * sgnpow(x2, 1/2) + x3
        dx3 = -self.k3 * np.sign(x1)

        return np.stack([dx1, dx2, dx3], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """
        Dynamics when x₁ > 0 (sign(x₁) = +1).

        For Filippov analysis in the region x₁ > 0.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = -self.k1 * sgnpow(x1, 1/3) - self.k2 * sgnpow(x2, 1/2) + x3
        dx3 = -self.k3 * np.ones_like(x1)  # sign(x1) = +1

        return np.stack([dx1, dx2, dx3], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """
        Dynamics when x₁ < 0 (sign(x₁) = -1).

        For Filippov analysis in the region x₁ < 0.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = -self.k1 * sgnpow(x1, 1/3) - self.k2 * sgnpow(x2, 1/2) + x3
        dx3 = self.k3 * np.ones_like(x1)  # sign(x1) = -1

        return np.stack([dx1, dx2, dx3], axis=1).squeeze()


def create_pidsmc_dynamics(
    lambda1: float = 2.7,
    lambda2: float = 5.345,
    lambda3: float = 1.1,
    delta: float = 1.0,
):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    Returns:
        (dynamics_plus, dynamics_minus) callable pair
    """
    system = PIDSMCSystem(lambda1, lambda2, lambda3, delta)
    return system.dynamics_plus, system.dynamics_minus


# Default configuration
PIDSMC_WEIGHTS = (3, 2, 1)
PIDSMC_KAPPA = -1.0
PIDSMC_DEGREE = 7  # From Pérez-Ventura

# Known Lyapunov function structure (Pérez-Ventura, equation 30)
# V(x) = α₁|x₁|^(7/3) + α₂|x₂|^(7/2) + α₃|x₃|^7
#      + α₄⌈x₁⌋^(5/3)x₂ + α₅|x₁|x₂² + α₆x₁⌈x₃⌋^4
#      + α₇|x₁|^(5/3)x₃² + α₈x₂x₃⁵ + α₉|x₂|^(3/2)x₃⁴


def get_pidsmc_dynamics_gf(
    lambda1: float = 2.7,
    lambda2: float = 5.345,
    lambda3: float = 1.1,
    delta: float = 1.0,
) -> List[GeneralizedForm]:
    """
    Get PID-SMC dynamics as Generalized Forms.

    For PID-SMC:
        f₁ = ẋ₁ = x₂
        f₂ = ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
        f₃ = ẋ₃ = -k₃⌈x₁⌋^0

    where kᵢ = λᵢ · δ^(appropriate power).

    Args:
        lambda1, lambda2, lambda3: Base controller gains
        delta: Scaling factor

    Returns:
        List of GFs [f₁, f₂, f₃]
    """
    # Compute scaled gains
    k1 = lambda1 * delta ** (2/3)
    k2 = lambda2 * delta ** (1/2)
    k3 = lambda3 * delta

    # f₁ = x₂
    # Weighted degree: 3*0 + 2*1 + 1*0 = 2
    # x₂ needs 'signed' type to allow negative values: x₂ = ⌈x₂⌋^1
    f1_terms = [
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1), Fraction(0)),
            term_types=('abs', 'signed', 'abs')  # x₂ = ⌈x₂⌋^1 allows negative
        ),
    ]
    f1 = GeneralizedForm(degree_m=2, weights_r=(3, 2, 1), terms=f1_terms)

    # f₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
    # Term 1: -k₁⌈x₁⌋^(1/3), weighted degree: 3*(1/3) = 1
    # Term 2: -k₂⌈x₂⌋^(1/2), weighted degree: 2*(1/2) = 1
    # Term 3: +x₃ = ⌈x₃⌋^1, weighted degree: 1*1 = 1
    f2_terms = [
        GFTerm(
            coefficient=-k1,
            exponents=(Fraction(1, 3), Fraction(0), Fraction(0)),
            term_types=('signed', 'abs', 'abs')
        ),
        GFTerm(
            coefficient=-k2,
            exponents=(Fraction(0), Fraction(1, 2), Fraction(0)),
            term_types=('abs', 'signed', 'abs')
        ),
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(0), Fraction(1)),
            term_types=('abs', 'abs', 'signed')  # x₃ = ⌈x₃⌋^1 allows negative
        ),
    ]
    f2 = GeneralizedForm(degree_m=1, weights_r=(3, 2, 1), terms=f2_terms)

    # f₃ = -k₃⌈x₁⌋^0 (= -k₃·sign(x₁))
    # Weighted degree: 3*0 + 2*0 + 1*0 = 0
    f3_terms = [
        GFTerm(
            coefficient=-k3,
            exponents=(Fraction(0), Fraction(0), Fraction(0)),
            term_types=('signed', 'abs', 'abs')  # sign(x₁)
        ),
    ]
    f3 = GeneralizedForm(degree_m=0, weights_r=(3, 2, 1), terms=f3_terms)

    return [f1, f2, f3]


def get_pidsmc_known_lyapunov_structure() -> GeneralizedForm:
    """
    Get the structure of the known Pérez-Ventura Lyapunov function.

    Returns a degree-7 GF with the 9 terms from equation (30),
    using placeholder coefficients (1.0).

    The actual coefficients depend on the gains and are found via SOSTOOLS.

    Returns:
        GeneralizedForm with 9-term structure
    """
    # All terms have weighted degree 7 with r = (3, 2, 1)
    terms = [
        # Diagonal terms
        # α₁|x₁|^(7/3): 3*(7/3) = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(7, 3), Fraction(0), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₂|x₂|^(7/2): 2*(7/2) = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(7, 2), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₃|x₃|^7: 1*7 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(0), Fraction(7)),
            term_types=('abs', 'abs', 'abs')
        ),

        # Cross terms
        # α₄⌈x₁⌋^(5/3)x₂: 3*(5/3) + 2*1 = 5 + 2 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(5, 3), Fraction(1), Fraction(0)),
            term_types=('signed', 'signed', 'abs')
        ),
        # α₅|x₁|x₂²: 3*1 + 2*2 = 3 + 4 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1), Fraction(2), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₆x₁⌈x₃⌋^4: 3*1 + 1*4 = 3 + 4 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(1), Fraction(0), Fraction(4)),
            term_types=('signed', 'abs', 'signed')
        ),
        # α₇|x₁|^(5/3)x₃²: 3*(5/3) + 1*2 = 5 + 2 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(5, 3), Fraction(0), Fraction(2)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₈x₂x₃⁵: 2*1 + 1*5 = 2 + 5 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1), Fraction(5)),
            term_types=('abs', 'signed', 'signed')
        ),
        # α₉|x₂|^(3/2)x₃⁴: 2*(3/2) + 1*4 = 3 + 4 = 7 ✓
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(3, 2), Fraction(4)),
            term_types=('abs', 'abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=7, weights_r=(3, 2, 1), terms=terms)


def get_pidsmc_known_lyapunov_d1() -> GeneralizedForm:
    """
    Get the Pérez-Ventura Lyapunov function with D₁ set coefficients.

    From D₁ gains: λ₁=2.7, λ₂=5.345, λ₃=1.1, δ=1.0

    Full 9-coefficient set from Pérez-Ventura (private communication, verified
    via SOSTOOLS):

        α₁ = 9.966e6    |x₁|^(7/3)          (diagonal)
        α₂ = 1.032e6    |x₂|^(7/2)          (diagonal)
        α₃ = 492.5      |x₃|^7              (diagonal)
        α₄ = 1.343e7    ⌈x₁⌋^(5/3)·x₂       (cross, signed)
        α₅ = 1.291e7    |x₁|·x₂²            (cross, abs)
        α₆ = -7.652e4   x₁·⌈x₃⌋^4           (cross, signed, NEGATIVE)
        α₇ = 2.295e6    |x₁|^(5/3)·x₃²      (cross, abs)
        α₈ = -1.533e4   x₂·x₃⁵              (cross, signed, NEGATIVE)
        α₉ = 4.965e4    |x₂|^(3/2)·x₃⁴      (cross, abs)

    Note: Some cross-term coefficients are NEGATIVE. This is valid because
    the overall Lyapunov function V(x) is still positive definite when verified
    via SOS on all 8 octants.

    Returns:
        GeneralizedForm with D₁ coefficients (all 9 terms)
    """
    # Full coefficient set from Pérez-Ventura email, verified via SOSTOOLS
    terms = [
        # Diagonal terms (3 terms)
        # α₁|x₁|^(7/3): coefficient = 9.966e6
        GFTerm(
            coefficient=9.966e6,
            exponents=(Fraction(7, 3), Fraction(0), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₂|x₂|^(7/2): coefficient = 1.032e6
        GFTerm(
            coefficient=1.032e6,
            exponents=(Fraction(0), Fraction(7, 2), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₃|x₃|^7: coefficient = 492.5
        GFTerm(
            coefficient=492.5,
            exponents=(Fraction(0), Fraction(0), Fraction(7)),
            term_types=('abs', 'abs', 'abs')
        ),

        # Cross terms (6 terms)
        # α₄⌈x₁⌋^(5/3)·x₂: coefficient = 1.343e7
        GFTerm(
            coefficient=1.343e7,
            exponents=(Fraction(5, 3), Fraction(1), Fraction(0)),
            term_types=('signed', 'signed', 'abs')
        ),
        # α₅|x₁|·x₂²: coefficient = 1.291e7
        GFTerm(
            coefficient=1.291e7,
            exponents=(Fraction(1), Fraction(2), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₆·x₁·⌈x₃⌋^4: coefficient = -7.652e4 (NEGATIVE!)
        GFTerm(
            coefficient=-7.652e4,
            exponents=(Fraction(1), Fraction(0), Fraction(4)),
            term_types=('signed', 'abs', 'signed')
        ),
        # α₇|x₁|^(5/3)·x₃²: coefficient = 2.295e6
        GFTerm(
            coefficient=2.295e6,
            exponents=(Fraction(5, 3), Fraction(0), Fraction(2)),
            term_types=('abs', 'abs', 'abs')
        ),
        # α₈·x₂·x₃⁵: coefficient = -1.533e4 (NEGATIVE!)
        GFTerm(
            coefficient=-1.533e4,
            exponents=(Fraction(0), Fraction(1), Fraction(5)),
            term_types=('abs', 'signed', 'signed')
        ),
        # α₉|x₂|^(3/2)·x₃⁴: coefficient = 4.965e4
        GFTerm(
            coefficient=4.965e4,
            exponents=(Fraction(0), Fraction(3, 2), Fraction(4)),
            term_types=('abs', 'abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=7, weights_r=(3, 2, 1), terms=terms)
