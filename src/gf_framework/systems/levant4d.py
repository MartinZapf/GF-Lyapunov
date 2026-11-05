from __future__ import annotations

"""
Levant's Third-Order Differentiator (4D) System Definition.

Non-recursive form of the error dynamics (Levant 2003):
    ė₁ = -λ₁⌈e₁⌋^(3/4) + e₂
    ė₂ = -λ₂⌈e₁⌋^(1/2) + e₃
    ė₃ = -λ₃⌈e₁⌋^(1/4) + e₄
    ė₄ = -λ₄⌈e₁⌋^0

This is r-homogeneous with:
    - weights r = (4, 3, 2, 1)
    - degree κ = -1
    - cheapest C¹ Lyapunov degree: m = 5 (R2: m > max rᵢ)

Default gains are the non-recursive conversion of Levant's published recursive
set (λ̂₀..λ̂₃) = (1.1, 1.5, 2, 3) for Lipschitz constant L = 1:
    λ₁ = λ̂₃                  = 3
    λ₂ = λ̂₂·λ₁^(2/3)         ≈ 4.16
    λ₃ = λ̂₁·λ₂^(1/2)         ≈ 3.06
    λ₄ = λ̂₀                  = 1.1

Role: the n = 4 scaling demonstrator. The μ-compatible C¹ cross-term library
is the largest on the benchmark (≈97 terms at m = 8, ≈159 at m = 9, with 16
octants per solve), yet the full-library SOS path still certifies.

References:
    - Levant, A. (2003). "Higher-order sliding modes, differentiation and
      output-feedback control", Int. J. Control 76(9-10):924-941.
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


# Default configuration
LEVANT4D_WEIGHTS = (4, 3, 2, 1)
LEVANT4D_KAPPA = -1.0
LEVANT4D_DEGREE = 5  # cheapest C¹ degree (m > max r_i); discovery escalates

_L1 = 3.0
_L2 = 2.0 * _L1 ** (2.0 / 3.0)      # ≈ 4.1602
_L3 = 1.5 * _L2 ** 0.5              # ≈ 3.0595
_L4 = 1.1


class Levant4DSystem:
    """
    Levant's Third-Order Differentiator (4D system).

    ė₁ = -λ₁⌈e₁⌋^(3/4) + e₂
    ė₂ = -λ₂⌈e₁⌋^(1/2) + e₃
    ė₃ = -λ₃⌈e₁⌋^(1/4) + e₄
    ė₄ = -λ₄⌈e₁⌋^0

    eᵢ = estimation error of the (i−1)-th derivative of the signal.
    """

    def __init__(
        self,
        lambda1: float = _L1,
        lambda2: float = _L2,
        lambda3: float = _L3,
        lambda4: float = _L4,
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        if min(lambda1, lambda2, lambda3, lambda4) <= 0:
            print("Warning: All gains must be positive for stability")

    @property
    def weights(self) -> Tuple[int, ...]:
        return LEVANT4D_WEIGHTS

    @property
    def kappa(self) -> float:
        return LEVANT4D_KAPPA

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """Nominal dynamics; x of shape (batch, 4) or (4,)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        e1, e2, e3, e4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        de1 = -self.lambda1 * sgnpow(e1, 3 / 4) + e2
        de2 = -self.lambda2 * sgnpow(e1, 1 / 2) + e3
        de3 = -self.lambda3 * sgnpow(e1, 1 / 4) + e4
        de4 = -self.lambda4 * np.sign(e1)
        return np.stack([de1, de2, de3, de4], axis=1).squeeze()

    def dynamics_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when e₁ > 0 (sign(e₁) = +1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        e1, e2, e3, e4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        a1 = np.abs(e1)
        de1 = -self.lambda1 * a1 ** (3 / 4) + e2
        de2 = -self.lambda2 * a1 ** (1 / 2) + e3
        de3 = -self.lambda3 * a1 ** (1 / 4) + e4
        de4 = -self.lambda4 * np.ones_like(e1)
        return np.stack([de1, de2, de3, de4], axis=1).squeeze()

    def dynamics_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when e₁ < 0 (sign(e₁) = -1)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        e1, e2, e3, e4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        a1 = np.abs(e1)
        de1 = self.lambda1 * a1 ** (3 / 4) + e2
        de2 = self.lambda2 * a1 ** (1 / 2) + e3
        de3 = self.lambda3 * a1 ** (1 / 4) + e4
        de4 = self.lambda4 * np.ones_like(e1)
        return np.stack([de1, de2, de3, de4], axis=1).squeeze()


def create_levant4d_dynamics(
    lambda1: float = _L1,
    lambda2: float = _L2,
    lambda3: float = _L3,
    lambda4: float = _L4,
):
    """Create (dynamics_plus, dynamics_minus) for numerical simulation and tests."""
    system = Levant4DSystem(lambda1, lambda2, lambda3, lambda4)
    return system.dynamics_plus, system.dynamics_minus


def get_levant4d_dynamics_gf(
    lambda1: float = _L1,
    lambda2: float = _L2,
    lambda3: float = _L3,
    lambda4: float = _L4,
) -> List[GeneralizedForm]:
    """
    Levant 4D differentiator dynamics as Generalized Forms.

    Weighted degrees (κ + rᵢ):
        f₁: -1 + 4 = 3   (4·(3/4) = 3 and e₂ has weight 3) ✓
        f₂: -1 + 3 = 2   (4·(1/2) = 2 and e₃ has weight 2) ✓
        f₃: -1 + 2 = 1   (4·(1/4) = 1 and e₄ has weight 1) ✓
        f₄: -1 + 1 = 0   (sign(e₁) has weighted degree 0)  ✓
    """
    weights_r = LEVANT4D_WEIGHTS

    def _z():
        return Fraction(0)

    f1 = GeneralizedForm(degree_m=3, weights_r=weights_r, terms=[
        GFTerm(coefficient=-lambda1,
               exponents=(Fraction(3, 4), _z(), _z(), _z()),
               term_types=('signed', 'abs', 'abs', 'abs')),
        GFTerm(coefficient=1.0,
               exponents=(_z(), Fraction(1), _z(), _z()),
               term_types=('abs', 'signed', 'abs', 'abs')),
    ])
    f2 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=[
        GFTerm(coefficient=-lambda2,
               exponents=(Fraction(1, 2), _z(), _z(), _z()),
               term_types=('signed', 'abs', 'abs', 'abs')),
        GFTerm(coefficient=1.0,
               exponents=(_z(), _z(), Fraction(1), _z()),
               term_types=('abs', 'abs', 'signed', 'abs')),
    ])
    f3 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=[
        GFTerm(coefficient=-lambda3,
               exponents=(Fraction(1, 4), _z(), _z(), _z()),
               term_types=('signed', 'abs', 'abs', 'abs')),
        GFTerm(coefficient=1.0,
               exponents=(_z(), _z(), _z(), Fraction(1)),
               term_types=('abs', 'abs', 'abs', 'signed')),
    ])
    f4 = GeneralizedForm(degree_m=0, weights_r=weights_r, terms=[
        GFTerm(coefficient=-lambda4,
               exponents=(_z(), _z(), _z(), _z()),
               term_types=('signed', 'abs', 'abs', 'abs')),
    ])
    return [f1, f2, f3, f4]
