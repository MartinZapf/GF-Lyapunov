from __future__ import annotations

"""
Levant's Fourth-Order Differentiator (5D) System Definition.

Non-recursive form of the error dynamics (Levant 2003):
    ė₁ = -λ₁⌈e₁⌋^(4/5) + e₂
    ė₂ = -λ₂⌈e₁⌋^(3/5) + e₃
    ė₃ = -λ₃⌈e₁⌋^(2/5) + e₄
    ė₄ = -λ₄⌈e₁⌋^(1/5) + e₅
    ė₅ = -λ₅⌈e₁⌋^0

This is r-homogeneous with:
    - weights r = (5, 4, 3, 2, 1)
    - degree κ = -1
    - cheapest C¹ Lyapunov degree: m = 6 (R2: m > max rᵢ)

Default gains are the non-recursive conversion of Levant's published recursive
set (λ̂₀..λ̂₄) = (1.1, 1.5, 2, 3, 5) for Lipschitz constant L = 1 (same
conversion pattern as the 4D module: λ₁ = λ̂₄; λᵢ = λ̂₅₋ᵢ·λᵢ₋₁^((5−i)/(6−i))):
    λ₁ = λ̂₄                  = 5
    λ₂ = λ̂₃·λ₁^(3/4)         ≈ 10.03
    λ₃ = λ̂₂·λ₂^(2/3)         ≈ 9.30
    λ₄ = λ̂₁·λ₃^(1/2)         ≈ 4.58
    λ₅ = λ̂₀                  = 1.1

Role: the n = 5 scale-boundary probe (NOT part of the benchmark matrix). At
m = 6 the per-octant dense Gram bound is C(10,4) = 210 rows over 32 octants;
whether the μ-filtered full-library path stays tractable here is exactly the
question the journal's scale paragraph answers.

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
LEVANT5D_WEIGHTS = (5, 4, 3, 2, 1)
LEVANT5D_KAPPA = -1.0
LEVANT5D_DEGREE = 6  # cheapest C¹ degree (m > max r_i); discovery escalates

_L1 = 5.0
_L2 = 3.0 * _L1 ** (3.0 / 4.0)      # ≈ 10.0312
_L3 = 2.0 * _L2 ** (2.0 / 3.0)      # ≈ 9.3042
_L4 = 1.5 * _L3 ** 0.5              # ≈ 4.5752
_L5 = 1.1


class Levant5DSystem:
    """
    Levant's Fourth-Order Differentiator (5D system).

    ė₁ = -λ₁⌈e₁⌋^(4/5) + e₂
    ė₂ = -λ₂⌈e₁⌋^(3/5) + e₃
    ė₃ = -λ₃⌈e₁⌋^(2/5) + e₄
    ė₄ = -λ₄⌈e₁⌋^(1/5) + e₅
    ė₅ = -λ₅⌈e₁⌋^0

    eᵢ = estimation error of the (i−1)-th derivative of the signal.
    """

    def __init__(
        self,
        lambda1: float = _L1,
        lambda2: float = _L2,
        lambda3: float = _L3,
        lambda4: float = _L4,
        lambda5: float = _L5,
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        if min(lambda1, lambda2, lambda3, lambda4, lambda5) <= 0:
            print("Warning: All gains must be positive for stability")

    @property
    def weights(self) -> Tuple[int, ...]:
        return LEVANT5D_WEIGHTS

    @property
    def kappa(self) -> float:
        return LEVANT5D_KAPPA

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """Nominal dynamics; x of shape (batch, 5) or (5,)."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        e1, e2, e3, e4, e5 = (x[:, i] for i in range(5))
        de1 = -self.lambda1 * sgnpow(e1, 4 / 5) + e2
        de2 = -self.lambda2 * sgnpow(e1, 3 / 5) + e3
        de3 = -self.lambda3 * sgnpow(e1, 2 / 5) + e4
        de4 = -self.lambda4 * sgnpow(e1, 1 / 5) + e5
        de5 = -self.lambda5 * np.sign(e1)
        return np.stack([de1, de2, de3, de4, de5], axis=1).squeeze()


def get_levant5d_dynamics_gf(
    lambda1: float = _L1,
    lambda2: float = _L2,
    lambda3: float = _L3,
    lambda4: float = _L4,
    lambda5: float = _L5,
) -> List[GeneralizedForm]:
    """
    Levant 5D differentiator dynamics as Generalized Forms.

    Weighted degrees (κ + rᵢ):
        f₁: -1 + 5 = 4   (5·(4/5) = 4 and e₂ has weight 4) ✓
        f₂: -1 + 4 = 3   (5·(3/5) = 3 and e₃ has weight 3) ✓
        f₃: -1 + 3 = 2   (5·(2/5) = 2 and e₄ has weight 2) ✓
        f₄: -1 + 2 = 1   (5·(1/5) = 1 and e₅ has weight 1) ✓
        f₅: -1 + 1 = 0   (sign(e₁) has weighted degree 0)  ✓
    """
    weights_r = LEVANT5D_WEIGHTS

    def _z():
        return Fraction(0)

    def _e1_term(coeff: float, p: Fraction) -> GFTerm:
        return GFTerm(coefficient=coeff,
                      exponents=(p, _z(), _z(), _z(), _z()),
                      term_types=('signed', 'abs', 'abs', 'abs', 'abs'))

    def _chain_term(var_idx: int) -> GFTerm:
        exps = [_z()] * 5
        exps[var_idx] = Fraction(1)
        types = ['abs'] * 5
        types[var_idx] = 'signed'
        return GFTerm(coefficient=1.0, exponents=tuple(exps),
                      term_types=tuple(types))

    f1 = GeneralizedForm(degree_m=4, weights_r=weights_r, terms=[
        _e1_term(-lambda1, Fraction(4, 5)), _chain_term(1),
    ])
    f2 = GeneralizedForm(degree_m=3, weights_r=weights_r, terms=[
        _e1_term(-lambda2, Fraction(3, 5)), _chain_term(2),
    ])
    f3 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=[
        _e1_term(-lambda3, Fraction(2, 5)), _chain_term(3),
    ])
    f4 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=[
        _e1_term(-lambda4, Fraction(1, 5)), _chain_term(4),
    ])
    f5 = GeneralizedForm(degree_m=0, weights_r=weights_r, terms=[
        _e1_term(-lambda5, Fraction(0)),
    ])
    return [f1, f2, f3, f4, f5]
