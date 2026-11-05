from __future__ import annotations

"""
FC-CTA (Fully-Connected Continuous Twisting Algorithm) System Definition.

NOVEL CONTROLLER — extends CTA with a third sign(x₃) relay term.

Dynamics:
    ẋ₁ = x₂
    ẋ₂ = -k₁⌈x₁⌋^{1/3} - k₂⌈x₂⌋^{1/2} + x₃
    ẋ₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0 - k₅⌈x₃⌋^0       [novel: -k₅·sign(x₃)]

r-homogeneous with r = (3, 2, 1), κ = -1, Lyapunov degree m = 5.
Stability requires k₃ > k₄ > 0 and k₅ > 0.

References:
    Torres-González, V., Fridman, L., Moreno, J. (2017). "Design of
    Continuous Twisting Algorithm." Automatica 80, 119-126.
"""

from typing import List, Tuple
from fractions import Fraction
import numpy as np

from gf_framework.algebra.core import GeneralizedForm, GFTerm


# eps is retained for call compatibility; nonnegative powers are not regularized.
def sgnpow(x: np.ndarray, p: float, eps: float = 1e-10) -> np.ndarray:
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


FC_CTA_WEIGHTS = (3, 2, 1)
FC_CTA_KAPPA = -1.0
FC_CTA_DEGREE = 5

FC_CTA_DEFAULT_GAINS = {
    'k1': 54.8, 'k2': 27.0, 'k3': 5.7, 'k4': 3.0, 'k5': 2.0,
}


class FCCTASystem:
    """Fully-Connected CTA closed-loop system."""

    def __init__(self, k1: float = 54.8, k2: float = 27.0,
                 k3: float = 5.7, k4: float = 3.0, k5: float = 2.0):
        self.k1, self.k2, self.k3, self.k4, self.k5 = k1, k2, k3, k4, k5
        if not (k3 > k4 > 0 and k5 > 0):
            print(f"Warning: FC-CTA stability requires k₃>k₄>0, k₅>0 "
                  f"(got k₃={k3}, k₄={k4}, k₅={k5})")

    @property
    def weights(self) -> Tuple[int, ...]:
        return FC_CTA_WEIGHTS

    @property
    def kappa(self) -> float:
        return FC_CTA_KAPPA

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        d1 = x2
        d2 = -self.k1 * sgnpow(x1, 1/3) - self.k2 * sgnpow(x2, 1/2) + x3
        d3 = (-self.k3 * np.sign(x1)
              - self.k4 * np.sign(x2)
              - self.k5 * np.sign(x3))
        return np.stack([d1, d2, d3], axis=1).squeeze()


def create_fc_cta_dynamics(k1: float = 54.8, k2: float = 27.0,
                            k3: float = 5.7, k4: float = 3.0,
                            k5: float = 2.0):
    """Create dynamics_plus / dynamics_minus pair (both = nominal for CTA family)."""
    system = FCCTASystem(k1, k2, k3, k4, k5)
    return system.dynamics_nominal, system.dynamics_nominal


def get_fc_cta_dynamics_gf(k1: float = 54.8, k2: float = 27.0,
                            k3: float = 5.7, k4: float = 3.0,
                            k5: float = 2.0) -> List[GeneralizedForm]:
    """FC-CTA dynamics as Generalized Forms.

    Weighted degrees per equation (must equal κ + rᵢ):
        f₁: deg = 2 (κ + r₁ = -1 + 3)
        f₂: deg = 1 (κ + r₂ = -1 + 2)
        f₃: deg = 0 (κ + r₃ = -1 + 1)
    """
    weights_r = FC_CTA_WEIGHTS

    f1_terms = [
        GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(1), Fraction(0)),
               term_types=('abs', 'signed', 'abs')),
    ]
    f1 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=f1_terms)

    f2_terms = [
        GFTerm(coefficient=-k1, exponents=(Fraction(1, 3), Fraction(0), Fraction(0)),
               term_types=('signed', 'abs', 'abs')),
        GFTerm(coefficient=-k2, exponents=(Fraction(0), Fraction(1, 2), Fraction(0)),
               term_types=('abs', 'signed', 'abs')),
        GFTerm(coefficient=1.0, exponents=(Fraction(0), Fraction(0), Fraction(1)),
               term_types=('abs', 'abs', 'signed')),
    ]
    f2 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=f2_terms)

    f3_terms = [
        GFTerm(coefficient=-k3, exponents=(Fraction(0), Fraction(0), Fraction(0)),
               term_types=('signed', 'abs', 'abs')),  # sign(x₁)
        GFTerm(coefficient=-k4, exponents=(Fraction(0), Fraction(0), Fraction(0)),
               term_types=('abs', 'signed', 'abs')),  # sign(x₂)
        GFTerm(coefficient=-k5, exponents=(Fraction(0), Fraction(0), Fraction(0)),
               term_types=('abs', 'abs', 'signed')),  # sign(x₃)  — NOVEL
    ]
    f3 = GeneralizedForm(degree_m=0, weights_r=weights_r, terms=f3_terms)

    return [f1, f2, f3]


def get_fc_cta_known_lyapunov() -> None:
    """No closed-form Lyapunov reference exists — novel controller."""
    return None
