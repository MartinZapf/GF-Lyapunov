from __future__ import annotations

"""
Continuous Twisting Algorithm (CTA) System Definition.

From Torres-González, Sánchez, Fridman & Moreno (2017):
"Design of continuous twisting algorithm"
Automatica, 80, 119-126.

System dynamics (the nominal closed loop, eq. (8) of the paper; eq. (3) with
the disturbance derivative on the x3 channel):
    ẋ₁ = x₂
    ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
    ẋ₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0

This is r-homogeneous with:
    - weights r = (3, 2, 1)
    - degree κ = -1 (finite-time stable)
    - Lyapunov degree: m = 5

IMPORTANT: This system is a VALID Generalized Form per Definition 3 of
Sanchez-Moreno (2019). ALL terms are products of single-variable powers:
    - ⌈x₁⌋^(1/3), ⌈x₂⌋^(1/2), x₃ are standard GF terms
    - ⌈x₁⌋^0 = sign(x₁), ⌈x₂⌋^0 = sign(x₂) are discontinuous GF terms

There are NO composite/nested terms. Full SOS verification is possible.

The paper's Lyapunov function (eq. (4)) has 6 terms with degree m=5.
Its Pólya design is eqs. (14)-(15); Table 3 prints coefficient sets for its
gain set G1 (row G1, Pólya, mu = 0.44; row G1', SOS, mu = 1).

Default gains of this module: k₁ = 54.8, k₂ = 27, k₃ = 5.7, k₄ = 3. These
benchmark gains are NOT a gain set of the paper (its Table 1 lists only
G1-G4, all with k₃ = 2.3, k₄ = 1.1) and their source is unknown; the paper
has no "Lemma 1", "eq. 5" or "eq. 7". The GF paper certifies G1 =
(25, 15, 23/10, 11/10) through exact records instead
(docs/papers/gf_ijrnc/reviews/cta_redo/, CANONICAL_STATE.md
"CTA at TG2017 G1"). Correction of 2026-09-28; the values are unchanged.

Stability condition: k₃ > k₄ (necessary for unique equilibrium at origin)

References:
    - Torres-González, V., Sánchez, T., Fridman, L.M., & Moreno, J.A. (2017).
      "Design of continuous twisting algorithm" Automatica, 80, 119-126.
    - Sanchez, T., & Moreno, J.A. (2019). "Design of Lyapunov functions
      for a class of homogeneous systems" Int J Robust Nonlinear Control.
"""

from typing import Tuple, List, Optional
from fractions import Fraction
import numpy as np

from gf_framework.algebra.core import GeneralizedForm, GFTerm


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# eps is retained for call compatibility; nonnegative powers are not regularized.
def sgnpow(x: np.ndarray, p: float, eps: float = 1e-10) -> np.ndarray:
    """
    Signed power: ⌈x⌋^p = sign(x) * |x|^p

    For p = 0, returns sign(x).
    """
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


def sgnpow_numpy(x: np.ndarray, p: float, eps: float = 1e-10) -> np.ndarray:
    """Exact nonnegative signed power; eps is retained only for compatibility."""
    if p == 0:
        return np.sign(x)
    return np.sign(x) * np.abs(x) ** p


# =============================================================================
# SYSTEM CONSTANTS
# =============================================================================

CTA_WEIGHTS = (3, 2, 1)
CTA_KAPPA = -1.0
CTA_DEGREE = 5  # Lyapunov degree of the paper's eq. (4)

# Module default (benchmark) gains; not a TG2017 gain set, source unknown (see the module docstring)
CTA_DEFAULT_GAINS = {
    'k1': 54.8,
    'k2': 27.0,
    'k3': 5.7,
    'k4': 3.0,
}

# Hand-supplied Lyapunov coefficients: TG2017 Table 3, row G1 (Pólya, for G1 at mu = 0.44)
# times 0.013444; not a function designed for the default gains above (PLAN (b), 2026-09-28)
CTA_KNOWN_LYAPUNOV_COEFFICIENTS = {
    'alpha1': 430.2148,
    'alpha2': 21.2618,
    'alpha3': 2.2080,
    'alpha4': -0.1253,  # Note: negative!
    'alpha5': 0.0003,
    'alpha6': 6.5197e-6,
}


# =============================================================================
# SYSTEM CLASS
# =============================================================================

class CTASystem:
    """
    Continuous Twisting Algorithm closed-loop system.

    Dynamics (Torres-González et al. 2017, eqs. (3) and (8)):
        ẋ₁ = x₂
        ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
        ẋ₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0

    The third equation has a Twisting structure (hence "Continuous Twisting").

    Stability requires k₃ > k₄ (Theorem 1).
    """

    def __init__(
        self,
        k1: float = 54.8,
        k2: float = 27.0,
        k3: float = 5.7,
        k4: float = 3.0,
    ):
        """
        Initialize CTA system.

        Args:
            k1, k2, k3, k4: Controller gains (all must be positive, k₃ > k₄)
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

        # Validate gains
        if any(k <= 0 for k in [k1, k2, k3, k4]):
            print("Warning: All gains must be positive for stability")
        if k3 <= k4:
            print(f"Warning: Stability requires k₃ > k₄, but k₃={k3}, k₄={k4}")

    @property
    def weights(self) -> Tuple[int, ...]:
        return CTA_WEIGHTS

    @property
    def kappa(self) -> float:
        return CTA_KAPPA

    def dynamics_nominal(self, x: np.ndarray) -> np.ndarray:
        """
        Nominal dynamics (continuous part only).

        ẋ₁ = x₂
        ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
        ẋ₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0

        Args:
            x: State tensor [x₁, x₂, x₃] of shape (batch, 3) or (3,)

        Returns:
            dx/dt of same shape
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = -self.k1 * sgnpow(x1, 1/3) - self.k2 * sgnpow(x2, 1/2) + x3
        dx3 = -self.k3 * np.sign(x1) - self.k4 * np.sign(x2)

        return np.stack([dx1, dx2, dx3], axis=1).squeeze()

    def dynamics_plus_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when x₁ > 0 AND x₂ > 0."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = (-self.k1 * np.abs(x1) ** (1/3)
               - self.k2 * np.abs(x2) ** (1/2) + x3)
        dx3 = -self.k3 - self.k4  # sign(x1)=+1, sign(x2)=+1

        return np.stack([dx1, dx2, np.full_like(x1, dx3)], axis=1).squeeze()

    def dynamics_plus_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when x₁ > 0 AND x₂ < 0."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = (-self.k1 * np.abs(x1) ** (1/3)
               + self.k2 * np.abs(x2) ** (1/2) + x3)
        dx3 = -self.k3 + self.k4  # sign(x1)=+1, sign(x2)=-1

        return np.stack([dx1, dx2, np.full_like(x1, dx3)], axis=1).squeeze()

    def dynamics_minus_plus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when x₁ < 0 AND x₂ > 0."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = (self.k1 * np.abs(x1) ** (1/3)
               - self.k2 * np.abs(x2) ** (1/2) + x3)
        dx3 = self.k3 - self.k4  # sign(x1)=-1, sign(x2)=+1

        return np.stack([dx1, dx2, np.full_like(x1, dx3)], axis=1).squeeze()

    def dynamics_minus_minus(self, x: np.ndarray) -> np.ndarray:
        """Dynamics when x₁ < 0 AND x₂ < 0."""
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        dx1 = x2
        dx2 = (self.k1 * np.abs(x1) ** (1/3)
               + self.k2 * np.abs(x2) ** (1/2) + x3)
        dx3 = self.k3 + self.k4  # sign(x1)=-1, sign(x2)=-1

        return np.stack([dx1, dx2, np.full_like(x1, dx3)], axis=1).squeeze()


# =============================================================================
# DYNAMICS FACTORY FUNCTIONS
# =============================================================================

def create_cta_dynamics(
    k1: float = 54.8,
    k2: float = 27.0,
    k3: float = 5.7,
    k4: float = 3.0,
):
    """
    Create (dynamics_plus, dynamics_minus) for numerical simulation and tests.

    For CTA with discontinuities on x₁=0 AND x₂=0 hyperplanes, we return
    the nominal dynamics for both f_plus and f_minus since the sign()
    function is handled internally.

    Note: Unlike PID-SMC which has a single sign(x₁) discontinuity, CTA
    has a 2D discontinuity surface; we use the nominal dynamics, which
    handles sign() consistently.

    Returns:
        (dynamics_plus, dynamics_minus) - both are the same nominal dynamics
    """
    system = CTASystem(k1, k2, k3, k4)
    # Return tuple as expected by framework - both same for CTA
    return system.dynamics_nominal, system.dynamics_nominal


def create_cta_dynamics_regions(
    k1: float = 54.8,
    k2: float = 27.0,
    k3: float = 5.7,
    k4: float = 3.0,
):
    """
    Create region-specific dynamics for Filippov analysis.

    Returns:
        Dictionary of dynamics for each sign combination
    """
    system = CTASystem(k1, k2, k3, k4)
    return {
        ('+', '+'): system.dynamics_plus_plus,
        ('+', '-'): system.dynamics_plus_minus,
        ('-', '+'): system.dynamics_minus_plus,
        ('-', '-'): system.dynamics_minus_minus,
    }


# =============================================================================
# GENERALIZED FORM REPRESENTATIONS
# =============================================================================

def get_cta_dynamics_gf(
    k1: float = 54.8,
    k2: float = 27.0,
    k3: float = 5.7,
    k4: float = 3.0,
) -> List[GeneralizedForm]:
    """
    Get CTA dynamics as Generalized Forms.

    For CTA (Torres-González et al. 2017, eqs. (3) and (8)):
        f₁ = ẋ₁ = x₂
        f₂ = ẋ₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
        f₃ = ẋ₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0

    Weighted degrees (must equal κ + rᵢ):
        f₁: κ + r₁ = -1 + 3 = 2   [x₂ has degree 2·1 = 2 ✓]
        f₂: κ + r₂ = -1 + 2 = 1   [all terms have degree 1 ✓]
        f₃: κ + r₃ = -1 + 1 = 0   [sign terms have degree 0 ✓]

    Args:
        k1, k2, k3, k4: Controller gains

    Returns:
        List of GFs [f₁, f₂, f₃]
    """
    weights_r = CTA_WEIGHTS

    # -------------------------------------------------------------------------
    # f₁ = x₂
    # Weighted degree: r₂ * 1 = 2 * 1 = 2
    # -------------------------------------------------------------------------
    f1_terms = [
        GFTerm(
            coefficient=1.0,
            exponents=(Fraction(0), Fraction(1), Fraction(0)),
            term_types=('abs', 'signed', 'abs')
        ),
    ]
    f1 = GeneralizedForm(degree_m=2, weights_r=weights_r, terms=f1_terms)

    # -------------------------------------------------------------------------
    # f₂ = -k₁⌈x₁⌋^(1/3) - k₂⌈x₂⌋^(1/2) + x₃
    # All terms have weighted degree 1:
    #   - k₁⌈x₁⌋^(1/3): 3 * (1/3) = 1 ✓
    #   - k₂⌈x₂⌋^(1/2): 2 * (1/2) = 1 ✓
    #   - x₃: 1 * 1 = 1 ✓
    # -------------------------------------------------------------------------
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
            term_types=('abs', 'abs', 'signed')
        ),
    ]
    f2 = GeneralizedForm(degree_m=1, weights_r=weights_r, terms=f2_terms)

    # -------------------------------------------------------------------------
    # f₃ = -k₃⌈x₁⌋^0 - k₄⌈x₂⌋^0 = -k₃·sign(x₁) - k₄·sign(x₂)
    # Both terms have weighted degree 0:
    #   - k₃⌈x₁⌋^0: 3 * 0 = 0 ✓
    #   - k₄⌈x₂⌋^0: 2 * 0 = 0 ✓
    # -------------------------------------------------------------------------
    f3_terms = [
        GFTerm(
            coefficient=-k3,
            exponents=(Fraction(0), Fraction(0), Fraction(0)),
            term_types=('signed', 'abs', 'abs')  # sign(x₁)
        ),
        GFTerm(
            coefficient=-k4,
            exponents=(Fraction(0), Fraction(0), Fraction(0)),
            term_types=('abs', 'signed', 'abs')  # sign(x₂)
        ),
    ]
    f3 = GeneralizedForm(degree_m=0, weights_r=weights_r, terms=f3_terms)

    return [f1, f2, f3]


def get_cta_known_lyapunov(
    alpha1: float = 430.2148,
    alpha2: float = 21.2618,
    alpha3: float = 2.2080,
    alpha4: float = -0.1253,
    alpha5: float = 0.0003,
    alpha6: float = 6.5197e-6,
) -> GeneralizedForm:
    """
    Get the Lyapunov function of the form of Torres-González et al. (2017), eq. (4).

    V(x) = α₁|x₁|^(5/3) + α₂x₁x₂ + α₃|x₂|^(5/2)
         + α₄x₁⌈x₃⌋² - α₅x₂x₃³ + α₆|x₃|^5

    All terms have weighted degree m = 5 with weights r = (3, 2, 1):
        - α₁|x₁|^(5/3): 3 * (5/3) = 5 ✓
        - α₂x₁x₂:       3*1 + 2*1 = 5 ✓
        - α₃|x₂|^(5/2): 2 * (5/2) = 5 ✓
        - α₄x₁⌈x₃⌋²:    3*1 + 1*2 = 5 ✓
        - α₅x₂x₃³:      2*1 + 1*3 = 5 ✓
        - α₆|x₃|^5:     1 * 5 = 5 ✓

    IMPORTANT NOTES:
    - α₄ is NEGATIVE (-0.1253) - this is valid for cross-terms
    - α₅ appears with minus sign in V but coefficient is positive
    - The paper uses notation where α₅ > 0 and writes "-α₅x₂x₃³"

    Args:
        alpha1-alpha6: Lyapunov coefficients (defaults: TG2017 Table 3 row G1
            times 0.013444, see CTA_KNOWN_LYAPUNOV_COEFFICIENTS)

    Returns:
        GeneralizedForm for the Lyapunov function
    """
    weights_r = CTA_WEIGHTS
    degree_m = CTA_DEGREE

    terms = [
        # Term 1: α₁|x₁|^(5/3)
        GFTerm(
            coefficient=alpha1,
            exponents=(Fraction(5, 3), Fraction(0), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # Term 2: α₂x₁x₂ = α₂⌈x₁⌋¹⌈x₂⌋¹
        GFTerm(
            coefficient=alpha2,
            exponents=(Fraction(1), Fraction(1), Fraction(0)),
            term_types=('signed', 'signed', 'abs')
        ),
        # Term 3: α₃|x₂|^(5/2)
        GFTerm(
            coefficient=alpha3,
            exponents=(Fraction(0), Fraction(5, 2), Fraction(0)),
            term_types=('abs', 'abs', 'abs')
        ),
        # Term 4: α₄x₁⌈x₃⌋² = α₄⌈x₁⌋¹⌈x₃⌋²
        # In the GF convention ⌈x⌋^a = |x|^a sign(x) for EVERY a, so
        # ⌈x₃⌋² = |x₃|² sign(x₃), which is 'signed', not 'abs'. The earlier
        # comment ("signed squared = absolute") read ⌈·⌋ as an ordinary power
        # and mis-typed this term; corrected 2026-09-03 (P5-01).
        GFTerm(
            coefficient=alpha4,
            exponents=(Fraction(1), Fraction(0), Fraction(2)),
            term_types=('signed', 'abs', 'signed')
        ),
        # Term 5: -α₅x₂x₃³ = -α₅⌈x₂⌋¹⌈x₃⌋³
        GFTerm(
            coefficient=-alpha5,
            exponents=(Fraction(0), Fraction(1), Fraction(3)),
            term_types=('abs', 'signed', 'signed')
        ),
        # Term 6: α₆|x₃|^5
        GFTerm(
            coefficient=alpha6,
            exponents=(Fraction(0), Fraction(0), Fraction(5)),
            term_types=('abs', 'abs', 'abs')
        ),
    ]

    return GeneralizedForm(degree_m=degree_m, weights_r=weights_r, terms=terms)


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================

def verify_homogeneity():
    """
    Verify the r-homogeneity property of CTA.

    For weights r = (3, 2, 1) and degree κ = -1:
        fᵢ(Λ_ε^r x) = ε^(κ+rᵢ) fᵢ(x)

    where Λ_ε^r = diag(ε³, ε², ε).
    """
    explanation = """
    CTA Homogeneity Verification (Torres-González 2017):

    Weights: r = (3, 2, 1)
    Degree: κ = -1 (finite-time stable)

    Dilation: Λ_ε^r x = (ε³x₁, ε²x₂, εx₃)

    Verify each component:

    f₁(Λ_ε^r x) = ε²x₂
                = ε² · f₁(x)
                = ε^(κ+r₁) · f₁(x)  [κ+r₁ = -1+3 = 2] ✓

    f₂(Λ_ε^r x) = -k₁⌈ε³x₁⌋^(1/3) - k₂⌈ε²x₂⌋^(1/2) + εx₃
                = -k₁·ε·⌈x₁⌋^(1/3) - k₂·ε·⌈x₂⌋^(1/2) + ε·x₃
                = ε · f₂(x)
                = ε^(κ+r₂) · f₂(x)  [κ+r₂ = -1+2 = 1] ✓

    f₃(Λ_ε^r x) = -k₃·sign(ε³x₁) - k₄·sign(ε²x₂)
                = -k₃·sign(x₁) - k₄·sign(x₂)  [sign is scale-invariant]
                = ε⁰ · f₃(x)
                = ε^(κ+r₃) · f₃(x)  [κ+r₃ = -1+1 = 0] ✓

    The system is r-homogeneous of degree κ = -1.
    Since κ < 0, the system exhibits FINITE-TIME STABILITY.
    """
    return explanation
